import torch
import torch.nn as nn
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel, AutoencoderKL
from diffusers import DDIMScheduler, DDPMScheduler
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import argparse
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from accelerate import Accelerator
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

class CatVTON(nn.Module):
    def __init__(self, pretrained_model_path="stable-diffusion-v1-5/stable-diffusion-inpainting"):
        super().__init__()
        # Load pre-trained components
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
        
        # Freeze all parameters by default
        for param in self.parameters():
            param.requires_grad = False
        
        # Unfreeze only self-attention modules (focused parameter-efficient training)
        for name, module in self.unet.named_modules():
            if "attn1" in name:  # self-attention module
                for param in module.parameters():
                    param.requires_grad = True
        
        # Remove text encoder by setting dummy embeddings
        self.dummy_text_embeds = nn.Parameter(torch.zeros(1, 77, 768), requires_grad=False)
        
        # Disable cross-attention functionality
        self._disable_cross_attention()
    
    def _disable_cross_attention(self):
        # Modify cross-attention modules to skip text conditioning
        for name, module in self.unet.named_modules():
            if "attn2" in name:  # cross-attention module
                # Create a forward pass that returns identity mapping
                def skip_cross_attn(hidden_states, encoder_hidden_states=None, attention_mask=None):
                    return hidden_states
                
                module.forward = skip_cross_attn
    
    def encode_images(self, images):
        # Encode images to latent space
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215  # scaling factor for SD VAE
        return latents
    
    def decode_latents(self, latents):
        # Decode latents to images
        with torch.no_grad():
            images = self.vae.decode(latents / 0.18215).sample
        return images
    
    def forward(self, noisy_latents, timesteps, person_latents, garment_latents, mask):
        # Concatenate person and garment latents in spatial dimension (as in paper)
        Xc = torch.cat([person_latents, garment_latents], dim=2)  # concat along height dimension
        
        # Concatenate mask (cloth-agnostic mask for person and all-zero mask for garment)
        mc = torch.cat([mask, torch.zeros_like(mask)], dim=2)
        
        # Concatenate along channel dimension for UNet input
        model_input = torch.cat([noisy_latents, mc, Xc], dim=1)
        
        # Forward pass through UNet
        noise_pred = self.unet(model_input, timesteps, encoder_hidden_states=self.dummy_text_embeds)
        
        return noise_pred.sample

def dream_training_step(model, batch, noise_scheduler, dream_lambda=10, device="cuda"):
    # Get batch components
    cloth_agnostic_person = batch['cloth_agnostic_person'].to(device)
    garment = batch['garment'].to(device)
    mask = batch['mask'].to(device)
    target_person = batch['person'].to(device)  # Ground truth with garment
    
    # Encode images to latent space
    with torch.no_grad():
        person_latents = model.encode_images(cloth_agnostic_person)
        garment_latents = model.encode_images(garment)
        target_latents = model.encode_images(target_person)
    
    # Sample noise and timesteps
    batch_size = person_latents.shape[0]
    noise = torch.randn_like(target_latents)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device)
    
    # Add noise to target latents
    noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)
    
    # First forward pass to get noise prediction
    model_pred = model(noisy_latents, timesteps, person_latents, garment_latents, mask)
    
    # Create modified noise with model predictions (DREAM strategy)
    dream_noise = noise + dream_lambda * model_pred
    
    # Get alpha_t for the timesteps
    alpha_t = noise_scheduler.alphas_cumprod[timesteps]
    alpha_t = alpha_t.view(-1, 1, 1, 1)
    
    # Create dream latents
    dream_latents = torch.sqrt(alpha_t) * target_latents + torch.sqrt(1 - alpha_t) * dream_noise
    
    # Second forward pass with dream latents
    dream_pred = model(dream_latents, timesteps, person_latents, garment_latents, mask)
    
    # Compute loss (MSE between dream_noise and predicted noise)
    loss = F.mse_loss(dream_noise, dream_pred)
    
    return loss

class DressCodeDataset(Dataset):
    def __init__(self, data_root, split='train', img_size=(512, 384)):
        self.data_root = os.path.join(data_root, "dresses")  # Focus on dresses category
        self.split = split
        self.img_size = img_size
        
        # Define image directories based on dataset structure
        self.image_dir = os.path.join(self.data_root, 'images')
        self.garment_dir = os.path.join(self.data_root, 'images')
        self.mask_dir = os.path.join(self.data_root, 'label_maps')
        
        # Get image pairs from category-specific train file
        self.image_pairs = self._get_image_pairs()
        
        # Define transforms (same as before)
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def _get_image_pairs(self):
        # Load category-specific train pairs
        pair_file = os.path.join(self.data_root, f"{self.split}_pairs.txt")
        pairs = []
        
        with open(pair_file, 'r') as f:
            for line in f:
                # Format: model_image.jpg garment_image.jpg
                model_name, garment_name = line.strip().split()
                pair_id = model_name.split('_')[0]  # Extract base ID
                
                # Verify existence of all required files
                if (os.path.exists(os.path.join(self.image_dir, model_name)) and
                    os.path.exists(os.path.join(self.garment_dir, garment_name)) and
                    os.path.exists(os.path.join(self.mask_dir, f"{pair_id}_4.png"))):
                    pairs.append((model_name, garment_name))
        return pairs

    def __getitem__(self, idx):
        model_name, garment_name = self.image_pairs[idx]
        pair_id = model_name.split('_')[0]

        # Load images with category-specific naming
        person_img = Image.open(os.path.join(self.image_dir, model_name)).convert('RGB')
        garment_img = Image.open(os.path.join(self.garment_dir, garment_name)).convert('RGB')
        mask_img = Image.open(os.path.join(self.mask_dir, f"{pair_id}_4.png")).convert('L')

        # Apply transforms
        person_tensor = self.transform(person_img)
        garment_tensor = self.transform(garment_img)
        mask_tensor = self.mask_transform(mask_img)

        # Create cloth-agnostic person image using the mask
        cloth_agnostic_person = person_tensor * (mask_tensor == 7).float()  # Dress class=7

        return {
            'person': person_tensor,
            'cloth_agnostic_person': cloth_agnostic_person,
            'garment': garment_tensor,
            'mask': (mask_tensor == 7).float(),  # Binary mask for dress region
            'pair_id': pair_id
        }


def train_catvton():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of the Dress Code dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_train_steps", type=int, default=16000, help="Number of training steps")
    parser.add_argument("--img_size", nargs=2, type=int, default=[512, 384], help="Image size for training")
    parser.add_argument("--dream_lambda", type=float, default=10.0, help="Lambda parameter for DREAM")
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="Conditional dropout probability")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every x steps")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Run evaluation every x steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Setup accelerator for distributed training
    accelerator = Accelerator()
    device = accelerator.device
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Initialize model
    model = CatVTON(pretrained_model_path="stable-diffusion-v1-5/stable-diffusion-inpainting")
    
    # Setup optimizer - only optimize self-attention parameters
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate
    )
    
    # Setup noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False
    )
    
    # Create dataset and dataloader
    train_dataset = DressCodeDataset(
        data_root=args.data_root,
        split='train',
        img_size=tuple(args.img_size)
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Prepare model, optimizer and dataloader for distributed training
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    
    # Training loop
    global_step = 0
    progress_bar = tqdm(
        range(args.num_train_steps),
        disable=not accelerator.is_local_main_process
    )
    
    # Create output directory
    if accelerator.is_local_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    model.train()
    train_dataloader_iter = iter(train_dataloader)
    
    while global_step < args.num_train_steps:
        # Get batch
        try:
            batch = next(train_dataloader_iter)
        except StopIteration:
            train_dataloader_iter = iter(train_dataloader)
            batch = next(train_dataloader_iter)
        
        # Training step with DREAM strategy
        cloth_agnostic_person = batch['cloth_agnostic_person']
        garment = batch['garment']
        mask = batch['mask']
        person = batch['person']  # Ground truth
        
        # Apply conditional dropout for classifier-free guidance training
        if args.dropout_prob > 0:
            mask_dropout = torch.bernoulli(torch.ones_like(mask) * (1 - args.dropout_prob))
            mask = mask * mask_dropout
        
        # Perform DREAM training step
        loss = dream_training_step(
            model=model,
            batch=batch,
            noise_scheduler=noise_scheduler,
            dream_lambda=args.dream_lambda,
            device=device
        )
        
        # Backward and optimize
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        
        # Update progress
        global_step += 1
        progress_bar.update(1)
        
        # Log loss
        if global_step % 10 == 0:
            progress_bar.set_postfix(loss=loss.item())
        
        # Save checkpoint
        if global_step % args.save_steps == 0 and accelerator.is_local_main_process:
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            os.makedirs(save_path, exist_ok=True)
            
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(save_path)
            
            # Save optimizer state
            torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
            
            print(f"Saved model checkpoint to {save_path}")
        
        # Run evaluation
        if global_step % args.eval_steps == 0 and accelerator.is_local_main_process:
            evaluate_model(model, args)
    
    # Save final model
    if accelerator.is_local_main_process:
        save_path = os.path.join(args.output_dir, "final_model")
        os.makedirs(save_path, exist_ok=True)
        
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_path)
        
        print(f"Saved final model to {save_path}")

import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import lpips

def evaluate_model(model, args):
    model.eval()
    
    # Create evaluation dataset and dataloader
    eval_dataset = DressCodeDataset(
        data_root=args.data_root,
        split='test',
        img_size=tuple(args.img_size)
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=8,  # Smaller batch size for evaluation
        shuffle=False,
        num_workers=4
    )
    
    # Initialize metrics
    fid = FrechetInceptionDistance(feature=2048).to(model.device)
    kid = KernelInceptionDistance(feature=2048).to(model.device)
    ssim = StructuralSimilarityIndexMeasure().to(model.device)
    lpips_fn = lpips.LPIPS(net='alex').to(model.device)
    
    # Create directories for saving generated images
    eval_dir = os.path.join(args.output_dir, f"eval_step_{global_step}")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Evaluation loop
    total_lpips = 0.0
    total_ssim = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            cloth_agnostic_person = batch['cloth_agnostic_person'].to(model.device)
            garment = batch['garment'].to(model.device)
            mask = batch['mask'].to(model.device)
            real_images = batch['person'].to(model.device)
            
            # Generate images with CatVTON
            generated_images = inference(
                model, 
                cloth_agnostic_person, 
                garment, 
                mask, 
                guidance_scale=2.5,  # Default value from paper
                num_inference_steps=50
            )
            
            # Convert images to the format expected by metrics
            real_images_uint8 = (real_images * 255).type(torch.uint8)
            generated_images_uint8 = (generated_images * 255).type(torch.uint8)
            
            # Update FID
            fid.update(real_images_uint8, real=True)
            fid.update(generated_images_uint8, real=False)
            
            # Update KID
            kid.update(real_images_uint8, real=True)
            kid.update(generated_images_uint8, real=False)
            
            # Calculate SSIM
            ssim_value = ssim(generated_images, real_images)
            total_ssim += ssim_value.item() * real_images.size(0)
            
            # Calculate LPIPS
            lpips_value = lpips_fn(generated_images, real_images).mean()
            total_lpips += lpips_value.item() * real_images.size(0)
            
            num_samples += real_images.size(0)
            
            # Save some generated images for visual inspection
            if i < 5:  # Save first 5 batches
                for j in range(min(4, real_images.size(0))):  # Save up to 4 images per batch
                    comparison = torch.cat([
                        cloth_agnostic_person[j],
                        garment[j],
                        generated_images[j],
                        real_images[j]
                    ], dim=2)
                    vutils.save_image(
                        comparison,
                        os.path.join(eval_dir, f"sample_{i}_{j}.png"),
                        normalize=True
                    )
    
    # Calculate final metrics
    fid_score = fid.compute().item()
    kid_mean, kid_std = kid.compute()
    avg_ssim = total_ssim / num_samples
    avg_lpips = total_lpips / num_samples
    
    # Print results
    print(f"Evaluation Results:")
    print(f"FID: {fid_score:.4f}")
    print(f"KID: {kid_mean.item():.4f} ± {kid_std.item():.4f}")
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"LPIPS: {avg_lpips:.4f}")
    
    # Save metrics to file
    with open(os.path.join(eval_dir, "metrics.txt"), "w") as f:
        f.write(f"FID: {fid_score:.4f}\n")
        f.write(f"KID: {kid_mean.item():.4f} ± {kid_std.item():.4f}\n")
        f.write(f"SSIM: {avg_ssim:.4f}\n")
        f.write(f"LPIPS: {avg_lpips:.4f}\n")

def inference(model, cloth_agnostic_person, garment, mask, guidance_scale=2.5, num_inference_steps=50):
    model.eval()
    
    # Encode images to latent space
    with torch.no_grad():
        person_latents = model.encode_images(cloth_agnostic_person)
        garment_latents = model.encode_images(garment)
    
    # Initialize scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False
    )
    
    # Set number of inference steps
    scheduler.set_timesteps(num_inference_steps)
    
    # Initialize latent with random noise
    latents = torch.randn_like(person_latents).to(person_latents.device)
    
    # Denoising loop
    for t in scheduler.timesteps:
        # Spatial concatenation as described in the paper
        Xc = torch.cat([person_latents, garment_latents], dim=2)
        mc = torch.cat([mask, torch.zeros_like(mask)], dim=2)
        
        # For classifier-free guidance, we need two forward passes
        with torch.no_grad():
            # Unconditional forward pass (with zero conditioning)
            zero_Xc = torch.zeros_like(Xc)
            zero_mc = torch.zeros_like(mc)
            unconditional_input = torch.cat([latents, zero_mc, zero_Xc], dim=1)
            unconditional_pred = model.unet(unconditional_input, t, encoder_hidden_states=model.dummy_text_embeds)
            
            # Conditional forward pass
            conditional_input = torch.cat([latents, mc, Xc], dim=1)
            conditional_pred = model.unet(conditional_input, t, encoder_hidden_states=model.dummy_text_embeds)
        
        # Apply classifier-free guidance
        pred = unconditional_pred.sample + guidance_scale * (conditional_pred.sample - unconditional_pred.sample)
        
        # Scheduler step
        latents = scheduler.step(pred, t, latents).prev_sample
    
    # Extract person part from latents
    person_result_latents = latents[:, :, :person_latents.shape[2], :]
    
    # Decode latents to image
    with torch.no_grad():
        images = model.decode_latents(person_result_latents)
    
    # Normalize images
    images = (images / 2 + 0.5).clamp(0, 1)
    
    return images

if __name__ == "__main__":
    train_catvton()
