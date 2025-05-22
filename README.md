## CatVTON Training Code

This repository contains the training code for **CatVTON**, a category-level virtual try-on framework. The model leverages pre-trained components provided by the original authors to synthesize realistic try-on results for garments such as dresses, tops, and outerwear. By following this guide, you can fine-tune the model using the **DressCode** dataset or other compatible datasets.

---

## 1. How to Run the Code

Follow the steps below to set up your environment and run the training pipeline:

### 1. Install Required Dependencies

Before running the code, ensure that all necessary Python packages are installed. You can install the dependencies listed in the `requirements.txt` file by executing:

```bash
pip install -r requirements.txt
```

> 💡 Make sure you're using a compatible Python environment (Python 3.7+ is recommended) with GPU support for optimal training performance.

---

### 2. Prepare the Dataset

* Download and extract the **DressCode** dataset or your preferred dataset.
* Place the extracted dataset folder in the same directory as this repository (or provide the correct path when running the script).
* Ensure the data directory contains the expected subfolders (e.g., `image`, `cloth`, `agnostic`, etc.), as required by the CatVTON pipeline.

Example folder structure:

```
CatVTON/
├── main.py
├── requirements.txt
├── your_dataset_folder/
│   ├── image/
│   ├── cloth/
│   ├── agnostic/
│   └── ... (other necessary folders)
```

---

### 3. Start Training the Model

To begin training or fine-tuning the CatVTON model, run the following command from the root directory:

```bash
python main.py --data_root <path_to_dataset_folder> --output_dir <path_to_save_outputs>
```

* `--data_root`: Path to the dataset directory (e.g., `./your_dataset_folder`)
* `--output_dir`: Path where the training results, logs, and checkpoints will be saved

Example usage:

```bash
python main.py --data_root ./DressCode --output_dir ./output
```

---

## 📂 Output

After training, the output directory will contain:

* Trained model checkpoints
* Intermediate results (e.g., synthesized images)

---
