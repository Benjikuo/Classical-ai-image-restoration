# Classical-ai-image-restoration

![License](https://img.shields.io/badge/License-MIT-yellow)
![Language](https://img.shields.io/badge/Language-Python-blue)

Rule-based classical image processing pipeline for restoring and evaluating degraded AI-generated images.

This project uses **40 clean AI-generated images** as reference images, creates **200 degraded images**, restores them using **classical image processing methods**, and evaluates the results with **MSE, PSNR, and SSIM**.

<br>

## 🛠️ Why I Built This

* This project was built for an **Image Processing final project**.
* The goal is to test whether classical image processing methods can restore common degradations in AI-generated images.
* Instead of using a deep learning restoration model, this project uses explainable and reproducible image processing techniques.
* The dataset is controlled because the clean reference images and degradation types are known.

<br>

## 🎯 Project Goal

This project investigates the following research questions:

1. Can classical image processing techniques restore common degradations in AI-generated images?
2. Can a rule-based diagnosis system automatically select suitable restoration methods?
3. Which degradation type is easiest or hardest to restore?

<br>

## 🧩 Features

* 🖼️ **AI-Generated Reference Images** – Uses 40 clean images generated locally with ComfyUI + FLUX.1.
* 🧪 **Controlled Degradation** – Creates five types of degraded images from each clean image.
* 🧠 **Rule-Based Diagnosis** – Detects image problems using brightness, contrast, sharpness, noise, and color shift scores.
* 🛠️ **Classical Restoration Methods** – Applies gamma correction, CLAHE, unsharp masking, median filtering, and gray-world correction.
* 📊 **Quantitative Evaluation** – Measures restoration quality using MSE, PSNR, and SSIM.
* 🔁 **Reproducible Pipeline** – Uses a fixed random seed for consistent Gaussian noise generation.
* 📁 **Metadata Tracking** – Stores raw image information, degradation metadata, restoration metadata, and evaluation results.

<br>

## 📂 Project Structure

```text
Classical-ai-image-restoration/
├── dataset/
│   ├── raw/                     # 40 clean AI-generated reference images
│   ├── degraded/                # 200 degraded images
│   │   ├── dark/
│   │   ├── low_contrast/
│   │   ├── blurry/
│   │   ├── noisy/
│   │   └── color_shift/
│   ├── restored/                # 200 restored images
│   ├── comparisons/             # Reference / degraded / restored comparison images
│   ├── metadata.csv             # Raw image filename and prompt information
│   ├── degraded_metadata.csv    # Degraded image metadata
│   ├── restored_metadata.csv    # Restored image metadata and diagnosis results
│   ├── metrics.csv              # Image-level evaluation results
│   └── summary_metrics.csv      # Average results by degradation type
│
├── src/
│   ├── main.py                  # Runs the full pipeline
│   ├── generate_degraded.py     # Creates degraded images
│   ├── diagnose.py              # Diagnoses image problems using rule-based metrics
│   ├── restore.py               # Applies restoration methods
│   └── evaluate.py              # Calculates MSE, PSNR, and SSIM
│
├── report/
│   └── report.pdf               # Final project report
│
├── presentation/
│   └── presentation.pptx        # Final presentation slides
│
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT license
└── README.md                    # Project documentation
```

<br>

## ⚙️ Requirements

Install dependencies before running:

```bash
pip install -r requirements.txt
```

Required Python packages:

```text
opencv-python
numpy
pandas
```

<br>

## ▶️ How to Run

1. Make sure the clean reference images are placed in:

```text
dataset/raw/
```

2. Make sure `metadata.csv` is placed in:

```text
dataset/metadata.csv
```

3. The expected raw image filenames are:

```text
raw_0001.png
raw_0002.png
...
raw_0040.png
```

4. Run the full pipeline:

```bash
python src/main.py
```

5. After execution, the program will generate:

```text
dataset/degraded/
dataset/restored/
dataset/comparisons/
dataset/degraded_metadata.csv
dataset/restored_metadata.csv
dataset/metrics.csv
dataset/summary_metrics.csv
```

<br>

## 🔄 Processing Pipeline

```text
Clean AI-generated images
        ↓
Generate degraded images
        ↓
Rule-based problem diagnosis
        ↓
Apply restoration method
        ↓
Evaluate MSE, PSNR, and SSIM
```

<br>

## 🧪 Degradation Types

Each clean image is transformed into five degraded versions:

| Degradation Type | Description                   | Method               |
| ---------------- | ----------------------------- | -------------------- |
| Dark             | Simulates underexposure       | Intensity scaling    |
| Low Contrast     | Reduces dynamic range         | Contrast compression |
| Blurry           | Simulates out-of-focus images | Gaussian blur        |
| Noisy            | Adds random image noise       | Gaussian noise       |
| Color Shift      | Changes color balance         | BGR channel scaling  |

Total degraded images:

```text
40 clean images × 5 degradation types = 200 degraded images
```

<br>

## 🧠 Rule-Based Diagnosis

The system diagnoses each degraded image using five image quality indicators:

| Metric      | Purpose                             |
| ----------- | ----------------------------------- |
| Brightness  | Detects dark or underexposed images |
| Contrast    | Detects low-contrast images         |
| Sharpness   | Detects blurry images               |
| Noise       | Detects noisy images                |
| Color Shift | Detects channel imbalance           |

The diagnosis output can be:

```text
dark
low_contrast
blurry
noisy
color_shift
normal
```

<br>

## 🛠️ Restoration Methods

Each detected problem is mapped to one classical image processing method:

| Diagnosed Problem | Restoration Method          |
| ----------------- | --------------------------- |
| Dark              | Gamma correction            |
| Low Contrast      | CLAHE on luminance channel  |
| Blurry            | Unsharp masking             |
| Noisy             | Median filtering            |
| Color Shift       | Gray-world color correction |

<br>

## 📊 Evaluation

The project evaluates restoration quality by comparing:

```text
Reference image vs. Degraded image
Reference image vs. Restored image
```

The following metrics are used:

| Metric | Meaning                                       |
| ------ | --------------------------------------------- |
| MSE    | Mean Squared Error; lower is better           |
| PSNR   | Peak Signal-to-Noise Ratio; higher is better  |
| SSIM   | Structural Similarity Index; higher is better |

Main output files:

```text
metrics.csv
summary_metrics.csv
```

<br>

## 📈 Result Files

After running the project, the main result files are:

| File                    | Description                                                             |
| ----------------------- | ----------------------------------------------------------------------- |
| `degraded_metadata.csv` | Records degraded image filenames, paths, degradation types, and prompts |
| `restored_metadata.csv` | Records restored images and predicted problem labels                    |
| `metrics.csv`           | Stores MSE, PSNR, and SSIM for each image                               |
| `summary_metrics.csv`   | Stores average improvement grouped by degradation type                  |
| `comparisons/`          | Contains side-by-side reference, degraded, and restored images          |

<br>

## 📌 Reproducibility

This project is designed to be reproducible.

* The source images are stored in `dataset/raw/`.
* The image prompts are recorded in `metadata.csv`.
* The degradation process uses a fixed random seed.
* The full pipeline can be executed with one command:

```bash
python src/main.py
```

<br>

## 📄 Final Project Submission

This repository contains the required materials for the Image Processing final project:

* Source code
* README documentation
* Installation instructions
* Dataset information
* Evaluation results
* Final report
* Presentation slides

The GitHub repository link should also be included in the final report.

<br>

## 🔗 GitHub Repository

```text
https://github.com/Benjikuo/Classical-ai-image-restoration
```

<br>

## 📜 License

Released under the **MIT License**.

You are free to modify and use this project for learning and academic purposes.

**This project demonstrates that classical image processing methods can still provide a clear, explainable, and reproducible approach to image restoration.**
