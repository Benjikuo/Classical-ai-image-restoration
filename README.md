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

## 📊 Dataset Information

This project uses **40 clean AI-generated images** as reference images.
Each image is used to generate **five degraded versions**:

* `dark`
* `low_contrast`
* `blurry`
* `noisy`
* `color_shift`

In total, the dataset contains:

```text
40 clean images
200 degraded images
200 restored images
```

All images and metadata are stored under the `dataset/` folder.

<br>

## 📈 Results

The evaluation results are saved in:

```text
dataset/metrics.csv
dataset/summary_metrics.csv
```

The project evaluates restoration quality using:

* `MSE`
* `PSNR`
* `SSIM`

Comparison images are saved in:

```text
dataset/comparisons/
```

<br>

## 📜 License

Released under the **MIT License**.

You are free to modify and use this project for learning and academic purposes.

**This project demonstrates that classical image processing methods can still provide a clear, explainable, and reproducible approach to image restoration.**
