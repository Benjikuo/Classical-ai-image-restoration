from pathlib import Path
import cv2
import numpy as np
import pandas as pd

DEGRADATION_TYPES = ["dark", "low_contrast", "blurry", "noisy", "color_shift"]

def degrade_dark(img):
    # Intensity scaling: make the image underexposed.
    return np.clip(img.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)

def degrade_low_contrast(img):
    # Compress intensities toward the middle gray range.
    return np.clip(128 + (img.astype(np.float32) - 128) * 0.45, 0, 255).astype(np.uint8)

def degrade_blurry(img):
    # Gaussian blur.
    return cv2.GaussianBlur(img, (13, 13), 0)

def degrade_noisy(img):
    # Gaussian noise.
    noise = np.random.normal(0, 25, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def degrade_color_shift(img):
    # BGR color shift.
    shifted = img.astype(np.float32).copy()
    shifted[:, :, 0] *= 1.25  # Blue
    shifted[:, :, 1] *= 1.05  # Green
    shifted[:, :, 2] *= 0.75  # Red
    return np.clip(shifted, 0, 255).astype(np.uint8)

DEGRADE_FUNCTIONS = {
    "dark": degrade_dark,
    "low_contrast": degrade_low_contrast,
    "blurry": degrade_blurry,
    "noisy": degrade_noisy,
    "color_shift": degrade_color_shift,
}

def generate_degraded_dataset(dataset_dir: Path):
    raw_dir = dataset_dir / "raw"
    metadata_path = dataset_dir / "metadata.csv"
    degraded_dir = dataset_dir / "degraded"

    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw directory: {raw_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.csv: {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    if "filename" not in metadata.columns:
        raise ValueError("metadata.csv must contain a filename column")
    if "prompt" not in metadata.columns:
        metadata["prompt"] = ""

    for degradation in DEGRADATION_TYPES:
        (degraded_dir / degradation).mkdir(parents=True, exist_ok=True)

    records = []

    for _, row in metadata.iterrows():
        raw_filename = str(row["filename"])
        raw_path = raw_dir / raw_filename
        img = cv2.imread(str(raw_path))

        if img is None:
            print(f"[WARN] Cannot read image: {raw_path}")
            continue

        raw_id = Path(raw_filename).stem.replace("raw_", "")

        for degradation in DEGRADATION_TYPES:
            degraded = DEGRADE_FUNCTIONS[degradation](img)
            degraded_filename = f"deg_{raw_id}_{degradation}.png"
            degraded_path = degraded_dir / degradation / degraded_filename

            cv2.imwrite(str(degraded_path), degraded)

            records.append({
                "raw_filename": raw_filename,
                "degraded_filename": degraded_filename,
                "degradation_type": degradation,
                "degraded_path": str(degraded_path.relative_to(dataset_dir)),
                "prompt": row.get("prompt", ""),
            })

    out = pd.DataFrame(records)
    out.to_csv(dataset_dir / "degraded_metadata.csv", index=False, encoding="utf-8-sig")

    print(f"[OK] Generated degraded images: {len(out)}")
    print(f"[OK] Saved: {dataset_dir / 'degraded_metadata.csv'}")
    return out

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    generate_degraded_dataset(project_root / "dataset")
