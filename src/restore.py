from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from diagnose import diagnose_problem, calculate_quality_metrics

def restore_dark(img):
    # Gamma correction.
    gamma = 0.65
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def restore_low_contrast(img):
    # CLAHE on luminance channel.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

def restore_blurry(img):
    # Unsharp masking.
    blurred = cv2.GaussianBlur(img, (0, 0), 1.2)
    return cv2.addWeighted(img, 1.6, blurred, -0.6, 0)

def restore_noisy(img):
    # Median filter.
    return cv2.medianBlur(img, 3)

def restore_color_shift(img):
    # Gray-world color correction.
    img_f = img.astype(np.float32)
    means = img_f.reshape(-1, 3).mean(axis=0)
    gray_mean = means.mean()
    gains = gray_mean / (means + 1e-6)
    corrected = img_f * gains.reshape(1, 1, 3)
    return np.clip(corrected, 0, 255).astype(np.uint8)

RESTORE_FUNCTIONS = {
    "dark": restore_dark,
    "low_contrast": restore_low_contrast,
    "blurry": restore_blurry,
    "noisy": restore_noisy,
    "color_shift": restore_color_shift,
}

def adaptive_restore(img):
    predicted = diagnose_problem(img)

    if predicted in RESTORE_FUNCTIONS:
        restored = RESTORE_FUNCTIONS[predicted](img)
    else:
        restored = img.copy()

    return restored, predicted, calculate_quality_metrics(img)

def restore_dataset(dataset_dir: Path):
    degraded_metadata_path = dataset_dir / "degraded_metadata.csv"
    restored_dir = dataset_dir / "restored"
    restored_dir.mkdir(parents=True, exist_ok=True)

    if not degraded_metadata_path.exists():
        raise FileNotFoundError(f"Missing degraded_metadata.csv: {degraded_metadata_path}")

    df = pd.read_csv(degraded_metadata_path)
    records = []

    for _, row in df.iterrows():
        degradation_type = row["degradation_type"]
        degraded_path = dataset_dir / row["degraded_path"]
        img = cv2.imread(str(degraded_path))

        if img is None:
            print(f"[WARN] Cannot read image: {degraded_path}")
            continue

        restored, predicted, input_metrics = adaptive_restore(img)

        restored_subdir = restored_dir / degradation_type
        restored_subdir.mkdir(parents=True, exist_ok=True)

        restored_filename = row["degraded_filename"].replace("deg_", "res_")
        restored_path = restored_subdir / restored_filename
        cv2.imwrite(str(restored_path), restored)

        records.append({
            "raw_filename": row["raw_filename"],
            "degraded_filename": row["degraded_filename"],
            "restored_filename": restored_filename,
            "degradation_type": degradation_type,
            "predicted_problem": predicted,
            "restored_path": str(restored_path.relative_to(dataset_dir)),
            **{f"input_{k}": v for k, v in input_metrics.items()},
        })

    out = pd.DataFrame(records)
    out.to_csv(dataset_dir / "restored_metadata.csv", index=False, encoding="utf-8-sig")

    print(f"[OK] Restored images: {len(out)}")
    print(f"[OK] Saved: {dataset_dir / 'restored_metadata.csv'}")
    return out

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    restore_dataset(project_root / "dataset")
