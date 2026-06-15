from pathlib import Path
import cv2
import numpy as np
import pandas as pd

def resize_to_match(img, target_shape):
    h, w = target_shape[:2]
    if img.shape[:2] == (h, w):
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

def mse(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.mean((a - b) ** 2))

def psnr(a, b):
    v = mse(a, b)
    if v == 0:
        return 99.0
    return float(20 * np.log10(255.0 / np.sqrt(v)))

def simple_ssim(a, b):
    """
    Lightweight SSIM implementation to avoid extra dependency.
    """
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)

    sigma_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    sigma_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b

    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a + sigma_b + c2) + 1e-8
    )
    return float(np.mean(ssim_map))

def make_comparison(raw, degraded, restored):
    raw = resize_to_match(raw, degraded.shape)
    restored = resize_to_match(restored, degraded.shape)

    h, w = degraded.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    def label(img, text):
        canvas = img.copy()
        cv2.rectangle(canvas, (0, 0), (w, 42), (0, 0, 0), -1)
        cv2.putText(canvas, text, (12, 28), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas

    return np.hstack([
        label(raw, "Reference"),
        label(degraded, "Degraded"),
        label(restored, "Restored"),
    ])

def evaluate_dataset(dataset_dir: Path):
    restored_metadata_path = dataset_dir / "restored_metadata.csv"
    raw_dir = dataset_dir / "raw"
    comparison_dir = dataset_dir / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    if not restored_metadata_path.exists():
        raise FileNotFoundError(f"Missing restored_metadata.csv: {restored_metadata_path}")

    df = pd.read_csv(restored_metadata_path)
    records = []

    for _, row in df.iterrows():
        raw_path = raw_dir / row["raw_filename"]
        degraded_path = dataset_dir / "degraded" / row["degradation_type"] / row["degraded_filename"]
        restored_path = dataset_dir / row["restored_path"]

        raw = cv2.imread(str(raw_path))
        degraded = cv2.imread(str(degraded_path))
        restored = cv2.imread(str(restored_path))

        if raw is None or degraded is None or restored is None:
            print(f"[WARN] Missing files for: {row.to_dict()}")
            continue

        raw = resize_to_match(raw, degraded.shape)
        restored = resize_to_match(restored, degraded.shape)

        d_mse = mse(raw, degraded)
        r_mse = mse(raw, restored)
        d_psnr = psnr(raw, degraded)
        r_psnr = psnr(raw, restored)
        d_ssim = simple_ssim(raw, degraded)
        r_ssim = simple_ssim(raw, restored)

        cmp_dir = comparison_dir / row["degradation_type"]
        cmp_dir.mkdir(parents=True, exist_ok=True)
        cmp_name = row["restored_filename"].replace("res_", "cmp_")
        cmp_path = cmp_dir / cmp_name

        cv2.imwrite(str(cmp_path), make_comparison(raw, degraded, restored))

        records.append({
            "raw_filename": row["raw_filename"],
            "degraded_filename": row["degraded_filename"],
            "restored_filename": row["restored_filename"],
            "degradation_type": row["degradation_type"],
            "predicted_problem": row["predicted_problem"],
            "mse_degraded": d_mse,
            "mse_restored": r_mse,
            "mse_improvement": d_mse - r_mse,
            "psnr_degraded": d_psnr,
            "psnr_restored": r_psnr,
            "psnr_improvement": r_psnr - d_psnr,
            "ssim_degraded": d_ssim,
            "ssim_restored": r_ssim,
            "ssim_improvement": r_ssim - d_ssim,
            "comparison_path": str(cmp_path.relative_to(dataset_dir)),
        })

    metrics = pd.DataFrame(records)
    metrics.to_csv(dataset_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    summary = metrics.groupby("degradation_type")[[
        "mse_improvement",
        "psnr_improvement",
        "ssim_improvement",
    ]].mean().reset_index()

    summary.to_csv(dataset_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig")

    print(f"[OK] Saved: {dataset_dir / 'metrics.csv'}")
    print(f"[OK] Saved: {dataset_dir / 'summary_metrics.csv'}")
    return metrics

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    evaluate_dataset(project_root / "dataset")
