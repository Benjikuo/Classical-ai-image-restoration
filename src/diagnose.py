import cv2
import numpy as np

def brightness_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def contrast_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))

def sharpness_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def noise_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 3)
    residual = gray.astype(np.float32) - median.astype(np.float32)
    return float(np.std(residual))

def color_shift_score(img):
    means = img.reshape(-1, 3).mean(axis=0)  # BGR
    return float(np.max(means) - np.min(means))

def calculate_quality_metrics(img):
    return {
        "brightness": brightness_score(img),
        "contrast": contrast_score(img),
        "sharpness": sharpness_score(img),
        "noise": noise_score(img),
        "color_shift": color_shift_score(img),
    }

def diagnose_problem(img):
    """
    Rule-based image problem diagnosis.
    """
    m = calculate_quality_metrics(img)

    if m["brightness"] < 85:
        return "dark"
    if m["contrast"] < 42:
        return "low_contrast"
    if m["sharpness"] < 80:
        return "blurry"
    if m["noise"] > 12:
        return "noisy"
    if m["color_shift"] > 45:
        return "color_shift"

    return "normal"

if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python src/diagnose.py dataset/degraded/dark/deg_0001_dark.png")
        raise SystemExit(1)

    path = Path(sys.argv[1])
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)

    print(calculate_quality_metrics(img))
    print("diagnosis:", diagnose_problem(img))
