"""
Single Patient Slice Analysis (AUTO-CORRECTING VERSION)

- Detects excessive padding
- Crops foreground
- Resizes back to 224x224
- Keeps correct patients unchanged
"""

from os import path
from importlib.resources import path
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image


# =============================================================================
# FOREGROUND DETECTION + CORRECTION
# =============================================================================

def needs_correction(img,
                     intensity_threshold=0.05,
                     min_foreground_ratio=0.85,
                     min_std=0.05):
    """
    Detect padded / corrupted slices.

    Criteria:
    1. Foreground ratio is suspicious OR
    2. Global std is too low OR
    3. Large constant background present
    """

    foreground_ratio = np.mean(img > intensity_threshold)
    global_std = img.std()

    # Detect near-constant background
    low_var_mask = np.abs(img - img.mean()) < 0.01
    low_var_ratio = np.mean(low_var_mask)

    if foreground_ratio < min_foreground_ratio:
        return True

    if global_std < min_std:
        return True

    if low_var_ratio > 0.4:   # large flat region
        return True

    return False


def crop_foreground(img, threshold=0.05, min_size=32):
    mask = img > threshold
    coords = np.where(mask)

    if coords[0].size == 0:
        return img, False

    y0, y1 = coords[0].min(), coords[0].max()
    x0, x1 = coords[1].min(), coords[1].max()

    if (y1 - y0) < min_size or (x1 - x0) < min_size:
        return img, False

    cropped = img[y0:y1 + 1, x0:x1 + 1]
    return cropped, True

import numpy as np
from scipy import ndimage

from scipy import ndimage
import numpy as np

def detect_and_crop_by_geometry(img, min_fill_ratio=0.75, min_std=1e-3):
    """
    Geometry-based padding detection with safety checks.
    """

    # Edge-based anatomy cue (more reliable than percentile)
    gx, gy = np.gradient(img)
    grad_mag = np.sqrt(gx**2 + gy**2)

    binary = grad_mag > np.percentile(grad_mag, 75)

    labeled, n = ndimage.label(binary)
    if n == 0:
        return img, False

    sizes = ndimage.sum(binary, labeled, range(1, n + 1))
    largest_label = np.argmax(sizes) + 1
    mask = labeled == largest_label

    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        return img, False

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    cropped = img[y0:y1, x0:x1]

    # SAFETY CHECKS
    fill_ratio = cropped.size / img.size
    if fill_ratio >= min_fill_ratio:
        return img, False

    if np.std(cropped) < min_std:
        # Reject crop — it destroyed signal
        return img, False

    return cropped, True


def load_and_fix_slice(npy_path, target_size=(224, 224)):
    img = np.load(npy_path)

    img, did_crop = detect_and_crop_by_geometry(img)

    if did_crop:
        print("⚠️ Geometry-based crop applied")
    else:
        print("✓ Slice geometry accepted")

    if img.shape != (224, 224):
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA
                         )
    corrected = False

    if needs_correction(img):

        print(f"⚠️ Detected padded slice: {npy_path.name}")
        img, did_crop = crop_foreground(img)
        if did_crop:
            img = np.array(
                Image.fromarray(img).resize(target_size, Image.BILINEAR),
                dtype=np.float32
            )
            corrected = True

    

    return img, corrected


# =============================================================================
# ANALYSIS FUNCTION
# =============================================================================

def analyze_single_patient(patient_folder, max_slices_to_show=10):
    patient_folder = Path(patient_folder)

    print("=" * 70)
    print(f"ANALYZING PATIENT: {patient_folder.name}")
    print("=" * 70)

    npy_files = sorted(patient_folder.glob("*.npy"))
    if not npy_files:
        print("❌ No .npy files found")
        return

    print(f"\n✓ Found {len(npy_files)} slices")

    all_slices = []
    all_stats = []

    print("\n" + "-" * 70)
    print("ANALYZING EACH SLICE")
    print("-" * 70)

    for i, f in enumerate(npy_files):
        slice_data, corrected = load_and_fix_slice(f)
        all_slices.append(slice_data)

        stats = {
            "slice_index": i,
            "filename": f.name,
            "shape": slice_data.shape,
            "dtype": slice_data.dtype,
            "min": float(slice_data.min()),
            "max": float(slice_data.max()),
            "mean": float(slice_data.mean()),
            "std": float(slice_data.std()),
            "median": float(np.median(slice_data)),
            "non_zero_pixels": int(np.count_nonzero(slice_data)),
            "total_pixels": slice_data.size,
            "corrected": corrected
        }
        all_stats.append(stats)

        if i == 0 or i == len(npy_files) // 2 or i == len(npy_files) - 1:
            print(f"\n📄 Slice {i}: {f.name}")
            print(f"   Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"   Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"   Foreground %: {stats['non_zero_pixels'] / stats['total_pixels'] * 100:.1f}%")
            if corrected:
                print("   ⚠️ Auto-corrected (foreground crop + resize)")

    stats_df = pd.DataFrame(all_stats)

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print(f"Avg mean: {stats_df['mean'].mean():.4f} ± {stats_df['mean'].std():.4f}")
    print(f"Avg std:  {stats_df['std'].mean():.4f} ± {stats_df['std'].std():.4f}")

    num_corrected = stats_df["corrected"].sum()
    if num_corrected > 0:
        print(f"\n⚠️ {num_corrected}/{len(stats_df)} slices were auto-corrected")
    else:
        print("\n✓ No corrupted slices detected")

    # -------------------------------------------------------------------------
    # VISUALIZATION: STATISTICS
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    axes[0, 0].plot(stats_df["slice_index"], stats_df["mean"], marker="o")
    axes[0, 0].set_title("Mean Intensity per Slice")
    axes[0, 0].grid(True)

    axes[0, 1].plot(stats_df["slice_index"], stats_df["min"], label="Min")
    axes[0, 1].plot(stats_df["slice_index"], stats_df["max"], label="Max")
    axes[0, 1].legend()
    axes[0, 1].set_title("Min / Max per Slice")
    axes[0, 1].grid(True)

    axes[0, 2].plot(stats_df["slice_index"], stats_df["std"], marker="o", color="purple")
    axes[0, 2].set_title("Std per Slice")
    axes[0, 2].grid(True)

    fg_pct = stats_df["non_zero_pixels"] / stats_df["total_pixels"] * 100
    axes[1, 0].bar(stats_df["slice_index"], fg_pct)
    axes[1, 0].set_title("Foreground Percentage")
    axes[1, 0].grid(True, axis="y")

    all_pixels = np.concatenate([s.flatten() for s in all_slices])
    if np.ptp(all_pixels) > 1e-6:
        axes[1, 1].hist(all_pixels, bins=100, log=True)
    else:
        axes[1, 1].text(
            0.5, 0.5,
            "Histogram skipped\n(constant image)",
            ha="center", va="center", transform=axes[1, 1].transAxes
    )
    axes[1, 1].set_title("Global Intensity Histogram")

    axes[1, 2].boxplot([stats_df["mean"], stats_df["std"], stats_df["median"]],
                       labels=["Mean", "Std", "Median"])
    axes[1, 2].set_title("Statistics Distribution")

    plt.tight_layout()
    out1 = patient_folder / f"analysis_{patient_folder.name}.png"
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"✓ Saved statistics plot: {out1}")

    # -------------------------------------------------------------------------
    # VISUALIZATION: SLICES
    # -------------------------------------------------------------------------

    num_show = min(max_slices_to_show, len(all_slices))
    idxs = np.linspace(0, len(all_slices) - 1, num_show, dtype=int)

    fig = plt.figure(figsize=(20, 4 * ((num_show + 4) // 5)))
    for i, idx in enumerate(idxs):
        ax = plt.subplot((num_show + 4) // 5, 5, i + 1)
        ax.imshow(all_slices[idx], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Slice {idx}\n{npy_files[idx].name}")
        ax.axis("off")

    plt.tight_layout()
    out2 = patient_folder / f"slices_visual_{patient_folder.name}.png"
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"✓ Saved slice images: {out2}")

    # -------------------------------------------------------------------------
    # SAVE CSV
    # -------------------------------------------------------------------------

    csv_path = patient_folder / f"slice_statistics_{patient_folder.name}.csv"
    stats_df.to_csv(csv_path, index=False)
    print(f"✓ Saved statistics CSV: {csv_path}")

    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)

    return stats_df


# =============================================================================
# PREPROCESSING TEST
# =============================================================================

def test_slice_preprocessing(slice_path):
    print("\n" + "=" * 70)
    print("TESTING PREPROCESSING PIPELINE")
    print("=" * 70)

    slice_data, corrected = load_and_fix_slice(slice_path)

    if corrected:
        print("⚠️ Slice was auto-corrected before preprocessing")

    print(f"\nOriginal range: [{slice_data.min():.4f}, {slice_data.max():.4f}]")

    slice_3ch = np.stack([slice_data] * 3, axis=0)
    tensor = torch.from_numpy(slice_3ch).float()

    transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    transformed = transform(tensor)

    print(f"After ImageNet norm: mean={transformed.mean():.4f}, std={transformed.std():.4f}")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(slice_data, cmap="gray")
    axes[0].set_title("Corrected Slice")
    axes[1].imshow(slice_3ch[0], cmap="gray")
    axes[1].set_title("3-Channel")
    axes[2].imshow(tensor[0], cmap="gray")
    axes[2].set_title("Tensor")
    denorm = transformed[0] * 0.229 + 0.485
    axes[3].imshow(denorm, cmap="gray")
    axes[3].set_title("After ImageNet Norm")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    out = Path(slice_path).parent / "preprocessing_test.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"✓ Saved preprocessing visualization: {out}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_patient.py /path/to/patient_folder")
        sys.exit(1)

    patient_folder = sys.argv[1]

    analyze_single_patient(patient_folder)
    first_slice = sorted(Path(patient_folder).glob("*.npy"))[0]
    test_slice_preprocessing(first_slice)

    print("\n✅ All done!")
