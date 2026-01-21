"""
Analisi distribuzione slope - Globale + K-Fold
==============================================

Questo script analizza:
1. Distribuzione globale degli slope
2. Distribuzione per fold (train / val / test)
3. Concentrazione centrale (±1σ)
4. Presenza di estremi
5. Grafici per dimostrare il rischio di mode collapse
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from scipy import stats
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utilities import IPFDataLoader

# =============================================================================
# CONFIG
# =============================================================================

CSV_PATH = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\train_with_coefs.csv'
FEATURES_PATH = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\patient_features.csv'
NPY_DIR = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy'
KFOLD_PATH = Path('D:\FrancescoP\ImagingBased-ProgressionPrediction\Training_2\Kfold_cyclic\kfold_cyclic_splits.pkl')

OUTPUT_DIR = Path(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training_2\CNN_Training\Cyclic_kfold\analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXTREME_THRESHOLD = 10.0

# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 80)
print("ANALISI DISTRIBUZIONE SLOPE - GLOBALE + FOLD")
print("=" * 80)

loader = IPFDataLoader(CSV_PATH, FEATURES_PATH, NPY_DIR)
patient_data, _ = loader.get_patient_data()

patient_ids = list(patient_data.keys())
slopes_all = np.array([patient_data[pid]['slope'] for pid in patient_ids])

print(f"✓ Pazienti caricati: {len(slopes_all)}")

# =============================================================================
# UTILS
# =============================================================================

def analyze_slopes(slopes):
    mean = np.mean(slopes)
    std = np.std(slopes)
    extreme_pct = np.mean(np.abs(slopes) > EXTREME_THRESHOLD) * 100
    concentration_1std = np.mean(np.abs(slopes - mean) <= std) * 100
    return mean, std, extreme_pct, concentration_1std

# =============================================================================
# ANALISI GLOBALE
# =============================================================================

print("\n" + "=" * 80)
print("📊 ANALISI GLOBALE")
print("=" * 80)

mean, std, extreme_pct, conc_1std = analyze_slopes(slopes_all)

print(f"""
[GLOBAL DATASET]
  N: {len(slopes_all)}
  Mean ± Std: {mean:.2f} ± {std:.2f}
  Median: {np.median(slopes_all):.2f}
  Min / Max: {slopes_all.min():.2f} / {slopes_all.max():.2f}
  Extreme % (|slope|>{EXTREME_THRESHOLD}): {extreme_pct:.1f}%
  ±1σ concentration: {conc_1std:.1f}%
""")

# =============================================================================
# FIGURA 1 — ISTOGRAMMA + KDE
# =============================================================================

plt.figure(figsize=(10, 6))
sns.histplot(slopes_all, bins=40, kde=True, stat="density", color="steelblue")
plt.axvline(mean, color='green', linestyle='--', label='Mean')
plt.axvline(np.median(slopes_all), color='orange', linestyle='--', label='Median')
plt.axvline(mean - std, color='red', linestyle=':', label='±1σ')
plt.axvline(mean + std, color='red', linestyle=':')
plt.axvline(-EXTREME_THRESHOLD, color='black', linestyle='-.', label='Extreme')
plt.axvline(EXTREME_THRESHOLD, color='black', linestyle='-.')
plt.title("Distribuzione globale degli slope")
plt.xlabel("Slope")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "global_hist_kde.png", dpi=300)
plt.close()

# =============================================================================
# FIGURA 2 — CDF
# =============================================================================

sorted_slopes = np.sort(slopes_all)
cdf = np.arange(1, len(sorted_slopes) + 1) / len(sorted_slopes)

plt.figure(figsize=(8, 6))
plt.plot(sorted_slopes, cdf, linewidth=2)
plt.axhline(0.8, color='red', linestyle='--', label='80%')
plt.xlabel("Slope")
plt.ylabel("Cumulative Probability")
plt.title("CDF degli slope")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "global_cdf.png", dpi=300)
plt.close()

# =============================================================================
# LOAD KFOLD
# =============================================================================

with open(KFOLD_PATH, 'rb') as f:
    splits = pickle.load(f)

print(f"\n✓ K-fold caricati: {len(splits)}")

# =============================================================================
# ANALISI PER FOLD
# =============================================================================

records = []

for fold_idx, fold in splits.items():
    print("\n" + "=" * 70)
    print(f"FOLD {fold_idx}")
    print("=" * 70)

    for split_name in ['train', 'val', 'test']:
        ids = fold[split_name]
        slopes = np.array([patient_data[pid]['slope'] for pid in ids])

        mean, std, extreme_pct, conc_1std = analyze_slopes(slopes)

        records.append({
            'fold': fold_idx,
            'split': split_name,
            'n': len(slopes),
            'mean': mean,
            'std': std,
            'median': np.median(slopes),
            'extreme_pct': extreme_pct,
            'conc_1std': conc_1std
        })

        print(f"""
                [{split_name.upper()}]
                N: {len(slopes)}
                Mean ± Std: {mean:.2f} ± {std:.2f}
                Extreme %: {extreme_pct:.1f}%
                ±1σ concentration: {conc_1std:.1f}%
                """)


# =============================================================================
# DATAFRAME RIASSUNTIVO
# =============================================================================

df_folds = pd.DataFrame(records)
df_folds.to_csv(OUTPUT_DIR / "fold_summary.csv", index=False)

# =============================================================================
# FIGURA 3 — BOXPLOT PER FOLD
# =============================================================================

plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df_folds,
    x='fold',
    y='mean',
    hue='split'
)
plt.title("Distribuzione media slope per fold / split")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "boxplot_fold_means.png", dpi=300)
plt.close()

# =============================================================================
# FIGURA 4 — % ESTREMI PER FOLD
# =============================================================================

plt.figure(figsize=(12, 6))
sns.barplot(
    data=df_folds,
    x='fold',
    y='extreme_pct',
    hue='split'
)
plt.axhline(10, color='red', linestyle='--', label='10%')
plt.ylabel("% estremi")
plt.title("Percentuale di slope estremi per fold")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "extreme_pct_per_fold.png", dpi=300)
plt.close()

# =============================================================================
# FIGURA 5 — CONCENTRAZIONE ±1σ
# =============================================================================

plt.figure(figsize=(12, 6))
sns.barplot(
    data=df_folds,
    x='fold',
    y='conc_1std',
    hue='split'
)
plt.axhline(70, color='red', linestyle='--', label='70%')
plt.ylabel("% entro ±1σ")
plt.title("Concentrazione centrale per fold")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "concentration_1std_per_fold.png", dpi=300)
plt.close()

print("\n✓ Analisi completata")
print(f"📁 Output: {OUTPUT_DIR}")
