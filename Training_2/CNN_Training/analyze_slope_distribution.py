"""
Analisi della distribuzione degli slope nel dataset
====================================================

Questo script analizza:
1. Distribuzione degli slope (mean, std, min, max, quantiles)
2. Presenza di outliers
3. Correlazione tra slope e features handcrafted
4. Visualizzazioni per capire il dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utilities import IPFDataLoader

# Configurazione
CSV_PATH = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\train_with_coefs.csv'
FEATURES_PATH = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\patient_features.csv'
NPY_DIR = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy'
OUTPUT_DIR = Path('Training_2/CNN_Training/data_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Carica dati
print("="*80)
print("ANALISI DISTRIBUZIONE SLOPE")
print("="*80)
print("\n📂 Caricamento dati...")

loader = IPFDataLoader(CSV_PATH, FEATURES_PATH, NPY_DIR)
patient_data, features_data = loader.get_patient_data()

print(f"✓ Caricati {len(patient_data)} pazienti")

# Estrai slopes
slopes = np.array([patient_data[pid]['slope'] for pid in patient_data.keys()])

# ============================================================================
# STATISTICHE DESCRITTIVE
# ============================================================================

print("\n" + "="*80)
print("📊 STATISTICHE DESCRITTIVE DEGLI SLOPE")
print("="*80)

print(f"\nNumero pazienti:    {len(slopes)}")
print(f"Mean:               {np.mean(slopes):.4f}")
print(f"Std:                {np.std(slopes):.4f}")
print(f"Min:                {np.min(slopes):.4f}")
print(f"Max:                {np.max(slopes):.4f}")
print(f"Mediana:            {np.median(slopes):.4f}")
print(f"\nQuantiles:")
print(f"  Q1 (25%):         {np.percentile(slopes, 25):.4f}")
print(f"  Q2 (50%):         {np.percentile(slopes, 50):.4f}")
print(f"  Q3 (75%):         {np.percentile(slopes, 75):.4f}")
print(f"  IQR:              {np.percentile(slopes, 75) - np.percentile(slopes, 25):.4f}")

# Test di normalità
statistic, p_value = stats.shapiro(slopes[:5000] if len(slopes) > 5000 else slopes)
print(f"\nShapiro-Wilk test (normalità):")
print(f"  Statistic:        {statistic:.4f}")
print(f"  P-value:          {p_value:.4f}")
print(f"  Distribuzione:    {'Normale' if p_value > 0.05 else 'NON normale'}")

# Skewness e Kurtosis
print(f"\nSkewness:           {stats.skew(slopes):.4f}")
print(f"Kurtosis:           {stats.kurtosis(slopes):.4f}")

# ============================================================================
# ANALISI OUTLIERS
# ============================================================================

print("\n" + "="*80)
print("🔍 ANALISI OUTLIERS")
print("="*80)

q1 = np.percentile(slopes, 25)
q3 = np.percentile(slopes, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = slopes[(slopes < lower_bound) | (slopes > upper_bound)]
outliers_pct = (len(outliers) / len(slopes)) * 100

print(f"\nIQR bounds:")
print(f"  Lower bound:      {lower_bound:.4f}")
print(f"  Upper bound:      {upper_bound:.4f}")
print(f"\nOutliers:           {len(outliers)} ({outliers_pct:.1f}%)")
print(f"Extreme values:")
print(f"  Top 5 max:        {sorted(slopes)[-5:]}")
print(f"  Top 5 min:        {sorted(slopes)[:5]}")

# ============================================================================
# ANALISI CONCENTRAZIONE
# ============================================================================

print("\n" + "="*80)
print("📈 ANALISI CONCENTRAZIONE VALORI")
print("="*80)

# Conta quanti valori sono vicini alla media
bins = [
    (-100, -10, "Molto negativi"),
    (-10, -5, "Negativi moderati"),
    (-5, -2, "Negativi lievi"),
    (-2, 0, "Quasi stabili"),
    (0, 5, "Positivi lievi"),
    (5, 100, "Molto positivi")
]

print(f"\nDistribuzione per range:")
for lower, upper, label in bins:
    count = np.sum((slopes >= lower) & (slopes < upper))
    pct = (count / len(slopes)) * 100
    print(f"  {label:20s} [{lower:6.1f}, {upper:6.1f}): {count:4d} ({pct:5.1f}%)")

# Concentrazione intorno alla media
mean_slope = np.mean(slopes)
within_1std = np.sum(np.abs(slopes - mean_slope) <= np.std(slopes))
within_2std = np.sum(np.abs(slopes - mean_slope) <= 2 * np.std(slopes))

print(f"\nConcentrazione intorno alla media ({mean_slope:.2f}):")
print(f"  ±1 std:           {within_1std} ({within_1std/len(slopes)*100:.1f}%)")
print(f"  ±2 std:           {within_2std} ({within_2std/len(slopes)*100:.1f}%)")

# ============================================================================
# CORRELAZIONE CON FEATURES
# ============================================================================

print("\n" + "="*80)
print("🔗 CORRELAZIONE SLOPE vs FEATURES")
print("="*80)

# Crea DataFrame con features e slope
features_list = []
for pid in patient_data.keys():
    if pid in features_data:
        row = {'patient_id': pid, 'slope': patient_data[pid]['slope']}
        row.update(features_data[pid])
        features_list.append(row)

df = pd.DataFrame(features_list)

# Calcola correlazioni
feature_cols = [
    'approx_vol', 'avg_num_tissue_pixel', 'avg_tissue', 
    'avg_tissue_thickness', 'avg_tissue_by_total', 'avg_tissue_by_lung',
    'mean', 'skew', 'kurtosis', 'age'
]

print("\nCorrelazioni (Pearson) con slope:")
correlations = []
for col in feature_cols:
    if col in df.columns:
        corr = df['slope'].corr(df[col])
        correlations.append((col, corr))
        print(f"  {col:25s}: {corr:7.4f}")

# Ordina per valore assoluto
correlations.sort(key=lambda x: abs(x[1]), reverse=True)
print("\nTop 5 correlazioni (valore assoluto):")
for col, corr in correlations[:5]:
    print(f"  {col:25s}: {corr:7.4f}")

# ============================================================================
# VISUALIZZAZIONI
# ============================================================================

print("\n" + "="*80)
print("📊 GENERAZIONE VISUALIZZAZIONI")
print("="*80)

# 1. Istogramma + KDE
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Subplot 1: Istogramma con KDE
ax = axes[0, 0]
ax.hist(slopes, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
from scipy.stats import gaussian_kde
kde = gaussian_kde(slopes)
x_range = np.linspace(slopes.min(), slopes.max(), 1000)
ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
ax.axvline(np.mean(slopes), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(slopes):.2f}')
ax.axvline(np.median(slopes), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(slopes):.2f}')
ax.set_xlabel('Slope', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Distribuzione Slope (con KDE)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Subplot 2: Boxplot
ax = axes[0, 1]
bp = ax.boxplot(slopes, vert=True, patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
ax.set_ylabel('Slope', fontsize=12)
ax.set_title('Boxplot Slope (outliers visibili)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Subplot 3: Q-Q Plot
ax = axes[1, 0]
stats.probplot(slopes, dist="norm", plot=ax)
ax.set_title('Q-Q Plot (Test Normalità)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Subplot 4: Cumulative Distribution
ax = axes[1, 1]
sorted_slopes = np.sort(slopes)
cumulative = np.arange(1, len(sorted_slopes) + 1) / len(sorted_slopes)
ax.plot(sorted_slopes, cumulative, linewidth=2, color='steelblue')
ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Mediana')
ax.axvline(np.median(slopes), color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Slope', fontsize=12)
ax.set_ylabel('Cumulative Probability', fontsize=12)
ax.set_title('Funzione di Distribuzione Cumulativa', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = OUTPUT_DIR / 'slope_distribution_analysis.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Salvato: {plot_path}")
plt.close()

# 2. Correlazione Matrix
fig, ax = plt.subplots(figsize=(12, 10))
corr_cols = ['slope'] + [col for col in feature_cols if col in df.columns]
corr_matrix = df[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Matrice di Correlazione (Slope vs Features)', fontsize=14, fontweight='bold')
plt.tight_layout()
plot_path = OUTPUT_DIR / 'correlation_matrix.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Salvato: {plot_path}")
plt.close()

# 3. Scatter plots delle top correlazioni
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (col, corr) in enumerate(correlations[:6]):
    if idx >= 6:
        break
    ax = axes[idx]
    ax.scatter(df[col], df['slope'], alpha=0.5, s=20)
    ax.set_xlabel(col, fontsize=10)
    ax.set_ylabel('Slope', fontsize=10)
    ax.set_title(f'{col} vs Slope (r={corr:.3f})', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Aggiungi linea di regressione
    z = np.polyfit(df[col].dropna(), df.loc[df[col].notna(), 'slope'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[col].min(), df[col].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.7)

plt.tight_layout()
plot_path = OUTPUT_DIR / 'feature_correlations.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Salvato: {plot_path}")
plt.close()

# ============================================================================
# SALVA REPORT
# ============================================================================

print("\n" + "="*80)
print("💾 SALVATAGGIO REPORT")
print("="*80)

report = {
    'n_patients': len(slopes),
    'statistics': {
        'mean': float(np.mean(slopes)),
        'std': float(np.std(slopes)),
        'min': float(np.min(slopes)),
        'max': float(np.max(slopes)),
        'median': float(np.median(slopes)),
        'q1': float(np.percentile(slopes, 25)),
        'q3': float(np.percentile(slopes, 75)),
        'iqr': float(np.percentile(slopes, 75) - np.percentile(slopes, 25)),
        'skewness': float(stats.skew(slopes)),
        'kurtosis': float(stats.kurtosis(slopes))
    },
    'outliers': {
        'count': int(len(outliers)),
        'percentage': float(outliers_pct),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound)
    },
    'normality_test': {
        'shapiro_statistic': float(statistic),
        'shapiro_pvalue': float(p_value),
        'is_normal': bool(p_value > 0.05)
    },
    'correlations': {col: float(corr) for col, corr in correlations}
}

import json
report_path = OUTPUT_DIR / 'slope_analysis_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f"✓ Report salvato: {report_path}")

# Salva anche CSV con dati completi
df.to_csv(OUTPUT_DIR / 'patient_data_with_slopes.csv', index=False)
print(f"✓ CSV salvato: {OUTPUT_DIR / 'patient_data_with_slopes.csv'}")

print("\n" + "="*80)
print("✅ ANALISI COMPLETATA!")
print("="*80)
print(f"\n📁 Output directory: {OUTPUT_DIR}")
print("\nFile generati:")
print("  - slope_distribution_analysis.png")
print("  - correlation_matrix.png")
print("  - feature_correlations.png")
print("  - slope_analysis_report.json")
print("  - patient_data_with_slopes.csv")

# ============================================================================
# DIAGNOSI MODE COLLAPSE
# ============================================================================

print("\n" + "="*80)
print("🔍 DIAGNOSI MODE COLLAPSE")
print("="*80)

print("\n⚠️  Analisi della concentrazione dei dati:")

# Calcola quanto è concentrata la distribuzione
concentration_1std = (within_1std / len(slopes)) * 100
concentration_2std = (within_2std / len(slopes)) * 100

if concentration_1std > 70:
    print(f"  🔴 {concentration_1std:.1f}% dei dati entro ±1σ dalla media")
    print(f"     Distribuzione molto concentrata → Mode collapse prevedibile!")
elif concentration_1std > 50:
    print(f"  🟡 {concentration_1std:.1f}% dei dati entro ±1σ dalla media")
    print(f"     Distribuzione moderatamente concentrata")
else:
    print(f"  🟢 {concentration_1std:.1f}% dei dati entro ±1σ dalla media")
    print(f"     Distribuzione sufficientemente dispersa")

print(f"\n💡 Predizione σ del modello vs σ reale:")
print(f"  σ reale:          {np.std(slopes):.2f}")
print(f"  σ osservata CNN:  0.04-0.17")
print(f"  Ratio:            {0.17/np.std(slopes):.3f} (dovrebbe essere ~1.0)")

print(f"\n🎯 Raccomandazioni:")
if outliers_pct > 20:
    print(f"  - Molti outliers ({outliers_pct:.1f}%) → Considera RobustScaler o clip")
if concentration_1std > 70:
    print(f"  - Dati molto concentrati → Considera data augmentation o resampling")
if abs(correlations[0][1]) < 0.3:
    print(f"  - Correlazioni deboli con features → Immagini potrebbero non essere informative")

print("\n" + "="*80)
