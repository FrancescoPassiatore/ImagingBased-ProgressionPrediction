import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# =========================
# Load data
# =========================
df = pd.read_csv(
    "D:\\FrancescoP\\ImagingBased-ProgressionPrediction\\Training\\Progression_prediction_risk_2\\data\\patient_progression_52w.csv"
)

train_df = pd.read_csv("Training/CNN_Slope_Prediction/train.csv")

# =========================
# Build baseline FVC
# =========================
baseline_fvc_dict = {}

for patient_id in train_df['Patient'].unique():
    patient_data = train_df[train_df['Patient'] == patient_id]
    earliest_idx = patient_data['Weeks'].idxmin()
    baseline_fvc_dict[patient_id] = patient_data.loc[earliest_idx, 'FVC']

df["baseline_fvc"] = df["Patient"].map(baseline_fvc_dict)

# Drop missing
df = df.dropna(subset=["baseline_fvc", "fvc_52"])

# =========================
# Derived targets
# =========================
df["delta_fvc"] = df["fvc_52"] - df["baseline_fvc"]
df["delta_fvc_pct"] = df["delta_fvc"] / df["baseline_fvc"] * 100

# =========================
# 1️⃣ Descriptive statistics
# =========================
print("\n===== DESCRIPTIVE STATS =====")
print(df[["baseline_fvc", "fvc_52", "delta_fvc", "delta_fvc_pct"]].describe())

# =========================
# 2️⃣ Distribution: FVC@52
# =========================
plt.figure()
plt.hist(df["fvc_52"], bins=50)
plt.xlabel("True FVC at week 52")
plt.ylabel("Count")
plt.title("Distribution of True FVC@52")
plt.show()

# =========================
# 3️⃣ Baseline vs FVC@52
# =========================
plt.figure()
plt.scatter(df["baseline_fvc"], df["fvc_52"], alpha=0.4)
plt.plot([0, 7000], [0, 7000], "--", label="y = x")
plt.xlabel("Baseline FVC")
plt.ylabel("True FVC@52")
plt.title("Baseline vs FVC@52")
plt.legend()
plt.show()

# =========================
# 4️⃣ Distribution: ΔFVC
# =========================
plt.figure()
plt.hist(df["delta_fvc"], bins=50)
plt.axvline(0, linestyle="--")
plt.xlabel("ΔFVC (52w - baseline)")
plt.ylabel("Count")
plt.title("Distribution of ΔFVC")
plt.show()

# =========================
# 5️⃣ Distribution: %ΔFVC
# =========================
plt.figure()
plt.hist(df["delta_fvc_pct"], bins=50)
plt.axvline(0, linestyle="--")
plt.xlabel("ΔFVC (%)")
plt.ylabel("Count")
plt.title("Distribution of %ΔFVC")
plt.show()

# =========================
# 6️⃣ Correlations
# =========================
corr_baseline_fvc52 = pearsonr(df["baseline_fvc"], df["fvc_52"])[0]
corr_baseline_delta = pearsonr(df["baseline_fvc"], df["delta_fvc"])[0]

print("\n===== CORRELATIONS =====")
print(f"Corr(baseline_fvc, fvc_52)   = {corr_baseline_fvc52:.3f}")
print(f"Corr(baseline_fvc, delta_fvc) = {corr_baseline_delta:.3f}")

# =========================
# 7️⃣ Regression sanity check
# =========================
coef = np.polyfit(df["baseline_fvc"], df["fvc_52"], 1)
x = np.linspace(0, 7000, 100)
y = coef[0] * x + coef[1]

plt.figure()
plt.scatter(df["baseline_fvc"], df["fvc_52"], alpha=0.3)
plt.plot(x, y, color="red", label=f"Linear fit (slope={coef[0]:.2f})")
plt.plot([0, 7000], [0, 7000], "--", label="y = x")
plt.xlabel("Baseline FVC")
plt.ylabel("True FVC@52")
plt.legend()
plt.title("Baseline vs FVC@52 with linear fit")
plt.show()
