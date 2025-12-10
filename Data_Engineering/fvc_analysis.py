import pandas as pd

data_csv = 'C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/CodeDevelopment/osic-pulmonary-fibrosis-progression/train.csv'
df = pd.read_csv(data_csv)

# Sort for safety
df = df.sort_values(['Patient', 'Weeks'])

# Compute ΔFVC per patient
df['Delta_FVC'] = df.groupby('Patient')['FVC'].diff()
df['Delta_Weeks'] = df.groupby('Patient')['Weeks'].diff()

# Optional: rate of change (FVC change per week)
df['Delta_FVC_per_Week'] = df['Delta_FVC'] / df['Delta_Weeks']

# Drop NaNs (first record per patient has no delta)
delta_summary = df.dropna(subset=['Delta_FVC'])

mean_delta = df['Delta_FVC'].mean()
median_delta = df['Delta_FVC'].median()
std_delta = df['Delta_FVC'].std()

print(f"Average ΔFVC between measurements: {mean_delta:.2f}")
print(f"Median ΔFVC: {median_delta:.2f}")
print(f"Standard Deviation of ΔFVC: {std_delta:.2f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.hist(df['Delta_FVC'].dropna(), bins=40, edgecolor='k')
plt.title('Distribution of FVC Changes Between Weeks')
plt.xlabel('ΔFVC (ml)')
plt.ylabel('Number of Observations')
plt.grid(True)
plt.show()

import seaborn as sns

plt.figure(figsize=(6, 5))
sns.boxplot(y=df['Delta_FVC_per_Week'])
plt.title('Distribution of FVC Change Rate (per Week)')
plt.ylabel('ΔFVC per Week (ml/week)')
plt.grid(True)
plt.show()

per_patient_delta = (
    df.groupby('Patient')['Delta_FVC_per_Week']
      .mean()
      .dropna()
      .reset_index(name='Mean_Delta_FVC_per_Week')
)

print(per_patient_delta.describe())  # summary statistics


