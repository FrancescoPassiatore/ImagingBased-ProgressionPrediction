import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, median_absolute_error
import matplotlib.pyplot as plt
import argparse

# === Command-line arguments ===
parser = argparse.ArgumentParser(description="Baseline generalization to new patients")
parser.add_argument('--linear', action='store_true', help='Use linear regression')
parser.add_argument('--poly', type=int, help='Use polynomial regression of given degree')
args = parser.parse_args()

# --- Model selection ---
if args.linear:
    model_type = 'linear'
elif args.poly:
    model_type = 'poly'
    degree = args.poly
else:
    raise ValueError("Specify --linear or --poly DEGREE")


# === Load Data ===
data_csv = 'C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/CodeDevelopment/osic-pulmonary-fibrosis-progression/train.csv'
df = pd.read_csv(data_csv)

# === Split by patients ===
patients = df['Patient'].unique()
np.random.shuffle(patients)

split = int(0.8 * len(patients))
train_patients = patients[:split]
test_patients  = patients[split:]

df_train = df[df['Patient'].isin(train_patients)]
df_test  = df[df['Patient'].isin(test_patients)]

print(f"\nTrain patients: {len(train_patients)}")
print(f"Test patients : {len(test_patients)}")


# === Prepare TRAIN data (all points of all train patients) ===
X_train = df_train['Weeks'].values.reshape(-1, 1)
y_train = df_train['FVC'].values

# === Fit model ===
if model_type == 'linear':
    model = LinearRegression()
    model.fit(X_train, y_train)

elif model_type == 'poly':
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

print("\nModel trained on all TRAIN patients.")


# === Evaluate on TEST patients (completely unseen) ===
results = []

for patient in test_patients:
    df_p = df_test[df_test['Patient'] == patient].sort_values('Weeks')

    X_test = df_p['Weeks'].values.reshape(-1, 1)
    y_test = df_p['FVC'].values

    if model_type == 'linear':
        y_pred = model.predict(X_test)

    elif model_type == 'poly':
        X_test_poly = poly.transform(X_test)
        y_pred = model.predict(X_test_poly)

    mae = mean_absolute_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)

    results.append({
        'Patient': patient,
        'MAE': mae,
        'MedianAE': medae,
        'N_test_points': len(y_test)
    })


# === Results ===
results_df = pd.DataFrame(results)

print("\n=== GENERALIZATION TO NEW PATIENTS ===")
print(results_df.describe())

print("\nMAE per test patient:")
print(results_df[['Patient', 'MAE', 'MedianAE']])

# === Distribution plot ===
plt.figure(figsize=(12,5))
plt.hist(results_df['MAE'], bins=20, alpha=0.7, label='MAE', color='skyblue', edgecolor='black')
plt.hist(results_df['MedianAE'], bins=20, alpha=0.7, label='MedianAE', color='orange', edgecolor='black')
plt.title("Error Distribution on UNSEEN Patients")
plt.xlabel("Error")
plt.ylabel("Count")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
