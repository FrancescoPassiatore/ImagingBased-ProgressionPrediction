import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ============================
# PARAMETERS
# ============================
TARGET_WEEK = 52
WINDOW = 8           # 52 ± 8 weeks
DROP_PCT = -10.0     # progression threshold (%)

# ============================
# BASELINE FUNCTION
# ============================
def get_baseline(df):
    """baseline_rows = []

    for pid, group in df.groupby('Patient'):
        if 0 in group['Weeks'].values:
            row = group[group['Weeks'] == 0].iloc[0]
        else:
            idx = (group['Weeks'] - 0).abs().idxmin()
            row = group.loc[idx]

        baseline_rows.append({
            'Patient': pid,
            'FVC_baseline': row['FVC'],
            'baseline_week': row['Weeks']
        })"""
    
    baseline_rows =[]
    #Carica da D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\train_with_coefs.csv
    train_df = pd.read_csv(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\train_with_coefs.csv')

    for pid in train_df['Patient'].unique():
        baseline_fvc = train_df[train_df['Patient'] == pid]['fvc_intercept0'].values[0]
        baseline_rows.append({
            'Patient': pid,
            'FVC_baseline': baseline_fvc,
            'baseline_week': 0
        })

    return pd.DataFrame(baseline_rows)

# ============================
# 52-WEEK PROGRESSION LABEL
# ============================
def compute_event_52(group):
    group = group.copy()

    group['FVC_drop_pct'] = (
        (group['FVC'] - group['FVC_baseline']) /
        group['FVC_baseline'] * 100
    )

    window_df = group[
        (group['Weeks'] >= TARGET_WEEK - WINDOW) &
        (group['Weeks'] <= TARGET_WEEK + WINDOW)
    ]

    if len(window_df) == 0:
        return pd.Series({
            'event_52': np.nan,
            'fvc_52': np.nan,
            'week_52': np.nan
        })

    idx = (window_df['Weeks'] - TARGET_WEEK).abs().idxmin()
    row = window_df.loc[idx]

    event_52 = int(row['FVC_drop_pct'] <= DROP_PCT)

    return pd.Series({
        'event_52': event_52,
        'fvc_52': row['FVC'],
        'week_52': row['Weeks']
    })

# ============================
# LOAD DATA
# ============================
data_csv = (
    r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\train.csv"
)

df = pd.read_csv(data_csv)

# ============================
# BASELINE + LABELS
# ============================
baseline_df = get_baseline(df)
df = df.merge(baseline_df, on='Patient', how='left')

event_52_df = (
    df.groupby('Patient')
      .apply(compute_event_52)
      .reset_index()
)

# Drop patients without usable 52-week label
event_52_df = event_52_df.dropna(subset=['event_52'])

print("Class balance:")
print(event_52_df['event_52'].value_counts())

print(event_52_df['week_52'].describe())
print(event_52_df['event_52'].mean())


# ============================
# TRAIN / VAL / TEST SPLIT
# ============================
patients = event_52_df['Patient'].tolist()

train_patients, temp_patients = train_test_split(
    patients,
    test_size=0.3,
    random_state=42,
    stratify=event_52_df['event_52']
)

val_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.5,
    random_state=42,
    stratify=event_52_df
        .set_index('Patient')
        .loc[temp_patients]['event_52']
)

train_df = event_52_df[event_52_df['Patient'].isin(train_patients)].reset_index(drop=True)
val_df   = event_52_df[event_52_df['Patient'].isin(val_patients)].reset_index(drop=True)
test_df  = event_52_df[event_52_df['Patient'].isin(test_patients)].reset_index(drop=True)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

event_52_df.to_csv(
    "patient_progression_52w_intercept.csv",
    index=False
)

train_df.to_csv("train_patients_52w_intercept.csv", index=False)
val_df.to_csv("val_patients_52w_intercept.csv", index=False)
test_df.to_csv("test_patients_52w_intercept.csv", index=False)


label_info = {
    "endpoint": "FVC decline >= 10%",
    "horizon_weeks": 52,
    "window_weeks": 8,
    "n_patients": len(event_52_df),
    "positive_rate": event_52_df["event_52"].mean()
}

import json
with open("label_definition_52w_intercept.json", "w") as f:
    json.dump(label_info, f, indent=4)