import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import glob


root_path = r"C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/CodeDevelopment/osic-pulmonary-fibrosis-progression/train"


def get_percentile_slices(slice_paths, start=0.3, end=0.6, num_samples=11):
    """
    Select unique slices between given percentiles (e.g. 30–60%),
    skipping duplicates when multiple percentiles map to the same index.
    """
    n = len(slice_paths)
    if n == 0:
        return []

    percentiles = np.linspace(start, end, num_samples)
    selected = []
    last_idx = -1

    for perc in percentiles:
        idx = int(round(perc * (n - 1)))
        idx = np.clip(idx, 0, n - 1)

        # Skip duplicates (same logic as 'flag' in the Kaggle code)
        if idx == last_idx:
            continue
        last_idx = idx

        selected.append(slice_paths[idx])
    
    return selected


def get_baseline(df):
    baseline_rows = []

    for pid, group in df.groupby('Patient'):
        # If week 0 exists, use it
        if 0 in group['Weeks'].values:
            baseline_fvc = group.loc[group['Weeks'] == 0, 'FVC'].values[0]
            baseline_week = 0
        else:
            # Otherwise, choose the week closest to 0
            closest_idx = (group['Weeks'] - 0).abs().idxmin()
            baseline_fvc = group.loc[closest_idx, 'FVC']
            baseline_week = group.loc[closest_idx, 'Weeks']
        
        baseline_rows.append({'Patient': pid, 'FVC_baseline': baseline_fvc, 'baseline_week': baseline_week})

    baseline_df = pd.DataFrame(baseline_rows)
    return baseline_df

def compute_event_time(group):
    """Calcola evento (1/0) e tempo fino all'evento o censura per ogni paziente."""
    # Calcola la variazione percentuale FVC dal baseline
    group = group.copy()
    group['FVC_drop_pct'] = (group['FVC'] - group['FVC_baseline']) / group['FVC_baseline'] * 100
    
    progression = group[group['FVC_drop_pct'] <= -10]
    
    if len(progression) > 0:
        event = 1
        event_week = progression.iloc[0]['Weeks']
        time = max(event_week - group['baseline_week'].iloc[0], 1)
    else:
        event = 0
        last_week = group['Weeks'].iloc[-1]
        time = max(last_week - group['baseline_week'].iloc[0], 1)
        
    return pd.Series({'event': event, 'time': time})

#Create db for training
# Carica i dati
data_csv = 'C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/CodeDevelopment/osic-pulmonary-fibrosis-progression/train.csv'
df = pd.read_csv(data_csv)

baseline_df = get_baseline(df)


# Merge baseline back to original df
df = df.merge(baseline_df[['Patient','FVC_baseline','baseline_week']], on='Patient', how='left')

event_time_df = df.groupby('Patient').apply(compute_event_time).reset_index()



# Merge back to the full df
df = df.merge(event_time_df, on='Patient', how='left')

# Create patient-level list
patient_list = []

patient_ids = sorted(os.listdir(root_path))
for pid in patient_ids:
    patient_folder = os.path.join(root_path, pid)
    if not os.path.isdir(patient_folder):
        continue

    # List all DICOM slices
    slice_paths = sorted(glob.glob(os.path.join(patient_folder, "*.dcm")))
    
    selected_slices = get_percentile_slices(slice_paths, start=0.3, end=0.6, num_samples=11)



    patient_list.append({
        'Patient': pid,
        'slice_paths': selected_slices
    })


# Convert to DataFrame
train_df = pd.DataFrame(patient_list)

# Save to CSV (optional: slice_paths as strings, join by ;)
train_df['slice_files'] = train_df['slice_paths'].apply(lambda paths: [os.path.basename(p) for p in paths])

# Merge by patient


event_slices_df = event_time_df.merge(train_df[['Patient','slice_files']], on='Patient', how='left')


event_slices_df.to_csv('patient_event_slice.csv',index=False)

from sklearn.model_selection import train_test_split

# Shuffle patients
patients = event_slices_df['Patient'].tolist()

# First split: train vs temp (val+test)
train_patients, temp_patients = train_test_split(
    patients, test_size=0.3, random_state=42, stratify=event_slices_df['event']
)

# Second split: val vs test (50/50 of temp)
val_patients, test_patients = train_test_split(
    temp_patients, test_size=0.5, random_state=42, stratify=event_slices_df.set_index('Patient').loc[temp_patients]['event']
)

train_df = event_slices_df[event_slices_df['Patient'].isin(train_patients)].reset_index(drop=True)
val_df   = event_slices_df[event_slices_df['Patient'].isin(val_patients)].reset_index(drop=True)
test_df  = event_slices_df[event_slices_df['Patient'].isin(test_patients)].reset_index(drop=True)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
