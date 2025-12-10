import pandas as pd
import numpy as np
from tqdm.auto import tqdm

data_path = 'C:/Users/frank/OneDrive/Desktop/Progression-Prediction/train.csv'
#Data path for patient_event_slice.csv
data_event = 'C:/Users/frank/OneDrive/Desktop/Progression-Prediction/Code/patient_event_slice.csv'

#Merge slice filesfrom data_event to the train file
# 1) Carico i due file
df = pd.read_csv(data_path)
# Carico solo le colonne necessarie per evitare duplicati inutili
df_event = pd.read_csv(data_event, usecols=["Patient", "slice_files"])

# 2) Join left: mantiene tutte le righe di df_train e aggiunge slice_files quando disponibile
df_merged = df.merge(df_event, on="Patient", how="left")



A = {}          # slope per patient
B = {}          # intercept per patient
TAB = {}        # whatever get_tab returns
P = []          # patient list in same order

for p, sub in tqdm(df_merged.groupby('Patient'), total=df_merged['Patient'].nunique()):
    w = sub['Weeks'].to_numpy(dtype=float)
    y = sub['FVC'].to_numpy(dtype=float)

    # need at least 2 distinct x to fit a line
    if len(w) < 2 or np.unique(w).size < 2:
        A[p] = np.nan
        B[p] = np.nan
        P.append(p)
        continue

    X = np.column_stack([w, np.ones_like(w)])
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]  # slope, intercept

    A[p] = float(a)
    B[p] = float(b)
    P.append(p)

coefs = pd.DataFrame({
    'Patient': P,
    'fvc_slope': [A[p] for p in P],
    'fvc_intercept0': [B[p] for p in P]
})


# 1) Attacca i coefficienti a tutte le righe di df_merged (ripetuti per paziente)
df_with_coefs = df_merged.merge(coefs, on='Patient', how='left')

print(df_with_coefs)

df_with_coefs.to_csv('C:/Users/frank/OneDrive/Desktop/Progression-Prediction/train_with_coefs.csv', index=False)