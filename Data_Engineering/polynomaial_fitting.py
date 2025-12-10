import pandas as pd
import numpy as np
from tqdm.auto import tqdm

data_path = 'C:/Users/frank/OneDrive/Desktop/Progression-Prediction/train.csv'
data_event = 'C:/Users/frank/OneDrive/Desktop/Progression-Prediction/Code/patient_event_slice.csv'

# Merge slice files from data_event to the train file
df = pd.read_csv(data_path)
df_event = pd.read_csv(data_event, usecols=["Patient", "slice_files"])
df_merged = df.merge(df_event, on="Patient", how="left")

# Scegli il grado della polinomiale (2 o 3)
POLY_DEGREE = 3  # Cambia a 3 per polinomiale cubica

coeffs_dict = {}  # dizionario per salvare tutti i coefficienti per paziente
P = []            # lista pazienti

for p, sub in tqdm(df_merged.groupby('Patient'), total=df_merged['Patient'].nunique()):
    w = sub['Weeks'].to_numpy(dtype=float)
    y = sub['FVC'].to_numpy(dtype=float)

    # Serve almeno (degree + 1) punti per fittare una polinomiale
    if len(w) < (POLY_DEGREE + 1) or np.unique(w).size < (POLY_DEGREE + 1):
        # Riempi con NaN se non ci sono abbastanza punti
        coeffs_dict[p] = [np.nan] * (POLY_DEGREE + 1)
        P.append(p)
        continue

    # np.polyfit restituisce coefficienti in ordine decrescente: [a_n, a_{n-1}, ..., a_1, a_0]
    # Per degree=2: [a, b, c] dove y = a*x^2 + b*x + c
    # Per degree=3: [a, b, c, d] dove y = a*x^3 + b*x^2 + c*x + d
    poly_coeffs = np.polyfit(w, y, POLY_DEGREE)
    
    coeffs_dict[p] = poly_coeffs.tolist()
    P.append(p)

# Crea nomi colonne dinamici in base al grado
if POLY_DEGREE == 2:
    col_names = ['fvc_poly2_coef', 'fvc_poly1_coef', 'fvc_poly0_coef']
elif POLY_DEGREE == 3:
    col_names = ['fvc_poly3_coef', 'fvc_poly2_coef', 'fvc_poly1_coef', 'fvc_poly0_coef']
else:
    col_names = [f'fvc_poly{POLY_DEGREE-i}_coef' for i in range(POLY_DEGREE + 1)]

# Costruisci il DataFrame
coefs_data = {'Patient': P}
for i, col_name in enumerate(col_names):
    coefs_data[col_name] = [coeffs_dict[p][i] for p in P]

coefs = pd.DataFrame(coefs_data)

# Merge con i dati originali
df_with_coefs = df_merged.merge(coefs, on='Patient', how='left')

print(df_with_coefs)

# Salva il risultato
output_path = f'C:/Users/frank/OneDrive/Desktop/Progression-Prediction/train_with_poly{POLY_DEGREE}_coefs.csv'
df_with_coefs.to_csv(output_path, index=False)
print(f"\nFile salvato in: {output_path}")