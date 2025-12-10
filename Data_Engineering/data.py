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
# Carica i dati
data_csv = 'C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/CodeDevelopment/osic-pulmonary-fibrosis-progression/train.csv'
df = pd.read_csv(data_csv)

# Esploriamo prima i dati
print("Shape del dataset:", df.shape)
print("\nPrime righe:")
print(df.head())
print("\nInfo sul dataset:")
print(df.info())
print("\nPazienti unici:", df['Patient'].nunique())
print("\nDistribuzione delle settimane:")
print(df['Weeks'].describe())

# Verifichiamo quanti pazienti hanno Week 0
patients_with_week0 = df[df['Weeks'] == 0]['Patient'].nunique()
total_patients = df['Patient'].nunique()
print(f"\nPazienti con Week 0: {patients_with_week0}/{total_patients}")
print(f"Pazienti SENZA Week 0: {total_patients - patients_with_week0}")

# Feature Engineering: Aggregate per paziente
def create_patient_features(df):
    """
    Crea features aggregate per ogni paziente per il clustering
    """
    patient_features = []
    
    for patient in df['Patient'].unique():
        patient_data = df[df['Patient'] == patient].sort_values('Weeks')
        
        # Features temporali
        weeks = patient_data['Weeks'].values
        fvc = patient_data['FVC'].values
        
        # Calcola slope (tasso di variazione FVC)
        if len(weeks) > 1:
            slope, intercept = np.polyfit(weeks, fvc, 1)
        else:
            slope = 0
            intercept = fvc[0] if len(fvc) > 0 else np.nan
        
        features = {
            'Patient': patient,
            'n_observations': len(patient_data),
            'weeks_range': weeks.max() - weeks.min(),
            'first_week': weeks.min(),
            'last_week': weeks.max(),
            'has_week0': int(0 in weeks),
            
            # FVC features
            'fvc_mean': fvc.mean(),
            'fvc_std': fvc.std() if len(fvc) > 1 else 0,
            'fvc_min': fvc.min(),
            'fvc_max': fvc.max(),
            'fvc_first': fvc[0],
            'fvc_last': fvc[-1],
            'fvc_range': fvc.max() - fvc.min(),
            'fvc_slope': slope,  # Tasso di declino/crescita
            'fvc_intercept': intercept,  # FVC predetto a week 0
            
            # Percent features
            'percent_mean': patient_data['Percent'].mean(),
            'percent_first': patient_data['Percent'].iloc[0],
            
            # Demographic features
            'Age': patient_data['Age'].iloc[0],
            'Sex': patient_data['Sex'].iloc[0],
            'SmokingStatus': patient_data['SmokingStatus'].iloc[0]
        }
        
        patient_features.append(features)
    
    return pd.DataFrame(patient_features)

# Crea il dataframe con le features per paziente
patient_df = create_patient_features(df)

print(f"\nShape: {patient_df.shape}")
print("\nPrime righe:")
print(patient_df.head(10))

print("\nStatistiche FVC slope (tasso di declino):")
print(patient_df['fvc_slope'].describe())

print("\nDistribuzione pazienti con/senza Week 0:")
print(patient_df['has_week0'].value_counts())

# Encode categorical variables
le_sex = LabelEncoder()
le_smoking = LabelEncoder()

patient_df['Sex_encoded'] = le_sex.fit_transform(patient_df['Sex'])
patient_df['SmokingStatus_encoded'] = le_smoking.fit_transform(patient_df['SmokingStatus'])

print("\nEncoding:")
print(f"Sex: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")
print(f"SmokingStatus: {dict(zip(le_smoking.classes_, le_smoking.transform(le_smoking.classes_)))}")


# Seleziona le features numeriche per il clustering
features_for_clustering = [
    'fvc_slope',           # MOLTO IMPORTANTE: tasso di declino
    'fvc_mean',            # Livello medio di FVC
    'fvc_std',             # Variabilità
    'percent_mean',        # Percentuale media
    'Age',                 # Età
    'Sex_encoded',         # Sesso
    'SmokingStatus_encoded', # Stato fumatore
    'n_observations',      # Quante misure abbiamo
    'weeks_range'          # Range temporale
]

X = patient_df[features_for_clustering].copy()

# Normalizzazione (importante per K-Means!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Features per clustering:")
print(X.describe())

# Proviamo diversi numeri di cluster
inertias = []
silhouette_scores = []
K_range = range(2, 10)

from sklearn.metrics import silhouette_score

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot per scegliere il numero ottimale di cluster
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow method
axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_xlabel('Numero di Cluster')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[0].grid(True)

# Silhouette score
axes[1].plot(K_range, silhouette_scores, 'ro-')
axes[1].set_xlabel('Numero di Cluster')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score per K')
axes[1].grid(True)

plt.tight_layout()
plt.show()


# Scegliamo k=4 come punto di partenza (possiamo modificare dopo)
optimal_k = 4
print(f"\n{'='*50}")
print(f"Clustering con K={optimal_k}")
print(f"{'='*50}")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
patient_df['Cluster'] = kmeans.fit_predict(X_scaled)

# Analisi dei cluster
print("\nDistribuzione pazienti per cluster:")
print(patient_df['Cluster'].value_counts().sort_index())

print("\nCaratteristiche medie per cluster:")
cluster_summary = patient_df.groupby('Cluster')[features_for_clustering + ['has_week0']].mean()
print(cluster_summary.round(2))


# Prepara i dati per la predizione
predictions_list = []
models_per_cluster = {}


for cluster_id in range(optimal_k):
    # Pazienti in questo cluster
    cluster_patients = patient_df[patient_df['Cluster'] == cluster_id]['Patient'].values
    cluster_data = df[df['Patient'].isin(cluster_patients)].copy()
    
    print(f"\n{'='*60}")
    print(f"CLUSTER {cluster_id}")
    print(f"{'='*60}")
    print(f"Pazienti: {len(cluster_patients)}")
    print(f"Osservazioni totali: {len(cluster_data)}")
    
    # Fit regressione: FVC ~ Weeks
    X_reg = cluster_data[['Weeks']].values
    y_reg = cluster_data['FVC'].values
    
    model = LinearRegression()
    model.fit(X_reg, y_reg)
    
    # Statistiche del modello in training -> optimistic error
    y_pred = model.predict(X_reg)
    r2 = r2_score(y_reg, y_pred)
    mae = mean_absolute_error(y_reg, y_pred)
    
    print(f"Slope: {model.coef_[0]:.2f} FVC/week")
    print(f"Intercept (FVC a Week 0): {model.intercept_:.2f}")
    print(f"R²: {r2:.3f}")
    print(f"MAE: {mae:.2f}")
    
    models_per_cluster[cluster_id] = {
        'model': model,
        'slope': model.coef_[0],
        'intercept': model.intercept_,
        'r2': r2,
        'mae': mae
    }
    
    # Predici Week 0 per pazienti SENZA Week 0 in questo cluster
    cluster_patients_no_week0 = patient_df[
        (patient_df['Cluster'] == cluster_id) & 
        (patient_df['has_week0'] == 0)
    ]
    
    print(f"\nPazienti da predire (senza Week 0): {len(cluster_patients_no_week0)}")
    
    for _, patient_row in cluster_patients_no_week0.iterrows():
        patient_id = patient_row['Patient']
        
        # Usa il modello del cluster per predire a Week 0
        predicted_fvc_week0 = model.predict([[0]])[0]
        
        # Oppure usa l'intercept calcolato per il singolo paziente
        # (più accurato se abbiamo il suo slope personale)
        patient_intercept = patient_row['fvc_intercept']
        
        predictions_list.append({
            'Patient': patient_id,
            'Cluster': cluster_id,
            'predicted_FVC_week0_cluster': predicted_fvc_week0,
            'predicted_FVC_week0_personal': patient_intercept,
            'fvc_mean': patient_row['fvc_mean'],
            'fvc_slope': patient_row['fvc_slope'],
            'first_week': patient_row['first_week']
        })

# Crea dataframe con predizioni
predictions_df = pd.DataFrame(predictions_list)


# Confronto tra metodo cluster e metodo personale
print("\nConfronto metodi di predizione:")
print(predictions_df[['predicted_FVC_week0_cluster', 'predicted_FVC_week0_personal']].describe())

# Visualizza le predizioni per cluster
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for cluster_id in range(optimal_k):
    ax = axes[cluster_id]
    
    # Dati del cluster
    cluster_patients = patient_df[patient_df['Cluster'] == cluster_id]['Patient'].values
    cluster_data = df[df['Patient'].isin(cluster_patients)].copy()
    
    # Plot i dati reali
    for patient in cluster_patients[:10]:  # Plot solo primi 10 per chiarezza
        patient_data = cluster_data[cluster_data['Patient'] == patient]
        ax.plot(patient_data['Weeks'], patient_data['FVC'], 'o-', alpha=0.3, linewidth=1)
    
    # Plot la regressione del cluster
    weeks_range = np.linspace(cluster_data['Weeks'].min(), cluster_data['Weeks'].max(), 100)
    model = models_per_cluster[cluster_id]['model']
    fvc_pred = model.predict(weeks_range.reshape(-1, 1))
    ax.plot(weeks_range, fvc_pred, 'r-', linewidth=3, label=f'Cluster {cluster_id} trend')
    
    # Marca Week 0
    ax.axvline(x=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Week 0')
    
    ax.set_xlabel('Weeks')
    ax.set_ylabel('FVC')
    ax.set_title(f'Cluster {cluster_id} (n={len(cluster_patients)})\nSlope: {model.coef_[0]:.2f}, R²: {models_per_cluster[cluster_id]["r2"]:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Pazienti con Week 0 (per validazione)
patients_with_week0 = patient_df[patient_df['has_week0'] == 1].copy()

validation_results = []

for _, patient_row in patients_with_week0.iterrows():
    patient_id = patient_row['Patient']
    
    # Valore reale di FVC a Week 0
    real_fvc_week0 = df[(df['Patient'] == patient_id) & (df['Weeks'] == 0)]['FVC'].values[0]
    
    # Predizione usando intercept personale
    predicted_fvc = patient_row['fvc_intercept']
    
    # Predizione usando cluster
    cluster_id = patient_row['Cluster']
    predicted_fvc_cluster = models_per_cluster[cluster_id]['intercept']
    
    error_personal = abs(real_fvc_week0 - predicted_fvc)
    error_cluster = abs(real_fvc_week0 - predicted_fvc_cluster)
    
    validation_results.append({
        'Patient': patient_id,
        'Cluster': cluster_id,
        'Real_FVC_Week0': real_fvc_week0,
        'Predicted_Personal': predicted_fvc,
        'Predicted_Cluster': predicted_fvc_cluster,
        'Error_Personal': error_personal,
        'Error_Cluster': error_cluster,
        'Percent_Error_Personal': (error_personal / real_fvc_week0) * 100,
        'Percent_Error_Cluster': (error_cluster / real_fvc_week0) * 100
    })

validation_df = pd.DataFrame(validation_results)

print(f"\nPazienti con Week 0 disponibili: {len(validation_df)}")
print("\n" + "="*60)
print("CONFRONTO ERRORI")
print("="*60)

print("\nMetodo PERSONALE (regressione individuale):")
print(f"MAE medio: {validation_df['Error_Personal'].mean():.2f}")
print(f"Errore percentuale medio: {validation_df['Percent_Error_Personal'].mean():.2f}%")
print("\nMetodo CLUSTER (regressione per gruppo):")
print(f"MAE medio: {validation_df['Error_Cluster'].mean():.2f}")
print(f"Errore percentuale medio: {validation_df['Percent_Error_Cluster'].mean():.2f}%")

# Print table
print("\n" + "="*60)
print("DETTAGLIO VALIDAZIONE")
print("="*60)
print(validation_df.sort_values('Error_Personal'))

# Prepare
real = validation_df['Real_FVC_Week0'].to_numpy()
pred  = validation_df['Predicted_Personal'].to_numpy()
clusters = validation_df['Cluster'].to_numpy()
errors = np.abs(real - pred)
mae = errors.mean()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (1) Real vs Predicted
ax = axes[0]
sc = ax.scatter(real, pred, c=clusters, cmap='viridis', s=80, alpha=0.7)
mn, mx = np.min([real.min(), pred.min()]), np.max([real.max(), pred.max()])
ax.plot([mn, mx], [mn, mx], linestyle='--', linewidth=2, label='Perfect prediction')
ax.set_xlim(mn, mx); ax.set_ylim(mn, mx); ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('Real FVC at Week 0')
ax.set_ylabel('Predicted FVC (Personal)')
ax.set_title(f'Personal Method Validation — MAE: {mae:.1f}')
ax.grid(True, alpha=0.3)
leg1 = ax.legend(loc='upper left')
cbar = fig.colorbar(sc, ax=ax, label='Cluster')

# (2) Residuals vs Real (diagnostics)
ax2 = axes[1]
resid = pred - real
ax2.axhline(0, linestyle='--', linewidth=1)
ax2.scatter(real, resid, c=clusters, cmap='viridis', s=60, alpha=0.7)
ax2.set_xlabel('Real FVC at Week 0')
ax2.set_ylabel('Residual (Pred - Real)')
ax2.set_title('Residuals by Real FVC')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# ANALISI AFFIDABILITÀ PREDIZIONI FVC WEEK 0 (VERSIONE CONCISA)
# ============================================================

print("="*70)
print("📊 ANALISI AFFIDABILITÀ PREDIZIONI")
print("="*70)

# Calcola metriche qualità per ogni paziente
quality_metrics = []

for patient_id in df['Patient'].unique():
    patient_data = df[df['Patient'] == patient_id].sort_values('Weeks')
    
    if len(patient_data) >= 2:
        weeks = patient_data['Weeks'].values.reshape(-1, 1)
        fvc = patient_data['FVC'].values
        
        # Regressione lineare
        model = LinearRegression()
        model.fit(weeks, fvc)
        
        quality_metrics.append({
            'Patient': patient_id,
            'R2': r2_score(fvc, model.predict(weeks)),
            'N_obs': len(patient_data),
            'Distance_Week0': abs(weeks.min()),
            'Has_Week0': int(0 in weeks.flatten())
        })

quality_df = pd.DataFrame(quality_metrics)

# Merge con predizioni
reliability_df = predictions_df.merge(quality_df, on='Patient')

# Classificazione affidabilità (score 0-10)
def calc_reliability_score(row):
    score = 0
    # R² (0-4 punti)
    if row['R2'] > 0.7: score += 4
    elif row['R2'] > 0.4: score += 2
    
    # Distanza Week 0 (0-3 punti)
    if row['Distance_Week0'] <= 5: score += 3
    elif row['Distance_Week0'] <= 15: score += 2
    
    # Numero osservazioni (0-3 punti)
    if row['N_obs'] >= 9: score += 3
    elif row['N_obs'] >= 7: score += 1
    
    return score

reliability_df['Score'] = reliability_df.apply(calc_reliability_score, axis=1)
reliability_df['Reliability'] = pd.cut(reliability_df['Score'], 
                                        bins=[0, 4, 7, 10], 
                                        labels=['BASSA', 'MEDIA', 'ALTA'])

# Statistiche
print("\n📈 DISTRIBUZIONE AFFIDABILITÀ:")
for level in ['ALTA', 'MEDIA', 'BASSA']:
    n = (reliability_df['Reliability'] == level).sum()
    pct = n / len(reliability_df) * 100
    emoji = {'ALTA': '🟢', 'MEDIA': '🟡', 'BASSA': '🔴'}[level]
    print(f"{emoji} {level:6s}: {n:3d} pazienti ({pct:5.1f}%)")

print("\n📊 CARATTERISTICHE PER LIVELLO:")
summary = reliability_df.groupby('Reliability')[['R2', 'Distance_Week0', 'N_obs']].mean()
print(summary.round(2))

# Visualizzazione
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Distribuzione
colors = {'ALTA': 'green', 'MEDIA': 'yellow', 'BASSA': 'red'}
counts = reliability_df['Reliability'].value_counts().sort_index()
axes[0].bar(counts.index, counts.values, 
            color=[colors[x] for x in counts.index], 
            edgecolor='black', alpha=0.7)
axes[0].set_ylabel('N. Pazienti')
axes[0].set_title('Distribuzione Affidabilità', fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# Plot 2: R² vs Distance colored by reliability
for level in ['ALTA', 'MEDIA', 'BASSA']:
    subset = reliability_df[reliability_df['Reliability'] == level]
    axes[1].scatter(subset['Distance_Week0'], subset['R2'],
                   label=level, s=80, alpha=0.6, 
                   color=colors[level], edgecolors='black')
axes[1].set_xlabel('Distanza da Week 0 (settimane)')
axes[1].set_ylabel('R²')
axes[1].set_title('Qualità Fit per Affidabilità', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n💡 CONCLUSIONE:")
alta = (reliability_df['Reliability'] == 'ALTA').sum()
media = (reliability_df['Reliability'] == 'MEDIA').sum()
usable = alta + media
print(f"   Predizioni utilizzabili: {usable}/{len(reliability_df)} ({usable/len(reliability_df)*100:.0f}%)")
print(f"   Raccomandazione: usa con cautela predizioni BASSA affidabilità")


print("="*70)
print("ANALISI OTTIMIZZAZIONE FEATURES PER CLUSTERING")
print("="*70)


print("\n1. CORRELAZIONE TRA FEATURES")
print("="*70)

X = patient_df[features_for_clustering].copy()
correlation_matrix = X.corr()

# Plot correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Matrice di Correlazione tra Features')
plt.tight_layout()
plt.show()

print("\nFeatures altamente correlate (|r| > 0.7):")
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            print(f"  {correlation_matrix.columns[i]} <-> {correlation_matrix.columns[j]}: {correlation_matrix.iloc[i, j]:.3f}")




print("\n" + "="*70)
print("2. TEST DIVERSE COMBINAZIONI DI FEATURES")
print("="*70)

feature_sets = {
    'Original (All 9)': features_for_clustering,
    
    'Clinical Only': ['Age', 'Sex_encoded', 'SmokingStatus_encoded'],
    
    'Temporal Only': ['fvc_slope', 'fvc_mean', 'fvc_std'],
    
    'Clinical + Temporal': ['fvc_slope', 'fvc_mean', 'Age', 'Sex_encoded', 'SmokingStatus_encoded'],

    'Slope + Demographics': ['fvc_slope', 'Age', 'Sex_encoded', 'SmokingStatus_encoded'],
    
    'FVC Features': ['fvc_slope', 'fvc_mean', 'fvc_std', 'percent_mean'],
    
    'Minimal': ['fvc_slope', 'Sex_encoded', 'SmokingStatus_encoded']
}

results = []

for set_name, features in feature_sets.items():
    X_test = patient_df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_test)
    
    # Test per k da 2 a 6
    for k in range(2, 7):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(X_scaled)
        
        silhouette = silhouette_score(X_scaled, labels)
        inertia = kmeans.inertia_

        
        results.append({
            'Feature_Set': set_name,
            'N_Features': len(features),
            'K': k,
            'Silhouette': silhouette,
            'Inertia': inertia
        })

results_df = pd.DataFrame(results)

# Trova i migliori per ogni metrica
print("\n📊 MIGLIORI COMBINAZIONI PER METRICA:")
print("\nTop 5 per Silhouette Score (più alto è meglio):")
top_silhouette = results_df.nlargest(5, 'Silhouette')[['Feature_Set', 'K', 'N_Features', 'Silhouette']]
print(top_silhouette.to_string(index=False))


# Single-plot Silhouette chart
fig, ax = plt.subplots(figsize=(10, 6))

for set_name in feature_sets.keys():
    subset = results_df[results_df['Feature_Set'] == set_name].sort_values('K')
    ax.plot(subset['K'], subset['Silhouette'], 'o-', label=set_name, linewidth=2)

ax.set_xlabel('Numero di Cluster (K)')
ax.set_ylabel('Silhouette Score')
ax.set_title('Silhouette Score per Feature Set')
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

plt.tight_layout()
plt.show()


# ============================================================
# CLUSTERING OTTIMIZZATO CON FEATURES RIDOTTE
# ============================================================

print("="*70)
print("CLUSTERING OTTIMIZZATO")
print("="*70)

# Features ottimali
optimal_features = ['fvc_slope', 'Sex_encoded', 'SmokingStatus_encoded']
optimal_k = 5

print(f"\nFeatures selezionate: {optimal_features}")
print(f"Numero di cluster: {optimal_k}")

# Prepara i dati
X_optimal = patient_df[optimal_features].copy()
scaler_optimal = StandardScaler()
X_scaled_optimal = scaler_optimal.fit_transform(X_optimal)

# Clustering
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
patient_df['Cluster_Optimal'] = kmeans_optimal.fit_predict(X_scaled_optimal)

# Metriche
from sklearn.metrics import silhouette_score

silhouette_opt = silhouette_score(X_scaled_optimal, patient_df['Cluster_Optimal'])

print(f"\n📊 METRICHE QUALITÀ:")
print(f"Silhouette Score: {silhouette_opt:.3f}")

# Confronto con clustering originale
print(f"\n📈 MIGLIORAMENTO vs ORIGINALE:")
print(f"Silhouette: 0.25 → {silhouette_opt:.3f} (+{(silhouette_opt-0.25)/0.25*100:.0f}%)")

# ============================================================
# ANALISI DEI NUOVI CLUSTER
# ============================================================

print("\n" + "="*70)
print("1. QUALITÀ DELLE REGRESSIONI INDIVIDUALI")
print("="*70)

# Per ogni paziente, calcola R² e altri indicatori di fit
regression_quality = []

for patient_id in df['Patient'].unique():
    patient_data = df[df['Patient'] == patient_id].sort_values('Weeks')
    
    if len(patient_data) >= 2:
        weeks = patient_data['Weeks'].values.reshape(-1, 1)
        fvc = patient_data['FVC'].values
        
        # Fit regressione
        model = LinearRegression()
        model.fit(weeks, fvc)
        
        # Calcola metriche
        y_pred = model.predict(weeks)
        r2 = r2_score(fvc, y_pred)
        mae = mean_absolute_error(fvc, y_pred)
        
        # Residui
        residuals = fvc - y_pred
        residuals_std = np.std(residuals)
        
        # Distanza da Week 0
        min_week = weeks.min()
        max_week = weeks.max()
        distance_from_week0 = abs(min_week)
        
        regression_quality.append({
            'Patient': patient_id,
            'n_points': len(patient_data),
            'r2': r2,
            'mae': mae,
            'residuals_std': residuals_std,
            'slope': model.coef_[0],
            'min_week': min_week,
            'max_week': max_week,
            'distance_from_week0': distance_from_week0,
            'has_week0': int(0 in weeks.flatten())
        })

regression_quality_df = pd.DataFrame(regression_quality)

# Merge con cluster info
regression_quality_df = regression_quality_df.merge(
    patient_df[['Patient', 'Cluster']], 
    on='Patient'
)

print("\n" + "="*70)
print("CARATTERISTICHE DEI 5 CLUSTER OTTIMALI")
print("="*70)

# Statistiche per cluster
cluster_stats = patient_df.groupby('Cluster_Optimal').agg({
    'Patient': 'count',
    'fvc_slope': 'mean',
    'fvc_mean': 'mean',
    'Age': 'mean',
    'Sex': lambda x: (x == 'Male').sum() / len(x) * 100,  # % maschi
    'SmokingStatus': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown',
    'has_week0': 'sum',
    'fvc_intercept': 'mean'
}).round(2)

cluster_stats.columns = ['N_Pazienti', 'FVC_Slope_Avg', 'FVC_Mean', 'Age_Avg', 
                         'Male_%', 'Smoking_Mode', 'N_with_Week0', 'FVC_Week0_Est']

print("\n")
print(cluster_stats)

# Interpretazione cluster
print("\n" + "="*70)
print("INTERPRETAZIONE CLINICA DEI CLUSTER")
print("="*70)

for cluster_id in range(optimal_k):
    cluster_data = patient_df[patient_df['Cluster_Optimal'] == cluster_id]
    
    print(f"\n🔵 CLUSTER {cluster_id} (n={len(cluster_data)})")
    print(f"   FVC Slope: {cluster_data['fvc_slope'].mean():.2f} ± {cluster_data['fvc_slope'].std():.2f}")
    print(f"   FVC Baseline: {cluster_data['fvc_mean'].mean():.0f}")
    print(f"   Età media: {cluster_data['Age'].mean():.1f} anni")
    
    # Composizione sesso
    sex_counts = cluster_data['Sex'].value_counts()
    print(f"   Sesso: {sex_counts.to_dict()}")
    
    # Smoking status
    smoking_counts = cluster_data['SmokingStatus'].value_counts()
    print(f"   Smoking: {smoking_counts.to_dict()}")
    
    # Interpretazione
    slope_mean = cluster_data['fvc_slope'].mean()
    if slope_mean < -7:
        severity = "⚠️  DECLINO RAPIDO"
    elif slope_mean < -3:
        severity = "⚡ DECLINO MODERATO"
    else:
        severity = "✓ DECLINO LENTO/STABILE"
    
    print(f"   Prognosi: {severity}")



# ============================================================
# VISUALIZZAZIONI
# ============================================================

print("\n" + "="*70)
print("VISUALIZZAZIONI")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for cluster_id in range(optimal_k):
    cluster_patients = patient_df[patient_df['Cluster_Optimal'] == cluster_id]['Patient'].values
    cluster_timeseries = df[df['Patient'].isin(cluster_patients)]
    
    row = cluster_id // 3
    col = cluster_id % 3
    ax = axes[row, col]
    
    # Plot alcune traiettorie
    for patient in np.random.choice(cluster_patients, min(15, len(cluster_patients)), replace=False):
        patient_data = cluster_timeseries[cluster_timeseries['Patient'] == patient]
        ax.plot(patient_data['Weeks'], patient_data['FVC'], alpha=0.3, linewidth=1)
    
    # Media del cluster
    mean_fvc = cluster_timeseries.groupby('Weeks')['FVC'].mean()
    ax.plot(mean_fvc.index, mean_fvc.values, 'r-', linewidth=3, label='Mean trajectory')
    
    ax.axvline(x=0, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax.set_xlabel('Weeks')
    ax.set_ylabel('FVC')
    ax.set_title(f'Cluster {cluster_id} (n={len(cluster_patients)})\nSlope: {patient_df[patient_df["Cluster_Optimal"]==cluster_id]["fvc_slope"].mean():.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Rimuovi subplot extra se K < 6
if optimal_k < 6:
    for i in range(optimal_k, 6):
        row = i // 3
        col = i % 3
        fig.delaxes(axes[row, col])

plt.tight_layout()
plt.show()

print("\n✓ Clustering ottimizzato completato!")


# ============================================================
# PREDIZIONE WEEK 0 CON CLUSTERING OTTIMIZZATO
# ============================================================

print("="*70)
print("PREDIZIONE WEEK 0 CON CLUSTER OTTIMIZZATI")
print("="*70)

# Ri-fai la regressione per ogni cluster ottimale
models_optimal = {}
predictions_optimal = []

for cluster_id in range(optimal_k):
    cluster_patients = patient_df[patient_df['Cluster_Optimal'] == cluster_id]['Patient'].values
    cluster_data = df[df['Patient'].isin(cluster_patients)].copy()
    
    print(f"\n{'='*70}")
    print(f"CLUSTER {cluster_id}")
    print(f"{'='*70}")
    print(f"Pazienti: {len(cluster_patients)}")
    print(f"Osservazioni: {len(cluster_data)}")
    
    # Regressione
    X_reg = cluster_data[['Weeks']].values
    y_reg = cluster_data['FVC'].values
    
    model = LinearRegression()
    model.fit(X_reg, y_reg)
    
    y_pred = model.predict(X_reg)
    r2 = r2_score(y_reg, y_pred)
    mae = mean_absolute_error(y_reg, y_pred)
    
    print(f"Slope cluster: {model.coef_[0]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"R²: {r2:.3f}")
    print(f"MAE: {mae:.2f}")
    
    models_optimal[cluster_id] = {
        'model': model,
        'slope': model.coef_[0],
        'intercept': model.intercept_,
        'r2': r2,
        'mae': mae
    }
    
    # Predici Week 0 per pazienti senza
    cluster_no_week0 = patient_df[
        (patient_df['Cluster_Optimal'] == cluster_id) & 
        (patient_df['has_week0'] == 0)
    ]
    
    for _, patient_row in cluster_no_week0.iterrows():
        predictions_optimal.append({
            'Patient': patient_row['Patient'],
            'Cluster_Optimal': cluster_id,
            'predicted_FVC_week0_cluster_opt': model.predict([[0]])[0],
            'predicted_FVC_week0_personal': patient_row['fvc_intercept'],
            'fvc_slope': patient_row['fvc_slope']
        })

predictions_optimal_df = pd.DataFrame(predictions_optimal)

print(f"\n{'='*70}")
print("VALIDAZIONE CON CLUSTER OTTIMIZZATI")
print(f"{'='*70}")

# Valida sui 18 pazienti con Week 0
validation_optimal = []

for _, patient_row in patient_df[patient_df['has_week0'] == 1].iterrows():
    patient_id = patient_row['Patient']
    cluster_id = patient_row['Cluster_Optimal']
    
    real_fvc = df[(df['Patient'] == patient_id) & (df['Weeks'] == 0)]['FVC'].values[0]
    pred_cluster = models_optimal[cluster_id]['intercept']
    pred_personal = patient_row['fvc_intercept']
    
    validation_optimal.append({
        'Patient': patient_id,
        'Cluster_Optimal': cluster_id,
        'Real_FVC': real_fvc,
        'Pred_Cluster_Opt': pred_cluster,
        'Pred_Personal': pred_personal,
        'Error_Cluster_Opt': abs(real_fvc - pred_cluster),
        'Error_Personal': abs(real_fvc - pred_personal)
    })

validation_optimal_df = pd.DataFrame(validation_optimal)

print(f"\n📊 CONFRONTO ERRORI:")
print(f"\nMetodo CLUSTER OTTIMIZZATO:")
print(f"  MAE: {validation_optimal_df['Error_Cluster_Opt'].mean():.2f}")
print(f"  Mediana: {validation_optimal_df['Error_Cluster_Opt'].median():.2f}")

print(f"\nMetodo PERSONALE:")
print(f"  MAE: {validation_optimal_df['Error_Personal'].mean():.2f}")
print(f"  Mediana: {validation_optimal_df['Error_Personal'].median():.2f}")

print(f"\nMetodo CLUSTER ORIGINALE (4 cluster, 9 features):")
print(f"  MAE: 536.12")

print(f"\n✨ MIGLIORAMENTO:")
print(f"  Cluster 4→5: {536.12:.0f} → {validation_optimal_df['Error_Cluster_Opt'].mean():.0f}")
print(f"  Riduzione errore: {(1 - validation_optimal_df['Error_Cluster_Opt'].mean()/536.12)*100:.0f}%")

# Visualizzazione confronto
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Errori per metodo
methods = ['Cluster\nOriginale\n(4 clust)', 'Cluster\nOttimizzato\n(5 clust)', 'Personale']
errors = [536.12, 
          validation_optimal_df['Error_Cluster_Opt'].mean(),
          validation_optimal_df['Error_Personal'].mean()]

colors_bars = ['red', 'orange', 'green']
bars = axes[0].bar(methods, errors, color=colors_bars, alpha=0.7, edgecolor='black', linewidth=2)
axes[0].set_ylabel('MAE (FVC units)')
axes[0].set_title('Confronto Errore di Predizione')
axes[0].grid(True, alpha=0.3, axis='y')

for i, (bar, err) in enumerate(zip(bars, errors)):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{err:.0f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

# Plot 2: Real vs Predicted (cluster ottimizzato)
axes[1].scatter(validation_optimal_df['Real_FVC'], 
               validation_optimal_df['Pred_Cluster_Opt'],
               c=validation_optimal_df['Cluster_Optimal'],
               s=150, alpha=0.7, edgecolors='black', linewidth=2, cmap='viridis')

min_val = validation_optimal_df[['Real_FVC', 'Pred_Cluster_Opt']].min().min()
max_val = validation_optimal_df[['Real_FVC', 'Pred_Cluster_Opt']].max().max()
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

axes[1].set_xlabel('Real FVC at Week 0')
axes[1].set_ylabel('Predicted FVC (Cluster Method)')
axes[1].set_title(f'Cluster Ottimizzato\nMAE: {validation_optimal_df["Error_Cluster_Opt"].mean():.0f}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Errori per cluster
validation_optimal_df.boxplot(column='Error_Cluster_Opt', by='Cluster_Optimal', ax=axes[2])
axes[2].set_xlabel('Cluster')
axes[2].set_ylabel('Absolute Error')
axes[2].set_title('Errore per Cluster Ottimizzato')
plt.suptitle('')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Statistiche per cluster
print(f"\n{'='*70}")
print("ERRORE DI PREDIZIONE PER CLUSTER")
print(f"{'='*70}")

for cluster_id in range(optimal_k):
    cluster_val = validation_optimal_df[validation_optimal_df['Cluster_Optimal'] == cluster_id]
    if len(cluster_val) > 0:
        print(f"\nCluster {cluster_id} (n={len(cluster_val)} validazioni):")
        print(f"  MAE cluster: {cluster_val['Error_Cluster_Opt'].mean():.2f}")
        print(f"  MAE personal: {cluster_val['Error_Personal'].mean():.2f}")
        print(f"  R² modello: {models_optimal[cluster_id]['r2']:.3f}")


# ============================================================
# STRATEGIA FINALE: APPROCCIO IBRIDO INTELLIGENTE
# ============================================================

print("="*70)
print("IMPLEMENTAZIONE STRATEGIA IBRIDA OTTIMALE")
print("="*70)

# Combina informazioni: cluster ottimizzati + predizioni personali + affidabilità
final_predictions = predictions_df.copy()

# Merge con cluster ottimizzato
final_predictions = final_predictions.merge(
    patient_df[['Patient', 'Cluster_Optimal']], 
    on='Patient'
)

# Merge con metriche di qualità
final_predictions = final_predictions.merge(
    regression_quality_df[['Patient', 'r2', 'distance_from_week0', 'n_points', 'mae', 'residuals_std']],
    on='Patient'
)

cluster_df = pd.DataFrame([
    {
        'Cluster_Optimal': cid,
        'cluster_intercept': models_optimal[cid]['intercept'],
        'cluster_mae_abs': models_optimal[cid]['mae']
    }
    for cid in models_optimal.keys()
])

final_predictions = final_predictions.merge(cluster_df, on='Cluster_Optimal', how='left')


# -------------------------------
# 2) Convert MAE to percent (scale-invariant)
#    Use Week-0 prediction as reference if true Week-0 not available
# -------------------------------
def _safe_pct(mae_abs, ref):
    if mae_abs is None or ref is None:
        return None
    ref = float(ref)
    if ref == 0:
        return None
    return 100.0 * float(mae_abs) / abs(ref)

# Personal MAE (already in regression_quality_df as 'mae' -> assume absolute units)
final_predictions['mae_personal_abs'] = final_predictions['mae']

# Reference scales (prefer the specific method’s Week-0 prediction)
final_predictions['ref_personal'] = final_predictions['predicted_FVC_week0_personal']
final_predictions['ref_cluster']  = final_predictions['cluster_intercept']

final_predictions['mae_personal_pct'] = final_predictions.apply(
    lambda r: _safe_pct(r['mae_personal_abs'], r['ref_personal']), axis=1
)
final_predictions['mae_cluster_pct'] = final_predictions.apply(
    lambda r: _safe_pct(r['cluster_mae_abs'], r['ref_cluster']), axis=1
)

# -------------------------------
# 3) Selection thresholds
# -------------------------------
HIGH_CONF_MAE_PCT = 5.0     # good
MED_CONF_MAE_PCT  = 10.0    # okay
LOW_CONF_MAE_PCT  = 12.0    # risky
ABS_HIGH_ML       = 100.0
ABS_MED_ML        = 200.0
ABS_LOW_ML        = 300.0

MAX_DIST_HIGH     = 15
MAX_DIST_MED      = 35
MIN_OBS_PERSONAL  = 4

# ============================================================
# CRITERIO DI DECISIONE INTELLIGENTE
# ============================================================

def select_best_prediction_method(row):
    dist   = float(row.get('distance_from_week0', 1e9))
    n_obs  = int(row.get('n_points', 0))

    # Predictions
    yhat_p = row.get('predicted_FVC_week0_personal')
    yhat_c = row.get('cluster_intercept')

    # Errors
    mae_p_abs = row.get('mae_personal_abs')   # absolute
    mae_c_abs = row.get('cluster_mae_abs')    # absolute
    mae_p_pct = row.get('mae_personal_pct')   # percent
    mae_c_pct = row.get('mae_cluster_pct')    # percent

    # Is personal even eligible?
    personal_allowed = (n_obs >= MIN_OBS_PERSONAL) and (dist <= MAX_DIST_MED)

    # Choose by lower MAE% (fall back to abs MAE if needed)
    def personal_better():
        p = mae_p_pct if mae_p_pct is not None else mae_p_abs
        c = mae_c_pct if mae_c_pct is not None else mae_c_abs
        if p is None and c is None:
            return None
        if p is None:
            return False
        if c is None:
            return True
        return p < c

    better_is_personal = personal_better()

    if personal_allowed and better_is_personal is True:
        # Confidence tiers (prefer %; use abs as guardrails)
        if mae_p_abs is not None and mae_p_abs > ABS_LOW_ML:
            return 'personal_reject_abs', yhat_p
        if mae_p_pct is not None:
            if mae_p_pct <= HIGH_CONF_MAE_PCT and dist <= MAX_DIST_HIGH:
                return 'personal_high_conf', yhat_p
            elif mae_p_pct <= MED_CONF_MAE_PCT:
                return 'personal_medium_conf', yhat_p
            elif mae_p_pct <= LOW_CONF_MAE_PCT:
                return 'personal_low_conf', yhat_p
            else:
                return 'personal_risky', yhat_p
        # No % available—fallback via absolute bands
        if mae_p_abs is not None:
            if mae_p_abs <= ABS_HIGH_ML and dist <= MAX_DIST_HIGH:
                return 'personal_high_conf_abs', yhat_p
            elif mae_p_abs <= ABS_MED_ML:
                return 'personal_medium_conf_abs', yhat_p
            elif mae_p_abs <= ABS_LOW_ML:
                return 'personal_low_conf_abs', yhat_p
            else:
                return 'personal_risky_abs', yhat_p
        return 'personal_uncertain', yhat_p

    # Else choose cluster (either better or personal not allowed)
    if yhat_c is not None:
        return 'cluster_optimized', yhat_c

    # Last resort
    return 'personal_low_conf', yhat_p

# Applica selezione
method_and_prediction = final_predictions.apply(
    lambda row: select_best_prediction_method(row), axis=1
)

final_predictions['selected_method'] = [x[0] for x in method_and_prediction]
final_predictions['final_FVC_week0'] = [x[1] for x in method_and_prediction]

# Statistiche
print("\n📊 DISTRIBUZIONE METODI SELEZIONATI:")
print(final_predictions['selected_method'].value_counts())

print("\n" + "="*70)
print("CARATTERISTICHE PER METODO SELEZIONATO")
print("="*70)

for method in final_predictions['selected_method'].unique():
    subset = final_predictions[final_predictions['selected_method'] == method]
    print(f"\n{method.upper().replace('_', ' ')} (n={len(subset)}):")
    print(f"  R² medio: {subset['r2'].mean():.3f}")
    print(f"  Distanza Week 0: {subset['distance_from_week0'].mean():.1f} settimane")
    print(f"  MAE medio: {subset['mae'].mean():.1f}")

# ============================================================
# CREAZIONE DATASET FINALE AUGMENTATO
# ============================================================

print("\n" + "="*70)
print("📦 CREAZIONE DATASET AUGMENTATO CON PREDIZIONI WEEK 0")
print("="*70)

# Crea nuove righe per Week 0 predetti
new_week0_rows = []

for _, pred_row in final_predictions.iterrows():
    patient_id = pred_row['Patient']
    
    # Prendi dati demografici dal paziente
    patient_info = df[df['Patient'] == patient_id].iloc[0]
    
    # Usa predizione finale (dal metodo ibrido)
    predicted_fvc = pred_row['final_FVC_week0']
    
    # Calcola Percent predicted proporzionalmente
    patient_data = df[df['Patient'] == patient_id]
    avg_percent_ratio = (patient_data['Percent'] / patient_data['FVC']).mean()
    predicted_percent = predicted_fvc * avg_percent_ratio
    
    new_row = {
        'Patient': patient_id,
        'Weeks': 0,
        'FVC': int(round(predicted_fvc)),
        'Percent': round(predicted_percent, 6),
        'Age': patient_info['Age'],
        'Sex': patient_info['Sex'],
        'SmokingStatus': patient_info['SmokingStatus'],
        'Predicted': True,  # Flag predizione
        'Prediction_Method': pred_row['selected_method']
    }
    
    new_week0_rows.append(new_row)

# Aggiungi flag ai dati originali
df['Predicted'] = False
df['Prediction_Method'] = None

# Crea dataset augmentato
df_augmented = pd.concat([df, pd.DataFrame(new_week0_rows)], ignore_index=True)
df_augmented = df_augmented.sort_values(['Patient', 'Weeks']).reset_index(drop=True)

print(f"\n✅ Dataset originale: {len(df)} righe")
print(f"✅ Nuove righe Week 0: {len(new_week0_rows)}")
print(f"✅ Dataset augmentato: {len(df_augmented)} righe")

print(f"\n📊 VERIFICA:")
print(f"   Pazienti con Week 0 PRIMA: {patients_with_week0}/{total_patients}")
patients_with_week0_after = df_augmented[df_augmented['Weeks'] == 0]['Patient'].nunique()
print(f"   Pazienti con Week 0 DOPO:  {patients_with_week0_after}/{total_patients}")

# Statistiche per metodo
print(f"\n📈 DISTRIBUZIONE PER METODO:")
method_stats = df_augmented[df_augmented['Predicted'] == True].groupby('Prediction_Method').size()
print(method_stats)

# Esempio paziente
sample_patient = new_week0_rows[0]['Patient']
print(f"\n📋 ESEMPIO: Paziente {sample_patient}")
sample_data = df_augmented[df_augmented['Patient'] == sample_patient][
    ['Weeks', 'FVC', 'Percent', 'Predicted', 'Prediction_Method']
].head(5)
print(sample_data.to_string(index=False))

# ============================================================
# SALVA DATASET AUGMENTATO
# ============================================================

output_augmented = 'C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/CodeDevelopment/osic-pulmonary-fibrosis-progression/train_augmented_final.csv'
df_augmented.to_csv(output_augmented, index=False)
print(f"\n💾 Dataset augmentato salvato: train_augmented_final.csv")