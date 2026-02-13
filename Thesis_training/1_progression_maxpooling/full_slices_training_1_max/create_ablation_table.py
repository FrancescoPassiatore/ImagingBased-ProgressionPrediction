import pandas as pd
import os

# Path base
base_path = r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\1_progression_maxpooling\full_slices_training_1_max"

# Configurazioni da analizzare
configurations = {
    'CNN-only': 'ablation_cnn_only',
    'CNN+Demo': 'ablation_cnn_demo',
    'CNN+Handcrafted': 'ablation_cnn_hand',
    'CNN+Demo+Hand': 'ablation_full'
}

# Lista per raccogliere i dati
results = []

# Per ogni configurazione, leggi i metrici
for config_name, folder_name in configurations.items():
    file_path = os.path.join(base_path, folder_name, 'aggregate_metrics_summary.csv')
    
    if os.path.exists(file_path):
        # Leggi il file CSV
        df = pd.read_csv(file_path)
        
        # Crea un dizionario per questa configurazione
        row = {'Configuration': config_name}
        
        # Estrai le metriche (usando Optimal threshold)
        metrics_map = {
            'Test AUC (Optimal)': 'AUC',
            'Test Accuracy (Optimal)': 'Accuracy',
            'Test Precision (Optimal)': 'Precision',
            'Test Recall (Optimal)': 'Recall',
            'Test F1 (Optimal)': 'F1',
            'Test Specificity (Optimal)': 'Specificity'
        }
        
        for metric_original, metric_short in metrics_map.items():
            # Trova la riga corrispondente
            metric_row = df[df['Metric'] == metric_original]
            if not metric_row.empty:
                mean = metric_row['Mean'].values[0]
                std = metric_row['Std'].values[0]
                # Formatta come Mean±Std
                row[metric_short] = f"{mean:.3f}±{std:.3f}"
            else:
                row[metric_short] = 'N/A'
        
        results.append(row)
    else:
        print(f"File non trovato: {file_path}")

# Crea il DataFrame finale
final_df = pd.DataFrame(results)

# Riordina le colonne
columns_order = ['Configuration', 'AUC', 'Accuracy', 'Precision', 'Recall', 'F1', 'Specificity']
final_df = final_df[columns_order]

# Salva il file
output_path = os.path.join(base_path, 'ablation_table.csv')
final_df.to_csv(output_path, index=False)

print("\nTabella salvata in:", output_path)
print("\nAnteprima:")
print(final_df.to_string(index=False))

# Crea anche una versione formattata per LaTeX (opzionale)
latex_output = os.path.join(base_path, 'ablation_table.tex')
with open(latex_output, 'w') as f:
    f.write("\\begin{table}[h]\n")
    f.write("\\centering\n")
    f.write("\\caption{Ablation Study Results - Progression Prediction}\n")
    f.write("\\label{tab:ablation_progression}\n")
    f.write("\\begin{tabular}{l" + "c" * (len(columns_order) - 1) + "}\n")
    f.write("\\hline\n")
    f.write(" & ".join(columns_order) + " \\\\\n")
    f.write("\\hline\n")
    
    for _, row in final_df.iterrows():
        row_values = [str(row[col]) for col in columns_order]
        f.write(" & ".join(row_values) + " \\\\\n")
    
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"\nTabella LaTeX salvata in: {latex_output}")
