import pandas as pd
import numpy as np

# Per ogni fold, carica le predictions
for fold in range(5):
    
    preds = pd.read_csv(f"D:\\FrancescoP\\ImagingBased-ProgressionPrediction\\Thesis_training\\1_progression\\results_resnet50_config_1\\fold_{fold}\\test_predictions.csv")
    
    print(f"\n=== FOLD {fold} ===")
    
    # Distribuzione delle probabilità
    print("\nPredicted Probability Distribution:")
    print(preds['predicted_prob'].describe())
    
    # Quante predictions > 0.5?
    high_prob = (preds['predicted_prob'] > 0.5).sum()
    print(f"\nPredictions > 0.5: {high_prob}/{len(preds)}")
    
    # Correlazione con la vera label
    from scipy.stats import pearsonr
    corr, pval = pearsonr(preds['true_label'], preds['predicted_prob'])
    print(f"Correlation with true label: {corr:.3f} (p={pval:.3f})")
    
    # Per ogni paziente, carica le slice probabilities
    # (devi salvare questo nel modello)
    # Vogliamo vedere se il MAX è sempre su una slice particolare