"""
Print sample CNN slope predictions from a specified predictions directory.
"""
import pickle
from pathlib import Path

# Path to the CNN predictions directory
PREDICTIONS_DIR = Path(r'D:/FrancescoP/ImagingBased-ProgressionPrediction/Training_2/CNN_Training/Cyclic_kfold/predictions_trainings/predictions_mse')

N_FOLDS = 5  # Adjust if needed
N_SAMPLES = 5  # Number of predictions to print per fold

for fold_idx in range(N_FOLDS):
    pred_path = PREDICTIONS_DIR / f'cnn_predictions_fold{fold_idx}.pkl'
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx}")
    print(f"{'='*60}")
    if not pred_path.exists():
        print(f"  ⚠️ Predictions not found: {pred_path}")
        continue
    with open(pred_path, 'rb') as f:
        preds = pickle.load(f)
    for split in ['train', 'val', 'test']:
        if split in preds:
            split_preds = preds[split]
            print(f"\n  {split.upper()} predictions (showing up to {N_SAMPLES}):")
            for i, (pid, slope) in enumerate(list(split_preds.items())[:N_SAMPLES]):
                print(f"    Patient: {pid} | Predicted slope: {slope:.4f}")
        else:
            print(f"  No predictions for split: {split}")
