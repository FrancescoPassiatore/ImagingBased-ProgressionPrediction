# Progression model 
# CT SCAN -> SLICES -> FEATURES -> MODEL -> PROBABILITY RISK OF PROGRESSION -> MAX BETWEEN SLICES 
# ANALYSIS OF THRESHOLD 
# ABLATION STUDY WITH ADDED FEATURES
from pathlib import Path
import pandas as pd
import pickle
import sys

import torch
sys.path.append(str(Path(__file__).parent.parent))
from utilities import (
    CNNFeatureExtractor,
    IPFDataLoader
    )

from model_train import train_single_fold



CONFIG = {
    #Paths
    "gt_path" : Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\ground_truth.csv"),
    "ct_scan_path" : Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy"),
    "patient_features_path" : Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\patient_features.csv"), #To update
    "kfold_splits_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\kfold_splits.pkl"),

    #Model parameters
    'backbone' : 'efficientnet_b1',
    'image_size' : (224, 224),

    # Output paths
    "features_save_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Features\slice_features_efficientnet_b1.csv"),
    "results_save_dir": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\Results"),
    
    # Model parameters
    'backbone': 'efficientnet_b1',
    'feature_dim': 1280,
    'aggregation': 'max',  # 'mean', 'max', 'attention', 'combined'
    'hidden_dims': [512, 256, 128],
    'dropout': 0.5,
    
    # Training parameters
    'batch_size': 8,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 100,
    'early_stopping_patience': 15,
    

}


def main():

    #Load data -> Dictionary [patient id: {gt_has_progressed , npy files, len(npy files)}]
    data_loader = IPFDataLoader(
        csv_path=CONFIG['gt_path'],
        features_path=CONFIG['patient_features_path'],
        npy_dir=CONFIG['ct_scan_path']
    )

    patient_data , features_data = data_loader.get_patient_data()

    print(f"Loaded data for {len(patient_data)} patients.")

    # Step 2: Extract CNN features (with patient grouping)
    print("\n" + "="*70)
    print("STEP 2: EXTRACTING CNN FEATURES (Patient-Grouped)")
    print("="*70)
    
    feature_extractor = CNNFeatureExtractor(
        model_name='efficientnet_b1',  # or 'resnet18', 'densenet121'
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    slice_features_df = feature_extractor.extract_features_patient_grouping(
        patient_data=patient_data,
        patients_per_batch=4,  # Process 4 patients at a time
        save_path=None
    )

    # Step 3: Train models
    print("\n" + "="*70)
    print("STEP 3: TRAINING MODELS")
    print("="*70)
    
    fold_results = []
    
    #Load Kfold from pickle
    kfold_path = Path(CONFIG['kfold_splits_path'])
    with open(kfold_path, 'rb') as f:
        kfold_splits = pickle.load(f)
    print(f"Loaded KFold splits from {kfold_path}")
    fold_keys = sorted(kfold_splits.keys())
    print(f"Folds found: {fold_keys}")
    for fold_idx in fold_keys:
        fold_data = kfold_splits[fold_idx]
        
        result = train_single_fold(
            features_df=slice_features_df,
            fold_data=fold_data,
            fold_idx=fold_idx,
            config=CONFIG,
            results_dir=CONFIG['results_save_dir']
        )
        
    fold_results.append(result)


    #Summary
    print("\n" + "="*70)
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print("="*70)
    
    results_df = pd.DataFrame(fold_results)
    
    print("\nPer-fold results:")
    print(results_df.to_string(index=False))
    
    print(f"\nMean Performance:")
    print(f"  Val AUC: {results_df['val_auc'].mean():.4f} ± {results_df['val_auc'].std():.4f}")
    print(f"  Test AUC: {results_df['test_auc'].mean():.4f} ± {results_df['test_auc'].std():.4f}")
    print(f"  Test Accuracy: {results_df['test_acc'].mean():.4f} ± {results_df['test_acc'].std():.4f}")
    print(f"  Test F1-Score: {results_df['test_f1'].mean():.4f} ± {results_df['test_f1'].std():.4f}")
    
    # Save summary
    results_df.to_csv(CONFIG['results_save_dir'] / "kfold_summary.csv", index=False)
    
    # Performance interpretation
    mean_test_auc = results_df['test_auc'].mean()
    print("\nPerformance Interpretation:")
    print(mean_test_auc)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {CONFIG['results_save_dir']}")



if __name__ == "__main__":
    main()