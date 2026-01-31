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

from model_train import train_single_fold,aggregate_fold_results



CONFIG = {
    #Paths
    "gt_path" : Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\ground_truth.csv"),
    "ct_scan_path" : Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy"),
    "patient_features_path" : Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\patient_features.csv"), #To update
    "kfold_splits_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\kfold_splits_3fold_stratified.pkl"),
    "resume_from_checkpoint": True,
    #Model parameters
    'backbone' : 'resnet50',
    'image_size' : (224, 224),

    # Output paths
    "features_save_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Features\slice_features_resnet50.csv"),
    "results_save_dir": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\1_progression\results_resnet50_3fold_stratified_config2"),
    
    # Model parameters
    'feature_dim': 2048,
    'aggregation': 'max',  # 'mean', 'max', 'attention', 'combined'
    'hidden_dims': [128,64],
    'dropout': 0.3,
    'use_batch_norm': True,
    
    # Training parameters
    'batch_size': 16,
    'learning_rate': 1e-5,
    'weight_decay': 0.001,
    'epochs': 100,
    'early_stopping_patience': 20,

    'label_smoothing': 0.1,
    'use_scheduler': True,
    'scheduler_patience': 5,
    'scheduler_factor': 0.5,
    'scheduler_min_lr': 1e-6,


}
def main():
    """
    Main training pipeline with enhanced metrics tracking and checkpoint support
    """
    
    # Step 1: Load data
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    
    data_loader = IPFDataLoader(
        csv_path=CONFIG['gt_path'],
        features_path=CONFIG['patient_features_path'],
        npy_dir=CONFIG['ct_scan_path']
    )

    patient_data, features_data = data_loader.get_patient_data()

    print(f"Loaded data for {len(patient_data)} patients.")

    # Step 2: Extract CNN features (with patient grouping)
    print("\n" + "="*70)
    print("STEP 2: EXTRACTING CNN FEATURES (Patient-Grouped)")
    print("="*70)
    
    feature_extractor = CNNFeatureExtractor(
        model_name='resnet50',  # or 'resnet18', 'densenet121'
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    slice_features_df = feature_extractor.extract_features_patient_grouping(
        patient_data=patient_data,
        patients_per_batch=4,  # Process 4 patients at a time
        save_path=None
    )

    # Step 3: Load KFold splits
    print("\n" + "="*70)
    print("STEP 3: LOADING K-FOLD SPLITS")
    print("="*70)
    
    kfold_path = Path(CONFIG['kfold_splits_path'])
    with open(kfold_path, 'rb') as f:
        kfold_splits = pickle.load(f)
    print(f"Loaded KFold splits from {kfold_path}")
    fold_keys = sorted(kfold_splits.keys())
    print(f"Number of folds: {len(fold_keys)}")
    print(f"Fold indices: {fold_keys}")

    # Step 4: Train models with enhanced tracking
    print("\n" + "="*70)
    print("STEP 4: TRAINING MODELS")
    print("="*70)
    print(f"Resume from checkpoint: {CONFIG['resume_from_checkpoint']}")
    
    # Create results directory
    CONFIG['results_save_dir'].mkdir(parents=True, exist_ok=True)
    
    fold_results = []
    
    for fold_n in fold_keys:
        fold_idx = int(fold_n.split("_")[1])  # Assuming fold_n is the index
        fold_data = kfold_splits[fold_n]
        print(f"fold_idx: {fold_idx}")
        print(fold_data)
        print(f"\n{'='*80}")
        print(f"PROCESSING FOLD {fold_idx + 1}/{len(fold_keys)}")
        print(f"{'='*80}")
        
        result = train_single_fold(
            features_df=slice_features_df,
            fold_data=fold_data,
            fold_idx=fold_idx,
            config=CONFIG,
            results_dir=CONFIG['results_save_dir'],
            resume_from_checkpoint=CONFIG['resume_from_checkpoint']
        )
        
        fold_results.append(result)
        
        # Show fold completion status
        if result.get('loaded_from_checkpoint', False):
            print(f"\n✓ Fold {fold_idx} results loaded from checkpoint")
        else:
            print(f"\n✓ Fold {fold_idx} training completed from scratch")
        
        # Print quick summary
        print(f"\nFold {fold_idx} Summary:")
        print(f"  Val AUC: {result['val_auc']:.4f}")
        print(f"  Optimal Threshold: {result['optimal_threshold']:.4f}")
        print(f"  Test AUC (Default): {result['test_metrics_default']['auc']:.4f}")
        print(f"  Test AUC (Optimal): {result['test_metrics_optimal']['auc']:.4f}")

    # Step 5: Aggregate results
    print("\n" + "="*70)
    print("STEP 5: AGGREGATING RESULTS ACROSS ALL FOLDS")
    print("="*70)
    
    summary_df, detailed_df = aggregate_fold_results(
        fold_results=fold_results,
        save_path=CONFIG['results_save_dir']
    )

    # Step 6: Enhanced summary with both thresholds
    print("\n" + "="*70)
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print("="*70)
    
    # Extract key metrics for easier viewing
    val_aucs = [r['val_auc'] for r in fold_results]
    optimal_thresholds = [r['optimal_threshold'] for r in fold_results]
    
    # Default threshold (0.5) metrics
    test_auc_default = [r['test_metrics_default']['auc'] for r in fold_results]
    test_acc_default = [r['test_metrics_default']['accuracy'] for r in fold_results]
    test_prec_default = [r['test_metrics_default']['precision'] for r in fold_results]
    test_rec_default = [r['test_metrics_default']['recall'] for r in fold_results]
    test_f1_default = [r['test_metrics_default']['f1'] for r in fold_results]
    test_spec_default = [r['test_metrics_default']['specificity'] for r in fold_results]
    
    # Optimal threshold metrics
    test_auc_optimal = [r['test_metrics_optimal']['auc'] for r in fold_results]
    test_acc_optimal = [r['test_metrics_optimal']['accuracy'] for r in fold_results]
    test_prec_optimal = [r['test_metrics_optimal']['precision'] for r in fold_results]
    test_rec_optimal = [r['test_metrics_optimal']['recall'] for r in fold_results]
    test_f1_optimal = [r['test_metrics_optimal']['f1'] for r in fold_results]
    test_spec_optimal = [r['test_metrics_optimal']['specificity'] for r in fold_results]
    
    print("\n" + "="*70)
    print("VALIDATION METRICS")
    print("="*70)
    print(f"Val AUC: {pd.Series(val_aucs).mean():.4f} ± {pd.Series(val_aucs).std():.4f}")
    print(f"Optimal Threshold: {pd.Series(optimal_thresholds).mean():.4f} ± {pd.Series(optimal_thresholds).std():.4f}")
    
    print("\n" + "="*70)
    print("TEST METRICS - DEFAULT THRESHOLD (0.5)")
    print("="*70)
    print(f"AUC:         {pd.Series(test_auc_default).mean():.4f} ± {pd.Series(test_auc_default).std():.4f}")
    print(f"Accuracy:    {pd.Series(test_acc_default).mean():.4f} ± {pd.Series(test_acc_default).std():.4f}")
    print(f"Precision:   {pd.Series(test_prec_default).mean():.4f} ± {pd.Series(test_prec_default).std():.4f}")
    print(f"Recall:      {pd.Series(test_rec_default).mean():.4f} ± {pd.Series(test_rec_default).std():.4f}")
    print(f"F1-Score:    {pd.Series(test_f1_default).mean():.4f} ± {pd.Series(test_f1_default).std():.4f}")
    print(f"Specificity: {pd.Series(test_spec_default).mean():.4f} ± {pd.Series(test_spec_default).std():.4f}")
    
    print("\n" + "="*70)
    print("TEST METRICS - OPTIMAL THRESHOLD (from validation)")
    print("="*70)
    print(f"AUC:         {pd.Series(test_auc_optimal).mean():.4f} ± {pd.Series(test_auc_optimal).std():.4f}")
    print(f"Accuracy:    {pd.Series(test_acc_optimal).mean():.4f} ± {pd.Series(test_acc_optimal).std():.4f}")
    print(f"Precision:   {pd.Series(test_prec_optimal).mean():.4f} ± {pd.Series(test_prec_optimal).std():.4f}")
    print(f"Recall:      {pd.Series(test_rec_optimal).mean():.4f} ± {pd.Series(test_rec_optimal).std():.4f}")
    print(f"F1-Score:    {pd.Series(test_f1_optimal).mean():.4f} ± {pd.Series(test_f1_optimal).std():.4f}")
    print(f"Specificity: {pd.Series(test_spec_optimal).mean():.4f} ± {pd.Series(test_spec_optimal).std():.4f}")
    
    # Performance interpretation
    print("\n" + "="*70)
    print("PERFORMANCE INTERPRETATION")
    print("="*70)
    
    mean_test_auc_default = pd.Series(test_auc_default).mean()
    mean_test_auc_optimal = pd.Series(test_auc_optimal).mean()
    
    print(f"\nDefault Threshold (0.5):")
    print(f"  Mean Test AUC: {mean_test_auc_default:.4f}")
    if mean_test_auc_default >= 0.90:
        print("  ✓ Excellent discrimination")
    elif mean_test_auc_default >= 0.80:
        print("  ✓ Good discrimination")
    elif mean_test_auc_default >= 0.70:
        print("  ⚠ Fair discrimination")
    else:
        print("  ✗ Poor discrimination")
    
    print(f"\nOptimal Threshold (from validation):")
    print(f"  Mean Test AUC: {mean_test_auc_optimal:.4f}")
    if mean_test_auc_optimal >= 0.90:
        print("  ✓ Excellent discrimination")
    elif mean_test_auc_optimal >= 0.80:
        print("  ✓ Good discrimination")
    elif mean_test_auc_optimal >= 0.70:
        print("  ⚠ Fair discrimination")
    else:
        print("  ✗ Poor discrimination")
    
    # Threshold analysis summary
    print("\n" + "="*70)
    print("THRESHOLD STRATEGY COMPARISON")
    print("="*70)
    
    print("\nAverage improvement using optimal threshold:")
    print(f"  Accuracy:    {(pd.Series(test_acc_optimal).mean() - pd.Series(test_acc_default).mean()):.4f}")
    print(f"  Precision:   {(pd.Series(test_prec_optimal).mean() - pd.Series(test_prec_default).mean()):.4f}")
    print(f"  Recall:      {(pd.Series(test_rec_optimal).mean() - pd.Series(test_rec_default).mean()):.4f}")
    print(f"  F1-Score:    {(pd.Series(test_f1_optimal).mean() - pd.Series(test_f1_default).mean()):.4f}")
    print(f"  Specificity: {(pd.Series(test_spec_optimal).mean() - pd.Series(test_spec_default).mean()):.4f}")
    
    # Files created
    print("\n" + "="*70)
    print("FILES CREATED")
    print("="*70)
    print(f"\nResults directory: {CONFIG['results_save_dir']}")
    print("\nPer-fold outputs (in fold_X/ subdirectories):")
    print("  - best_model.pth (checkpoint with all metrics)")
    print("  - training_history.png")
    print("  - validation_roc_threshold_analysis.png")
    print("  - test_evaluation_default_threshold.png")
    print("  - test_evaluation_optimal_threshold.png")
    print("  - test_predictions.csv")
    print("  - metrics_summary.csv")
    print("\nAggregate outputs:")
    print("  - aggregate_metrics_summary.csv")
    print("  - detailed_fold_results.csv")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    # Save configuration for reference
    config_df = pd.DataFrame([{
        'parameter': k,
        'value': str(v)
    } for k, v in CONFIG.items()])
    config_df.to_csv(CONFIG['results_save_dir'] / "training_config.csv", index=False)
    print(f"\nConfiguration saved to: {CONFIG['results_save_dir'] / 'training_config.csv'}")
    
    return fold_results, summary_df, detailed_df


if __name__ == "__main__":
    main()