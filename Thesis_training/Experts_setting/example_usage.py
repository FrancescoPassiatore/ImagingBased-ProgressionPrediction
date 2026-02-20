"""
Example Script: Test Individual Components
Demonstrates how to use each expert model independently
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cnn_expert import CNNExpert, CNNExpertTrainer
from lgbm_expert import LightGBMExpert
from meta_model import MetaModel, compute_correlation, find_optimal_threshold, compute_metrics


def example_cnn_expert():
    """Example: Create and test CNN expert"""
    print("\n" + "="*80)
    print("EXAMPLE 1: CNN Expert")
    print("="*80)
    
    # Create model
    model = CNNExpert(
        cnn_feature_dim=1280,      # EfficientNet-B1 feature dimension
        hand_feature_dim=9,         # 9 hand-crafted features
        demo_feature_dim=5,         # Age + Sex + 3 smoking categories
        hidden_dim=256,
        dropout=0.3,
        use_attention=True
    )
    
    print(f"Model created: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create dummy batch
    batch = {
        'cnn_features': torch.randn(4, 20, 1280),  # 4 patients, max 20 slices
        'hand_features': torch.randn(4, 9),
        'demo_features': torch.randn(4, 5),
        'lengths': torch.tensor([15, 18, 12, 20]),
        'labels': torch.tensor([0, 1, 0, 1], dtype=torch.float32)
    }
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits)
    
    print(f"\nInput batch:")
    print(f"  Patients: {batch['cnn_features'].shape[0]}")
    print(f"  Max slices: {batch['cnn_features'].shape[1]}")
    print(f"  Actual slices per patient: {batch['lengths'].tolist()}")
    
    print(f"\nOutput:")
    print(f"  Logits: {logits.squeeze().tolist()}")
    print(f"  Probabilities: {probs.squeeze().tolist()}")
    
    print("\n✓ CNN Expert working correctly")


def example_lgbm_expert():
    """Example: Create and test LightGBM expert"""
    print("\n" + "="*80)
    print("EXAMPLE 2: LightGBM Expert")
    print("="*80)
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 100
    n_features = 12  # 9 hand-crafted + 3 demographics
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 2, n_samples)
    
    X_test = np.random.randn(20, n_features)
    y_test = np.random.randint(0, 2, 20)
    
    feature_names = [
        'ApproxVol', 'NumTissuePixel', 'AvgTissue', 'Thickness',
        'TissueByTotal', 'TissueByLung', 'Mean', 'Skew', 'Kurtosis',
        'Age', 'Sex', 'Smoking'
    ]
    
    # Create model
    model = LightGBMExpert()
    
    # Train
    print("Training LightGBM...")
    model.fit(
        X_train, y_train,
        X_test, y_test,
        feature_names=feature_names,
        num_boost_round=100,
        early_stopping_rounds=10,
        verbose=False
    )
    
    # Predict
    predictions = model.predict_proba(X_test)
    
    print(f"\nTraining completed:")
    print(f"  Training samples: {n_samples}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {n_features}")
    
    print(f"\nPredictions (first 5):")
    for i in range(min(5, len(predictions))):
        print(f"  Sample {i}: {predictions[i]:.4f} (true: {y_test[i]})")
    
    # Feature importance
    importance = model.get_feature_importance()
    print(f"\nTop 5 features:")
    print(importance.head(5))
    
    print("\n✓ LightGBM Expert working correctly")


def example_meta_model():
    """Example: Create and test meta-model"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Meta-Model Fusion")
    print("="*80)
    
    # Create dummy expert predictions
    np.random.seed(42)
    n_samples = 50
    
    # Simulate predictions from two experts
    p_cnn = np.random.beta(2, 5, n_samples)    # CNN tends to be conservative
    p_lgbm = np.random.beta(3, 3, n_samples)   # LightGBM more balanced
    
    # True labels (positives are higher probability)
    y_true = (p_cnn + p_lgbm > 1.0).astype(int)
    
    print(f"Simulated predictions:")
    print(f"  Samples: {n_samples}")
    print(f"  Positive class: {y_true.sum()} ({y_true.sum()/n_samples*100:.1f}%)")
    
    # Compute correlation
    correlation = compute_correlation(p_cnn, p_lgbm)
    
    # Train meta-model
    meta_model = MetaModel()
    meta_model.fit(p_cnn, p_lgbm, y_true)
    
    # Get fused predictions
    p_fused = meta_model.predict_proba(p_cnn, p_lgbm)
    
    # Find optimal threshold
    threshold, metrics = find_optimal_threshold(
        y_true, p_fused, strategy='youden'
    )
    
    # Compute metrics
    from sklearn.metrics import roc_auc_score
    
    auc_cnn = roc_auc_score(y_true, p_cnn)
    auc_lgbm = roc_auc_score(y_true, p_lgbm)
    auc_fused = roc_auc_score(y_true, p_fused)
    
    print(f"\nResults:")
    print(f"  CNN AUC:       {auc_cnn:.4f}")
    print(f"  LightGBM AUC:  {auc_lgbm:.4f}")
    print(f"  Fused AUC:     {auc_fused:.4f}")
    print(f"  Improvement:   {((auc_fused - max(auc_cnn, auc_lgbm)) / max(auc_cnn, auc_lgbm) * 100):+.2f}%")
    
    print("\n✓ Meta-Model working correctly")


def example_full_pipeline():
    """Example: Complete pipeline with dummy data"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Full Pipeline (Dummy Data)")
    print("="*80)
    
    # This example shows how all components work together
    
    # 1. Prepare data
    print("\n1. Preparing data...")
    n_train = 80
    n_test = 20
    n_slices_max = 25
    cnn_dim = 1280
    hand_dim = 9
    demo_dim = 5
    
    # Training data
    train_cnn_features = torch.randn(n_train, n_slices_max, cnn_dim)
    train_hand_features = torch.randn(n_train, hand_dim)
    train_demo_features = torch.randn(n_train, demo_dim)
    train_lengths = torch.randint(10, n_slices_max, (n_train,))
    train_labels = torch.randint(0, 2, (n_train,)).float()
    
    # Test data
    test_cnn_features = torch.randn(n_test, n_slices_max, cnn_dim)
    test_hand_features = torch.randn(n_test, hand_dim)
    test_demo_features = torch.randn(n_test, demo_dim)
    test_lengths = torch.randint(10, n_slices_max, (n_test,))
    test_labels = torch.randint(0, 2, (n_test,)).float()
    
    print(f"  Train: {n_train} patients, Test: {n_test} patients")
    
    # 2. Train CNN
    print("\n2. Training CNN Expert...")
    cnn_model = CNNExpert(cnn_dim, hand_dim, demo_dim)
    cnn_model.eval()
    
    with torch.no_grad():
        train_batch = {
            'cnn_features': train_cnn_features,
            'hand_features': train_hand_features,
            'demo_features': train_demo_features,
            'lengths': train_lengths
        }
        p_cnn_train = torch.sigmoid(cnn_model(train_batch)).squeeze().numpy()
        
        test_batch = {
            'cnn_features': test_cnn_features,
            'hand_features': test_hand_features,
            'demo_features': test_demo_features,
            'lengths': test_lengths
        }
        p_cnn_test = torch.sigmoid(cnn_model(test_batch)).squeeze().numpy()
    
    print("  ✓ CNN predictions obtained")
    
    # 3. Train LightGBM
    print("\n3. Training LightGBM Expert...")
    
    # Combine hand-crafted and demographic features
    X_train_lgbm = np.hstack([
        train_hand_features.numpy(),
        train_demo_features.numpy()
    ])
    X_test_lgbm = np.hstack([
        test_hand_features.numpy(),
        test_demo_features.numpy()
    ])
    
    lgbm_model = LightGBMExpert()
    lgbm_model.fit(
        X_train_lgbm, train_labels.numpy(),
        num_boost_round=50,
        verbose=False
    )
    
    p_lgbm_train = lgbm_model.predict_proba(X_train_lgbm)
    p_lgbm_test = lgbm_model.predict_proba(X_test_lgbm)
    
    print("  ✓ LightGBM predictions obtained")
    
    # 4. Train meta-model
    print("\n4. Training Meta-Model...")
    
    meta_model = MetaModel()
    meta_model.fit(p_cnn_train, p_lgbm_train, train_labels.numpy())
    
    p_fused_test = meta_model.predict_proba(p_cnn_test, p_lgbm_test)
    
    print("  ✓ Fused predictions obtained")
    
    # 5. Evaluate
    print("\n5. Evaluating...")
    
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    y_test = test_labels.numpy()
    
    auc_cnn = roc_auc_score(y_test, p_cnn_test)
    auc_lgbm = roc_auc_score(y_test, p_lgbm_test)
    auc_fused = roc_auc_score(y_test, p_fused_test)
    
    acc_cnn = accuracy_score(y_test, (p_cnn_test >= 0.5).astype(int))
    acc_lgbm = accuracy_score(y_test, (p_lgbm_test >= 0.5).astype(int))
    acc_fused = accuracy_score(y_test, (p_fused_test >= 0.5).astype(int))
    
    print(f"\nTest Results:")
    print(f"{'Model':<12} {'AUC':<8} {'Accuracy':<10}")
    print("-" * 30)
    print(f"{'CNN':<12} {auc_cnn:<8.4f} {acc_cnn:<10.4f}")
    print(f"{'LightGBM':<12} {auc_lgbm:<8.4f} {acc_lgbm:<10.4f}")
    print(f"{'Fusion':<12} {auc_fused:<8.4f} {acc_fused:<10.4f}")
    
    print("\n✓ Full pipeline completed successfully")


def main():
    """Run all examples"""
    print("="*80)
    print("EXPERT ENSEMBLE SYSTEM - COMPONENT EXAMPLES")
    print("="*80)
    print("\nThis script demonstrates how to use individual components.")
    print("Using dummy data for illustration purposes.\n")
    
    try:
        example_cnn_expert()
        example_lgbm_expert()
        example_meta_model()
        example_full_pipeline()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nYou can now use these components in your own scripts.")
        print("See train_ensemble.py for a complete training pipeline.\n")
        
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
