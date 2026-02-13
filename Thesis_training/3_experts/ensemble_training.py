# Training Pipeline for Three-Expert Ensemble
# Trains CNN Expert, LightGBM Expert, and Fusion Layer

from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
from typing import Dict, List, Tuple
import sys

# Import ensemble components
from ensemble_experts import (
    CNNExpert,
    LightGBMExpert,
    FusionExpert,
    ThreeExpertEnsemble,
    plot_expert_comparison,
    plot_fusion_weights
)

# Import utilities (assuming your existing utilities.py)
sys.path.append(str(Path(__file__).parent.parent))
from utilities import (
    IPFDataLoader,
    CNNFeatureExtractor,
    create_dataloaders,
    compute_class_weights
)


class CNNExpertTrainer:
    """Trainer for CNN Expert"""
    
    def __init__(
        self,
        model: CNNExpert,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        class_weights: Tuple[float, float] = None
    ):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Loss function with class weights
        if class_weights is not None:
            pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        self.history = {
            'train_loss': [], 'train_auc': [],
            'val_loss': [], 'val_auc': []
        }
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        
        all_losses = []
        all_preds = []
        all_labels = []
        
        for batch in dataloader:
            cnn_features = batch['cnn_features'].to(self.device)
            demo_features = batch.get('demo_features')
            
            if demo_features is not None:
                demo_features = demo_features.to(self.device)
            
            lengths = batch['lengths'].to(self.device)
            labels = batch['labels'].to(self.device).float()
            
            # Forward pass
            logits = self.model(cnn_features, lengths, demo_features).squeeze(-1)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Record metrics
            all_losses.append(loss.item())
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        from sklearn.metrics import roc_auc_score
        metrics = {
            'loss': np.mean(all_losses),
            'auc': roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
        }
        
        return metrics
    
    def evaluate(self, dataloader):
        """Evaluate on validation/test set"""
        self.model.eval()
        
        all_losses = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                cnn_features = batch['cnn_features'].to(self.device)
                demo_features = batch.get('demo_features')
                
                if demo_features is not None:
                    demo_features = demo_features.to(self.device)
                
                lengths = batch['lengths'].to(self.device)
                labels = batch['labels'].to(self.device).float()
                
                # Forward pass
                logits = self.model(cnn_features, lengths, demo_features).squeeze(-1)
                loss = self.criterion(logits, labels)
                
                # Record metrics
                all_losses.append(loss.item())
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        from sklearn.metrics import roc_auc_score
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        metrics = {
            'loss': np.mean(all_losses),
            'auc': roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5,
            'predictions': all_preds,
            'labels': all_labels
        }
        
        return metrics
    
    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 100,
        early_stopping_patience: int = 15,
        verbose: bool = True
    ):
        """Train the CNN expert"""
        
        best_val_auc = 0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            
            # Scheduler step
            self.scheduler.step(val_metrics['auc'])
            
            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_auc'].append(train_metrics['auc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_auc'].append(val_metrics['auc'])
            
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}")
            
            # Early stopping
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best validation AUC: {best_val_auc:.4f}")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return best_val_auc


def prepare_lgb_features(
    features_df: pd.DataFrame,
    patient_ids: List[str],
    hand_feature_cols: List[str],
    demo_feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features for LightGBM expert (handcrafted + demographics)
    
    Returns:
        X: (n_patients, n_features) - feature matrix
        y: (n_patients,) - labels
    """
    # Filter to specified patients
    patient_df = features_df[features_df['patient_id'].isin(patient_ids)].copy()
    
    # Get one row per patient (all slices have same patient-level features)
    patient_df = patient_df.groupby('patient_id').first().reset_index()
    
    # Extract features
    all_feature_cols = hand_feature_cols + demo_feature_cols
    available_cols = [c for c in all_feature_cols if c in patient_df.columns]
    
    X = patient_df[available_cols].values
    y = patient_df['gt_has_progressed'].values
    
    return X, y


def train_single_fold_ensemble(
    features_df: pd.DataFrame,
    fold_data: dict,
    fold_idx: int,
    config: dict,
    results_dir: Path,
    hand_feature_cols: List[str],
    demo_feature_cols: List[str],
    encoding_info: dict = None
):
    """
    Train all three experts for a single fold
    
    Returns:
        dict with results from all experts
    """
    
    print("\n" + "="*70)
    print(f"TRAINING THREE-EXPERT ENSEMBLE - FOLD {fold_idx}")
    print("="*70)
    
    fold_dir = results_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Split data
    train_ids = fold_data['train']
    val_ids = fold_data['val']
    test_ids = fold_data['test']
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_ids)} patients")
    print(f"  Val: {len(val_ids)} patients")
    print(f"  Test: {len(test_ids)} patients")
    
    # =================================================================
    # EXPERT 1: TRAIN CNN EXPERT (CNN FEATURES ONLY)
    # =================================================================
    
    print("\n" + "="*70)
    print("TRAINING EXPERT 1: CNN ONLY (NO DEMOGRAPHICS)")
    print("="*70)
    
    # CNN Expert uses ONLY CNN features, no demographics
    demo_dim = 0
    
    print(f"\nCNN Expert configuration:")
    print(f"  CNN features: YES")
    print(f"  Hand-crafted features: NO")
    print(f"  Demographic features: NO")
    
    # Create dataloaders for CNN
    # Pass empty list for demographics - CNN expert doesn't use them
    train_loader, val_loader, test_loader = create_dataloaders(
        features_df,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        batch_size=config['batch_size'],
        num_workers=4,
        hand_feature_cols=[],  # CNN expert doesn't use handcrafted features
        demo_feature_cols=[],  # CNN expert doesn't use demographics
        encoding_info=encoding_info
    )
    
    # Get CNN feature dimension from first batch
    sample_batch = next(iter(train_loader))
    cnn_feature_dim = sample_batch['cnn_features'].shape[2]
    
    print(f"\n✓ CNN Expert dimensions:")
    print(f"  CNN feature dimension: {cnn_feature_dim}")
    print(f"  Demographic dimension: {demo_dim}")
    
    # Create CNN expert
    cnn_expert = CNNExpert(
        cnn_feature_dim=cnn_feature_dim,
        demo_feature_dim=demo_dim,
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout'],
        pooling_type=config.get('pooling_type', 'max')
    )
    
    # Train CNN expert
    class_weights = compute_class_weights(features_df, train_ids)
    
    cnn_trainer = CNNExpertTrainer(
        model=cnn_expert,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        class_weights=class_weights
    )
    
    best_cnn_auc = cnn_trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        early_stopping_patience=config['early_stopping_patience'],
        verbose=True
    )
    
    # Get CNN predictions on validation and test sets
    val_cnn_results = cnn_trainer.evaluate(val_loader)
    test_cnn_results = cnn_trainer.evaluate(test_loader)
    
    print(f"\n✓ CNN Expert trained!")
    print(f"  Val AUC: {val_cnn_results['auc']:.4f}")
    print(f"  Test AUC: {test_cnn_results['auc']:.4f}")
    
    # =================================================================
    # EXPERT 2: TRAIN LIGHTGBM EXPERT
    # =================================================================
    
    print("\n" + "="*70)
    print("TRAINING EXPERT 2: HANDCRAFTED FEATURES + DEMOGRAPHICS (LightGBM)")
    print("="*70)
    
    # Prepare features for LightGBM
    # For LightGBM, we need the PREPROCESSED column names (same as CNN uses internally)
    available_hand_cols = [c for c in hand_feature_cols if c in features_df.columns]
    
    # Build list of preprocessed demographic columns (same preprocessing as CNN)
    lgb_demo_cols = []
    if demo_feature_cols:
        if 'Age' in demo_feature_cols and 'Age_normalized' in features_df.columns:
            lgb_demo_cols.append('Age_normalized')
        if 'Sex' in demo_feature_cols and 'Sex_encoded' in features_df.columns:
            lgb_demo_cols.append('Sex_encoded')
        if 'SmokingStatus' in demo_feature_cols and encoding_info:
            smoking_cols = encoding_info.get('smoking_columns', [])
            existing_smoking = [c for c in smoking_cols if c in features_df.columns]
            lgb_demo_cols.extend(existing_smoking)
    
    lgb_feature_cols = available_hand_cols + lgb_demo_cols
    
    print(f"\nLightGBM features:")
    print(f"  Handcrafted: {len(available_hand_cols)}")
    print(f"  Demographics: {len(lgb_demo_cols)} (preprocessed)")
    print(f"  Preprocessed demo columns: {lgb_demo_cols}")
    print(f"  Total: {len(lgb_feature_cols)}")
    
    X_train, y_train = prepare_lgb_features(features_df, train_ids, available_hand_cols, lgb_demo_cols)
    X_val, y_val = prepare_lgb_features(features_df, val_ids, available_hand_cols, lgb_demo_cols)
    X_test, y_test = prepare_lgb_features(features_df, test_ids, available_hand_cols, lgb_demo_cols)
    
    # Verify LightGBM feature dimensions
    print(f"\n✓ LightGBM dimension verification:")
    print(f"  Expected features: {len(lgb_feature_cols)}")
    print(f"  Actual features from X_train: {X_train.shape[1]}")
    print(f"  Hand-crafted: {len(available_hand_cols)}")
    print(f"  Demographics: {len(lgb_demo_cols)}")
    if X_train.shape[1] != len(lgb_feature_cols):
        print(f"  ⚠️ WARNING: Feature dimension mismatch!")
    
    # Architecture summary
    print(f"\n{'='*70}")
    print("EXPERT ARCHITECTURE SUMMARY")
    print(f"{'='*70}")
    print(f"CNN Expert:")
    print(f"  - CNN features only: {cnn_feature_dim} dimensions")
    print(f"  - Demographics: NO")
    print(f"\nLightGBM Expert:")
    print(f"  - Hand-crafted features: {len(available_hand_cols)}")
    print(f"  - Demographics: {len(lgb_demo_cols)} (preprocessed)")
    print(f"  - Total: {len(lgb_feature_cols)}")
    print(f"\nDifferentiated approach: CNN focuses on imaging, LightGBM on tabular data")
    print(f"{'='*70}")
    
    # Create and train LightGBM expert
    lgb_expert = LightGBMExpert(
        params=config.get('lgb_params', None),
        feature_names=lgb_feature_cols
    )
    
    lgb_expert.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_boost_round=config.get('lgb_num_boost_round', 500),
        early_stopping_rounds=config.get('lgb_early_stopping', 50)
    )
    
    # Get LightGBM predictions
    val_lgb_preds = lgb_expert.predict(X_val)
    test_lgb_preds = lgb_expert.predict(X_test)
    
    from sklearn.metrics import roc_auc_score
    val_lgb_auc = roc_auc_score(y_val, val_lgb_preds)
    test_lgb_auc = roc_auc_score(y_test, test_lgb_preds)
    
    print(f"\n✓ LightGBM Expert trained!")
    print(f"  Val AUC: {val_lgb_auc:.4f}")
    print(f"  Test AUC: {test_lgb_auc:.4f}")
    
    # =================================================================
    # EXPERT 3: TRAIN FUSION LAYER
    # =================================================================
    
    print("\n" + "="*70)
    print("TRAINING EXPERT 3: FUSION LAYER")
    print("="*70)
    
    # Create fusion expert
    fusion_expert = FusionExpert(fusion_method=config.get('fusion_method', 'learned'))
    
    # Fit fusion on validation set
    fusion_expert.fit(
        cnn_preds=val_cnn_results['predictions'],
        lgb_preds=val_lgb_preds,
        y_true=y_val,
        initial_weights=(0.5, 0.5)
    )
    
    # Get fusion predictions on test set
    test_fusion_preds = fusion_expert.predict(
        cnn_preds=test_cnn_results['predictions'],
        lgb_preds=test_lgb_preds
    )
    
    test_fusion_auc = roc_auc_score(y_test, test_fusion_preds)
    
    print(f"\n✓ Fusion Expert trained!")
    print(f"  Test AUC: {test_fusion_auc:.4f}")
    
    # =================================================================
    # CREATE ENSEMBLE & EVALUATE
    # =================================================================
    
    print("\n" + "="*70)
    print("FINAL ENSEMBLE EVALUATION")
    print("="*70)
    
    ensemble = ThreeExpertEnsemble(
        cnn_expert=cnn_expert,
        lgb_expert=lgb_expert,
        fusion_expert=fusion_expert,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Evaluate on test set
    test_results = ensemble.evaluate(
        cnn_dataloader=test_loader,
        lgb_features=X_test,
        y_true=y_test,
        threshold=0.5
    )
    
    print("\nTest Set Results:")
    print("-" * 70)
    for expert, metrics in test_results.items():
        print(f"\n{expert.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # =================================================================
    # SAVE RESULTS
    # =================================================================
    
    # Save models
    torch.save({
        'cnn_model_state': cnn_expert.state_dict(),
        'config': config,
        'fold_idx': fold_idx
    }, fold_dir / "cnn_expert.pth")
    
    lgb_expert.save(fold_dir / "lgb_expert.txt")
    
    with open(fold_dir / "fusion_expert.pkl", 'wb') as f:
        pickle.dump(fusion_expert, f)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'patient_id': test_ids,
        'true_label': y_test,
        'cnn_pred': test_cnn_results['predictions'],
        'lgb_pred': test_lgb_preds,
        'fusion_pred': test_fusion_preds
    })
    predictions_df.to_csv(fold_dir / "test_predictions.csv", index=False)
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'expert': ['CNN', 'LightGBM', 'Fusion'],
        'val_auc': [val_cnn_results['auc'], val_lgb_auc, fusion_expert.get_weights()[0] * val_cnn_results['auc'] + fusion_expert.get_weights()[1] * val_lgb_auc],
        'test_auc': [test_results['cnn']['auc'], test_results['lgb']['auc'], test_results['fusion']['auc']],
        'test_accuracy': [test_results['cnn']['accuracy'], test_results['lgb']['accuracy'], test_results['fusion']['accuracy']],
        'test_f1': [test_results['cnn']['f1'], test_results['lgb']['f1'], test_results['fusion']['f1']]
    })
    metrics_df.to_csv(fold_dir / "metrics_summary.csv", index=False)
    
    # Visualizations
    plot_expert_comparison(test_results, save_path=fold_dir / "expert_comparison.png")
    plot_fusion_weights(fusion_expert, save_path=fold_dir / "fusion_weights.png")
    
    # Save LightGBM feature importance
    feature_importance = lgb_expert.get_feature_importance()
    feature_importance.to_csv(fold_dir / "lgb_feature_importance.csv", index=False)
    
    print(f"\n✓ Results saved to: {fold_dir}")
    
    return {
        'fold_idx': fold_idx,
        'cnn_val_auc': val_cnn_results['auc'],
        'lgb_val_auc': val_lgb_auc,
        'fusion_weights': fusion_expert.get_weights(),
        'test_results': test_results
    }

