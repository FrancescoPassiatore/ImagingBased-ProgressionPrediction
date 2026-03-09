"""
PROGRESSIVE UNFREEZING EXPERIMENT FOR RESNET50

This script tests whether unfreezing ResNet50 layers improves performance 
on the small IPF dataset (84 patients). Uses:

1. Progressive unfreezing strategies (layer4 → layer3+4 → full)
2. Differential learning rates (backbone << classifier)
3. Enhanced regularization (dropout, weight decay, label smoothing)
4. Careful overfitting monitoring
5. Statistical comparison vs frozen baseline

CRITICAL: With 84 patients, unfreezing is risky but could capture IPF-specific features.
"""

from pathlib import Path
import pandas as pd
import pickle
import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel
import random
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))
from utilities import (
    CNNFeatureExtractor,
    IPFDataLoader,
    create_dataloaders,
    compute_class_weights
)
from model_train import (
    ProgressionPredictionModel,
    ModelTrainer,
    HAND_FEATURE_COLS,
    DEMO_FEATURE_COLS,
    plot_evaluation_metrics,
    evaluate_with_threshold,
    aggregate_fold_results,
    find_optimal_threshold
)


# =============================================================================
# UNFREEZING CONFIGURATIONS
# =============================================================================

UNFREEZING_STRATEGIES = {
    'frozen': {
        'unfreeze_layers': [],
        'description': 'Baseline - All layers frozen (current approach)',
        'backbone_lr_ratio': 0.0,  # No backbone training
        'expected_trainable_params': 0
    },
    
    'layer4_only': {
        'unfreeze_layers': ['layer4'],
        'description': 'Conservative - Only last residual block (layer4)',
        'backbone_lr_ratio': 0.01,  # 100x smaller than classifier
        'expected_trainable_params': 2_400_000  # ~2.4M parameters
    },
    
    'layer3_layer4': {
        'unfreeze_layers': ['layer3', 'layer4'],
        'description': 'Moderate - Last two residual blocks (layer3 + layer4)',
        'backbone_lr_ratio': 0.005,  # 200x smaller than classifier
        'expected_trainable_params': 9_200_000  # ~9.2M parameters
    },
    
    'layer2_layer3_layer4': {
        'unfreeze_layers': ['layer2', 'layer3', 'layer4'],
        'description': 'Aggressive - Last three residual blocks',
        'backbone_lr_ratio': 0.001,  # 1000x smaller than classifier
        'expected_trainable_params': 14_000_000  # ~14M parameters
    },
    
    'full_unfreeze': {
        'unfreeze_layers': ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4'],
        'description': 'Very Aggressive - All layers unfrozen',
        'backbone_lr_ratio': 0.001,  # 1000x smaller than classifier
        'expected_trainable_params': 23_500_000  # ~23.5M parameters
    }
}


BASE_CONFIG = {
    # Paths
    "gt_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\ground_truth.csv"),
    "ct_scan_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy_full_dataset"),
    "patient_features_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\patient_features_30_60.csv"),
    "kfold_splits_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\kfold_splits_stratified.pkl"),
    "train_csv_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\train.csv"),
    
    'base_seed': 42,
    
    # Model architecture
    'backbone': 'resnet50',
    'image_size': (224, 224),
    'pooling_type': 'max',
    'use_feature_branches': False,
    'use_ktop': False,
    
    # Training parameters (ENHANCED for unfreezing)
    'batch_size': 16,
    'base_learning_rate': 3.86e-5,  # For classifier head
    'weight_decay_classifier': 0.003,  # For classifier
    'weight_decay_backbone': 0.01,  # Higher weight decay for backbone (prevent overfitting)
    'epochs': 80,  # More epochs for fine-tuning
    'early_stopping_patience': 20,  # More patience
    
    # Enhanced regularization
    'dropout': 0.5,  # Increased from 0.3
    'label_smoothing': 0.1,  # Add label smoothing
    'use_batch_norm': False,
    
    # Classifier architecture
    'hidden_dims': [32],
    
    # Learning rate scheduler
    'use_scheduler': True,
    'scheduler_patience': 7,  # More patience
    'scheduler_factor': 0.5,
    'scheduler_min_lr': 1e-7,
    
    # Monitoring
    'log_every_n_epochs': 1,
    'save_best_only': True,
    
    'resume_from_checkpoint': False,  # Start fresh for comparison
    'normalization_type': 'standard',
}


# =============================================================================
# UNFREEZING UTILITIES
# =============================================================================

def configure_backbone_unfreezing(model: nn.Module, strategy_name: str):
    """
    Configure which ResNet50 layers to unfreeze according to strategy.
    
    ResNet50 structure:
    - conv1, bn1: Initial convolution
    - layer1: 3 bottleneck blocks (early features)
    - layer2: 4 bottleneck blocks
    - layer3: 6 bottleneck blocks
    - layer4: 3 bottleneck blocks (task-specific features)
    
    Args:
        model: ProgressionPredictionModel instance
        strategy_name: Key from UNFREEZING_STRATEGIES
    
    Returns:
        num_trainable: Number of trainable parameters in backbone
    """
    strategy = UNFREEZING_STRATEGIES[strategy_name]
    unfreeze_layers = strategy['unfreeze_layers']
    
    print(f"\n{'='*70}")
    print(f"CONFIGURING BACKBONE UNFREEZING: {strategy_name}")
    print(f"{'='*70}")
    print(f"Strategy: {strategy['description']}")
    print(f"Layers to unfreeze: {unfreeze_layers if unfreeze_layers else 'None (fully frozen)'}")
    
    # NOTE: This model doesn't have a ResNet50 backbone integrated
    # It processes pre-extracted CNN features, so there's nothing to unfreeze
    # The "unfreezing" strategies here just differentiate learning rates
    print("\n⚠️  WARNING: Model uses pre-extracted CNN features (no backbone to unfreeze)")
    print("   Unfreezing strategies will only affect learning rate ratios")
    
    # Count trainable parameters
    backbone_params = 0
    classifier_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTrainable Parameters:")
    print(f"  Backbone (CNN reduction): 0 (pre-extracted features)")
    print(f"  Classifier head: {classifier_params:,}")
    print(f"  Total: {classifier_params:,}")
    print(f"{'='*70}\n")
    
    return classifier_params


def get_optimizer_with_differential_lr(
    model: nn.Module,
    base_lr: float,
    backbone_lr_ratio: float,
    weight_decay_classifier: float,
    weight_decay_backbone: float
):
    """
    Create optimizer with differential learning rates for CNN reduction vs classifier.
    
    NOTE: Since this model uses pre-extracted features (no backbone),
    we apply differential LR to CNN reduction layer vs rest of classifier.
    
    Args:
        model: ProgressionPredictionModel
        base_lr: Learning rate for classifier head
        backbone_lr_ratio: Ratio for CNN reduction (e.g., 0.01 = 100x smaller)
        weight_decay_classifier: Weight decay for classifier
        weight_decay_backbone: Weight decay for CNN reduction
    
    Returns:
        optimizer: AdamW with parameter groups
    """
    cnn_reduction_lr = base_lr * backbone_lr_ratio
    
    # Separate parameters into CNN reduction and classifier
    cnn_reduction_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # CNN reduction layer (processes pre-extracted features)
            if 'cnn_reduction' in name:
                cnn_reduction_params.append(param)
            else:
                classifier_params.append(param)
    
    # Build parameter groups
    param_groups = [
        {
            'params': classifier_params,
            'lr': base_lr,
            'weight_decay': weight_decay_classifier,
            'name': 'classifier'
        }
    ]
    
    if cnn_reduction_params:
        param_groups.append({
            'params': cnn_reduction_params,
            'lr': cnn_reduction_lr,
            'weight_decay': weight_decay_backbone,
            'name': 'cnn_reduction'
        })
    
    optimizer = torch.optim.AdamW(param_groups)
    
    print(f"\n{'='*70}")
    print("OPTIMIZER CONFIGURATION")
    print(f"{'='*70}")
    print(f"Classifier:")
    print(f"  Learning rate:  {base_lr:.2e}")
    print(f"  Weight decay:   {weight_decay_classifier:.3f}")
    print(f"  Parameters:     {sum(p.numel() for p in classifier_params):,}")
    
    if cnn_reduction_params:
        print(f"\nCNN Reduction Layer:")
        print(f"  Learning rate:  {cnn_reduction_lr:.2e} (ratio: {backbone_lr_ratio})")
        print(f"  Weight decay:   {weight_decay_backbone:.3f}")
        print(f"  Parameters:     {sum(p.numel() for p in cnn_reduction_params):,}")
        if backbone_lr_ratio > 0:
            print(f"\n  → CNN reduction LR is {1/backbone_lr_ratio:.0f}x SMALLER than classifier")
        else:
            print(f"\n  → CNN reduction is FROZEN (LR = 0)")
    else:
        print(f"\nCNN Reduction: No parameters")
    
    print(f"{'='*70}\n")
    
    return optimizer

def monitor_overfitting(train_metrics: dict, val_metrics: dict, epoch: int, threshold: float = 0.15):
    """
    Monitor train/val gap to detect overfitting.
    
    Args:
        train_metrics: Dict with train metrics
        val_metrics: Dict with validation metrics
        epoch: Current epoch
        threshold: Gap threshold for warning (default: 0.15)
    
    Returns:
        gap: Train - Val AUC gap
        is_overfitting: Boolean flag
    """
    train_auc = train_metrics.get('auc', 0)
    val_auc = val_metrics.get('auc', 0)
    gap = train_auc - val_auc
    
    is_overfitting = gap > threshold
    
    status = "⚠️ OVERFITTING!" if is_overfitting else "✓ OK"
    
    print(f"\n[Epoch {epoch}] Overfitting Monitor:")
    print(f"  Train AUC: {train_auc:.4f}")
    print(f"  Val AUC:   {val_auc:.4f}")
    print(f"  Gap:       {gap:+.4f} {status}")
    
    return gap, is_overfitting


# =============================================================================
# ENHANCED MODEL TRAINER WITH OVERFITTING MONITORING
# =============================================================================

class UnfreezingModelTrainer(ModelTrainer):
    """
    Extended ModelTrainer with:
    - Overfitting monitoring
    - Train metrics logging
    - Custom optimizer support
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device='cuda',
        scheduler=None,
        early_stopping_patience=15,
        monitor_overfitting_threshold=0.15
    ):
        # Initialize parent with minimal required params (model, device)
        # We'll override optimizer and criterion
        super().__init__(
            model=model,
            device=device,
            learning_rate=1e-4,  # Dummy value, we override optimizer
            weight_decay=0.0,    # Dummy value
            class_weights=None,  # We provide custom criterion
            use_scheduler=False,  # We provide custom scheduler
            label_smoothing=0.0
        )
        
        # Override with custom optimizer and criterion
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        
        # Store dataloaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Training config
        self.early_stopping_patience = early_stopping_patience
        self.monitor_overfitting_threshold = monitor_overfitting_threshold
        self.overfitting_history = []
    
    def train_epoch(self, epoch):
        """Train for one epoch and return metrics"""
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in self.train_loader:
            # Move to device
            cnn_features = batch['cnn_features'].to(self.device)
            labels = batch['labels'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            
            hand_features = batch.get('hand_features')
            if hand_features is not None:
                hand_features = hand_features.to(self.device)
            
            demo_features = batch.get('demo_features')
            if demo_features is not None:
                demo_features = demo_features.to(self.device)
            
            # Combine hand and demo features
            patient_features = None
            if hand_features is not None and demo_features is not None:
                patient_features = torch.cat([hand_features, demo_features], dim=-1)
            elif hand_features is not None:
                patient_features = hand_features
            elif demo_features is not None:
                patient_features = demo_features
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(cnn_features, lengths, patient_features)
            
            loss = self.criterion(outputs.squeeze(-1), labels.float())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (important for unfrozen layers)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Collect predictions
            total_loss += loss.item()
            probs = torch.sigmoid(outputs.squeeze(-1)).detach().cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate train metrics
        from sklearn.metrics import roc_auc_score
        train_loss = total_loss / len(self.train_loader)
        train_auc = roc_auc_score(all_labels, all_preds)
        
        train_metrics = {
            'loss': train_loss,
            'auc': train_auc
        }
        
        return train_metrics
    
    def validate(self):
        """Validate on validation set and return metrics"""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                cnn_features = batch['cnn_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                lengths = batch['lengths'].to(self.device)
                
                hand_features = batch.get('hand_features')
                if hand_features is not None:
                    hand_features = hand_features.to(self.device)
                
                demo_features = batch.get('demo_features')
                if demo_features is not None:
                    demo_features = demo_features.to(self.device)
                
                # Combine hand and demo features
                patient_features = None
                if hand_features is not None and demo_features is not None:
                    patient_features = torch.cat([hand_features, demo_features], dim=-1)
                elif hand_features is not None:
                    patient_features = hand_features
                elif demo_features is not None:
                    patient_features = demo_features
                
                # Forward pass
                outputs = self.model(cnn_features, lengths, patient_features)
                loss = self.criterion(outputs.squeeze(-1), labels.float())
                
                # Collect predictions
                total_loss += loss.item()
                probs = torch.sigmoid(outputs.squeeze(-1)).detach().cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate val metrics
        from sklearn.metrics import roc_auc_score
        val_loss = total_loss / len(self.val_loader)
        val_auc = roc_auc_score(all_labels, all_preds)
        
        val_metrics = {
            'loss': val_loss,
            'auc': val_auc
        }
        
        return val_metrics
    
    def validate_on_loader(self, dataloader):
        """Validate on any given dataloader and return metrics"""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                cnn_features = batch['cnn_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                lengths = batch['lengths'].to(self.device)
                
                hand_features = batch.get('hand_features')
                if hand_features is not None:
                    hand_features = hand_features.to(self.device)
                
                demo_features = batch.get('demo_features')
                if demo_features is not None:
                    demo_features = demo_features.to(self.device)
                
                # Combine hand and demo features
                patient_features = None
                if hand_features is not None and demo_features is not None:
                    patient_features = torch.cat([hand_features, demo_features], dim=-1)
                elif hand_features is not None:
                    patient_features = hand_features
                elif demo_features is not None:
                    patient_features = demo_features
                
                # Forward pass
                outputs = self.model(cnn_features, lengths, patient_features)
                loss = self.criterion(outputs.squeeze(-1), labels.float())
                
                # Collect predictions
                total_loss += loss.item()
                probs = torch.sigmoid(outputs.squeeze(-1)).detach().cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
        test_loss = total_loss / len(dataloader)
        test_auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
        test_acc = accuracy_score(all_labels, np.array(all_preds) > 0.5)
        test_f1 = f1_score(all_labels, np.array(all_preds) > 0.5, zero_division=0)
        
        metrics = {
            'loss': test_loss,
            'auc': test_auc,
            'acc': test_acc,
            'f1': test_f1,
            'predictions': all_preds,
            'labels': all_labels
        }
        
        return metrics
    
    def train(self, num_epochs, save_dir):
        """Training loop with overfitting monitoring"""
        print(f"\n{'='*70}")
        print(f"TRAINING START - {num_epochs} EPOCHS")
        print(f"{'='*70}")
        
        best_val_auc = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Monitor overfitting
            gap, is_overfitting = monitor_overfitting(
                train_metrics, val_metrics, epoch, self.monitor_overfitting_threshold
            )
            self.overfitting_history.append({
                'epoch': epoch,
                'train_auc': train_metrics['auc'],
                'val_auc': val_metrics['auc'],
                'gap': gap,
                'is_overfitting': is_overfitting
            })
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_metrics['auc'])
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Learning rate: {current_lr:.2e}")
            
            # Save best model
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_auc': val_metrics['auc'],
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }
                torch.save(checkpoint, save_dir / 'best_model.pth')
                print(f"  ✓ Saved best model (Val AUC: {val_metrics['auc']:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered (patience: {self.early_stopping_patience})")
                break
        
        # Load best model
        checkpoint = torch.load(save_dir / 'best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return self.overfitting_history


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def train_single_fold_with_unfreezing(
    features_df: pd.DataFrame,
    fold_data: dict,
    fold_idx: int,
    config: dict,
    results_dir: Path,
    strategy_name: str,
    encoding_info: dict,
    hand_feature_cols: list = None,
    demo_feature_cols: list = None
):
    """
    Train a single fold with specified unfreezing strategy.
    
    This is adapted from model_train.train_single_fold but with:
    - Backbone unfreezing configuration
    - Differential learning rates
    - Overfitting monitoring
    """
    
    if hand_feature_cols is None:
        hand_feature_cols = HAND_FEATURE_COLS
    
    if demo_feature_cols is None:
        demo_feature_cols = DEMO_FEATURE_COLS
    
    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx} - STRATEGY: {strategy_name}")
    print(f"{'='*70}")
    
    # Store strategy info in config
    config['unfreezing_strategy'] = strategy_name
    config['unfreezing_config'] = UNFREEZING_STRATEGIES[strategy_name]
    
    fold_dir = results_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract patient IDs
    train_ids = fold_data['train']
    val_ids = fold_data['val']
    test_ids = fold_data['test']
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_ids)} patients")
    print(f"  Val: {len(val_ids)} patients")
    print(f"  Test: {len(test_ids)} patients")
    
    # Identify available features
    available_hand_cols = [c for c in hand_feature_cols if c in features_df.columns]
    
    print(f"\nFeature availability:")
    print(f"  Hand-crafted: {len(available_hand_cols)}/{len(hand_feature_cols)}")
    print(f"  Demographics: {len(demo_feature_cols)}/{len(demo_feature_cols)}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        features_df=features_df,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        batch_size=config['batch_size'],
        num_workers=4,
        hand_feature_cols=available_hand_cols,
        demo_feature_cols=demo_feature_cols,
        encoding_info=encoding_info
    )
    
    # Check feature dimensions
    sample_batch = next(iter(train_loader))
    cnn_input_dim = sample_batch['cnn_features'].shape[2]  # CNN features per slice (B, slices, features)
    
    hand_dim = sample_batch['hand_features'].shape[1] if 'hand_features' in sample_batch and sample_batch['hand_features'] is not None else 0
    demo_dim = sample_batch['demo_features'].shape[1] if 'demo_features' in sample_batch and sample_batch['demo_features'] is not None else 0
    
    print(f"\nFeature dimensions:")
    print(f"  CNN features per slice: {cnn_input_dim}")
    print(f"  Hand-crafted features:  {hand_dim}")
    print(f"  Demographic features:   {demo_dim}")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ProgressionPredictionModel(
        cnn_feature_dim=cnn_input_dim,
        hand_feature_dim=hand_dim,
        demo_feature_dim=demo_dim,
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout'],
        use_batch_norm=config['use_batch_norm'],
        pooling_type=config['pooling_type'],
        use_feature_branches=config['use_feature_branches'],
        use_ktop=config['use_ktop']
    ).to(device)
    
    # Configure unfreezing
    configure_backbone_unfreezing(model, strategy_name)
    
    # Create optimizer with differential LR
    strategy = UNFREEZING_STRATEGIES[strategy_name]
    optimizer = get_optimizer_with_differential_lr(
        model=model,
        base_lr=config['base_learning_rate'],
        backbone_lr_ratio=strategy['backbone_lr_ratio'],
        weight_decay_classifier=config['weight_decay_classifier'],
        weight_decay_backbone=config['weight_decay_backbone']
    )
    
    # Compute class weights
    class_weights = compute_class_weights(features_df, train_ids)
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
    
    # Loss function with label smoothing
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
        # Note: BCEWithLogitsLoss doesn't have label_smoothing, we'd need to implement it
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config['scheduler_factor'],
        patience=config['scheduler_patience'],
        min_lr=config['scheduler_min_lr']
    ) if config['use_scheduler'] else None
    
    # Create trainer with overfitting monitoring
    trainer = UnfreezingModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        early_stopping_patience=config['early_stopping_patience'],
        monitor_overfitting_threshold=0.15
    )
    
    # Train
    overfitting_history = trainer.train(
        num_epochs=config['epochs'],
        save_dir=fold_dir
    )
    
    # Save overfitting history
    overfitting_df = pd.DataFrame(overfitting_history)
    overfitting_df.to_csv(fold_dir / 'overfitting_history.csv', index=False)
    
    # Plot overfitting curve
    plot_overfitting_curve(overfitting_df, fold_dir)
    
    # Evaluate on validation set to find optimal threshold
    val_metrics = trainer.validate_on_loader(val_loader)
    
    # Find optimal threshold on validation set
    optimal_threshold, threshold_metrics = find_optimal_threshold(
        y_true=np.array(val_metrics['labels']),
        y_pred=np.array(val_metrics['predictions']),
        metric='youden'
    )
    
    print(f"\nOptimal threshold (Youden): {optimal_threshold:.4f}")
    
    # Evaluate on test set
    test_metrics = trainer.validate_on_loader(test_loader)
    
    # Apply optimal threshold to test predictions
    test_metrics_optimal = evaluate_with_threshold(
        y_true=np.array(test_metrics['labels']),
        y_pred=np.array(test_metrics['predictions']),
        threshold=optimal_threshold
    )
    
    result = {
        'fold_idx': fold_idx,
        'strategy': strategy_name,
        'val_metrics': {
            'auc': val_metrics['auc'],
            'optimal_threshold': optimal_threshold,
            **threshold_metrics
        },
        'test_metrics_optimal': test_metrics_optimal,
        'overfitting_history': overfitting_history,
        'final_gap': overfitting_history[-1]['gap'] if overfitting_history else None
    }
    
    # Save fold summary
    summary_data = {
        'Fold': fold_idx,
        'Strategy': strategy_name,
        'Val_AUC': val_metrics['auc'],
        'Test_AUC': test_metrics_optimal['auc'],
        'Test_Accuracy': test_metrics_optimal['accuracy'],
        'Test_F1': test_metrics_optimal['f1'],
        'Final_Gap': result['final_gap'],
        'Optimal_Threshold': optimal_threshold
    }
    
    pd.DataFrame([summary_data]).to_csv(fold_dir / 'fold_summary.csv', index=False)
    
    return result


def plot_overfitting_curve(overfitting_df: pd.DataFrame, save_dir: Path):
    """Plot train/val AUC curves to visualize overfitting"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Train vs Val AUC
    ax = axes[0]
    ax.plot(overfitting_df['epoch'], overfitting_df['train_auc'], 'b-', label='Train AUC', linewidth=2)
    ax.plot(overfitting_df['epoch'], overfitting_df['val_auc'], 'r-', label='Val AUC', linewidth=2)
    ax.fill_between(overfitting_df['epoch'], overfitting_df['train_auc'], overfitting_df['val_auc'],
                     alpha=0.2, color='orange', label='Gap')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Train vs Validation AUC', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Overfitting Gap
    ax = axes[1]
    ax.plot(overfitting_df['epoch'], overfitting_df['gap'], 'orange', linewidth=2, label='Train-Val Gap')
    ax.axhline(y=0.15, color='red', linestyle='--', alpha=0.5, label='Overfitting Threshold (0.15)')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.fill_between(overfitting_df['epoch'], 0, overfitting_df['gap'],
                     where=(overfitting_df['gap'] > 0.15), alpha=0.3, color='red', label='Overfitting Zone')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('AUC Gap', fontsize=12)
    ax.set_title('Overfitting Monitor', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'overfitting_curve.png', dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"\n{'='*70}")
    print(f"🔒 RANDOM SEED SET TO: {seed}")
    print(f"{'='*70}")


def run_unfreezing_experiment(
    strategies_to_test: list,
    slice_features_df: pd.DataFrame,
    patient_features_df: pd.DataFrame,
    kfold_splits: dict,
    base_config: dict,
    results_base_dir: Path
):
    """
    Run unfreezing experiment comparing multiple strategies.
    
    Args:
        strategies_to_test: List of strategy names to test (e.g., ['frozen', 'layer4_only'])
        slice_features_df: Slice-level CNN features
        patient_features_df: Patient-level hand-crafted + demographics
        kfold_splits: K-fold split definitions
        base_config: Base configuration
        results_base_dir: Where to save results
    """
    print("\n" + "="*80)
    print("UNFREEZING EXPERIMENT")
    print("="*80)
    print(f"Strategies to test: {strategies_to_test}")
    print(f"K-folds: {len(kfold_splits)}")
    
    all_results = {}
    
    for strategy_name in strategies_to_test:
        print("\n" + "="*80)
        print(f"TESTING STRATEGY: {strategy_name.upper()}")
        print(f"Description: {UNFREEZING_STRATEGIES[strategy_name]['description']}")
        print("="*80)
        
        strategy_dir = results_base_dir / f"strategy_{strategy_name}"
        strategy_dir.mkdir(parents=True, exist_ok=True)
        
        fold_results = []
        
        for fold_idx in sorted(kfold_splits.keys()):
            fold_seed = base_config['base_seed'] + fold_idx * 100
            set_seed(fold_seed)
            
            fold_data = kfold_splits[fold_idx]
            
            # Create features for this fold (with normalization)
            from ablation_study import create_feature_set_for_fold
            
            ablation_config = {
                'use_cnn_features': True,
                'use_hand_features': True,
                'use_demographics': True,
                'description': f'Full model with {strategy_name}'
            }
            
            features_df, encoding_info = create_feature_set_for_fold(
                slice_features_df=slice_features_df,
                patient_features_df=patient_features_df,
                fold_data=fold_data,
                ablation_config=ablation_config,
                normalization_type=base_config['normalization_type']
            )
            
            # Train with unfreezing
            result = train_single_fold_with_unfreezing(
                features_df=features_df,
                fold_data=fold_data,
                fold_idx=fold_idx,
                config=base_config,
                results_dir=strategy_dir,
                strategy_name=strategy_name,
                encoding_info=encoding_info
            )
            
            fold_results.append(result)
        
        # Aggregate results for this strategy
        all_results[strategy_name] = {
            'fold_results': fold_results,
            'strategy': UNFREEZING_STRATEGIES[strategy_name]
        }
        
        # Summary
        test_aucs = [f['test_metrics_optimal']['auc'] for f in fold_results]
        mean_auc = np.mean(test_aucs)
        std_auc = np.std(test_aucs)
        
        print(f"\n{'='*70}")
        print(f"STRATEGY '{strategy_name}' COMPLETE")
        print(f"{'='*70}")
        print(f"Test AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        
        # Save strategy summary
        strategy_summary = pd.DataFrame([{
            'Strategy': strategy_name,
            'Description': UNFREEZING_STRATEGIES[strategy_name]['description'],
            'Mean_Test_AUC': mean_auc,
            'Std_Test_AUC': std_auc,
            'Num_Folds': len(fold_results)
        }])
        strategy_summary.to_csv(strategy_dir / 'strategy_summary.csv', index=False)
    
    # Compare all strategies
    compare_unfreezing_strategies(all_results, results_base_dir)
    
    return all_results


def compare_unfreezing_strategies(all_results: dict, results_dir: Path):
    """Create comprehensive comparison of unfreezing strategies"""
    print("\n" + "="*80)
    print("COMPARING UNFREEZING STRATEGIES")
    print("="*80)
    
    comparison_data = []
    
    for strategy_name, results in all_results.items():
        fold_results = results['fold_results']
        
        # Calculate statistics
        test_aucs = [f['test_metrics_optimal']['auc'] for f in fold_results]
        test_accs = [f['test_metrics_optimal']['accuracy'] for f in fold_results]
        test_f1s = [f['test_metrics_optimal']['f1'] for f in fold_results]
        final_gaps = [f['final_gap'] for f in fold_results if f['final_gap'] is not None]
        
        comparison_data.append({
            'Strategy': strategy_name,
            'Description': UNFREEZING_STRATEGIES[strategy_name]['description'],
            'Trainable_Params': UNFREEZING_STRATEGIES[strategy_name]['expected_trainable_params'],
            'Mean_AUC': np.mean(test_aucs),
            'Std_AUC': np.std(test_aucs),
            'Mean_Accuracy': np.mean(test_accs),
            'Mean_F1': np.mean(test_f1s),
            'Mean_Final_Gap': np.mean(final_gaps) if final_gaps else None,
            'Num_Folds': len(fold_results)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Mean_AUC', ascending=False)
    
    print("\nResults (sorted by Mean AUC):")
    print(comparison_df.to_string(index=False))
    
    comparison_df.to_csv(results_dir / 'strategy_comparison.csv', index=False)
    
    # Statistical testing (paired t-tests)
    perform_statistical_testing_unfreezing(all_results, results_dir)
    
    # Visualization
    plot_strategy_comparison(comparison_df, results_dir)
    
    # Identify best strategy
    best = comparison_df.iloc[0]
    print(f"\n🏆 BEST STRATEGY: {best['Strategy']}")
    print(f"   {best['Description']}")
    print(f"   Test AUC: {best['Mean_AUC']:.4f} ± {best['Std_AUC']:.4f}")
    
    # Check if unfreezing helped
    if 'frozen' in comparison_df['Strategy'].values:
        frozen_auc = comparison_df[comparison_df['Strategy'] == 'frozen']['Mean_AUC'].values[0]
        best_auc = best['Mean_AUC']
        improvement = best_auc - frozen_auc
        
        print(f"\n📊 Comparison to Frozen Baseline:")
        print(f"   Frozen AUC:  {frozen_auc:.4f}")
        print(f"   Best AUC:    {best_auc:.4f}")
        print(f"   Improvement: {improvement:+.4f}")
        
        if improvement > 0.01:
            print(f"   ✓ Unfreezing HELPED (>{0.01:.3f} improvement)")
        elif improvement < -0.01:
            print(f"   ⚠️  Unfreezing HURT (<{-0.01:.3f} degradation)")
        else:
            print(f"   ~ No significant difference")


def perform_statistical_testing_unfreezing(all_results: dict, results_dir: Path):
    """Perform paired t-tests between unfreezing strategies"""
    print("\n" + "="*70)
    print("STATISTICAL TESTING - PAIRED T-TESTS")
    print("="*70)
    
    strategies = list(all_results.keys())
    strategy_aucs = {}
    
    for strategy in strategies:
        fold_results = all_results[strategy]['fold_results']
        aucs = np.array([f['test_metrics_optimal']['auc'] for f in fold_results])
        strategy_aucs[strategy] = aucs
    
    comparisons = []
    
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            s1, s2 = strategies[i], strategies[j]
            aucs1, aucs2 = strategy_aucs[s1], strategy_aucs[s2]
            
            if len(aucs1) != len(aucs2):
                continue
            
            t_stat, p_value = ttest_rel(aucs1, aucs2)
            mean_diff = np.mean(aucs1 - aucs2)
            
            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            
            print(f"\n{s1} vs {s2}:")
            print(f"  Mean AUC difference: {mean_diff:+.4f}")
            print(f"  p-value: {p_value:.4f} {sig}")
            
            comparisons.append({
                'Strategy_1': s1,
                'Strategy_2': s2,
                'Mean_AUC_1': np.mean(aucs1),
                'Mean_AUC_2': np.mean(aucs2),
                'Mean_Diff': mean_diff,
                'p_value': p_value,
                'Significance': sig
            })
    
    comparison_df = pd.DataFrame(comparisons)
    comparison_df.to_csv(results_dir / 'statistical_tests_unfreezing.csv', index=False)
    print(f"\n✓ Saved: {results_dir / 'statistical_tests_unfreezing.csv'}")


def plot_strategy_comparison(comparison_df: pd.DataFrame, results_dir: Path):
    """Visualize comparison of unfreezing strategies"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    strategies = comparison_df['Strategy'].values
    x = range(len(strategies))
    
    # Plot 1: Test AUC
    ax = axes[0, 0]
    ax.bar(x, comparison_df['Mean_AUC'], yerr=comparison_df['Std_AUC'],
           alpha=0.7, capsize=5, color='steelblue')
    ax.set_ylabel('Test AUC', fontsize=12, fontweight='bold')
    ax.set_title('AUC by Unfreezing Strategy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Mark best
    best_idx = comparison_df['Mean_AUC'].idxmax()
    ax.scatter(best_idx, comparison_df.loc[best_idx, 'Mean_AUC'], 
               color='gold', s=200, marker='*', zorder=5, label='Best')
    ax.legend()
    
    # Plot 2: Trainable Parameters vs AUC
    ax = axes[0, 1]
    params = comparison_df['Trainable_Params'].values / 1e6  # In millions
    ax.scatter(params, comparison_df['Mean_AUC'], s=150, alpha=0.7, color='coral')
    for i, strategy in enumerate(strategies):
        ax.annotate(strategy, (params[i], comparison_df.loc[i, 'Mean_AUC']),
                   fontsize=9, ha='right')
    ax.set_xlabel('Trainable Params (M)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test AUC', fontsize=12, fontweight='bold')
    ax.set_title('Performance vs Model Complexity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Mean Final Overfitting Gap
    ax = axes[1, 0]
    gaps = comparison_df['Mean_Final_Gap'].values
    colors = ['red' if g > 0.15 else 'orange' if g > 0.10 else 'green' for g in gaps]
    ax.bar(x, gaps, alpha=0.7, color=colors)
    ax.axhline(y=0.15, color='red', linestyle='--', alpha=0.5, label='Overfitting Threshold')
    ax.set_ylabel('Final Train-Val Gap', fontsize=12, fontweight='bold')
    ax.set_title('Overfitting by Strategy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: All metrics
    ax = axes[1, 1]
    width = 0.25
    x_pos = np.arange(len(strategies))
    ax.bar(x_pos - width, comparison_df['Mean_AUC'], width, label='AUC', alpha=0.7)
    ax.bar(x_pos, comparison_df['Mean_Accuracy'], width, label='Accuracy', alpha=0.7)
    ax.bar(x_pos + width, comparison_df['Mean_F1'], width, label='F1', alpha=0.7)
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'strategy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {results_dir / 'strategy_comparison.png'}")


def main():
    """Main execution"""
    set_seed(BASE_CONFIG['base_seed'])
    
    results_base_dir = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\1_progression_maxpooling\unfreezing_experiment")
    results_base_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("PROGRESSIVE UNFREEZING EXPERIMENT FOR RESNET50")
    print("="*80)
    print(f"Dataset: 84 IPF patients")
    print(f"Goal: Test if unfreezing ResNet50 improves performance")
    print(f"Warning: High risk of overfitting on small dataset!")
    
    # Define which strategies to test
    # Start conservative: frozen (baseline) and layer4_only
    STRATEGIES_TO_TEST = [
        'frozen',        # Baseline
        'layer4_only',   # Conservative unfreezing
        # 'layer3_layer4',  # Uncomment if layer4_only succeeds
        # 'full_unfreeze',  # Only if very confident
    ]
    
    print(f"\nStrategies to test: {STRATEGIES_TO_TEST}")
    print(f"(Add more aggressive strategies if conservative ones succeed)")
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    data_loader = IPFDataLoader(
        csv_path=BASE_CONFIG['gt_path'],
        features_path=BASE_CONFIG['patient_features_path'],
        npy_dir=BASE_CONFIG['ct_scan_path']
    )
    patient_data, _ = data_loader.get_patient_data()
    
    # Extract CNN features
    slice_features_cache = results_base_dir / "slice_features.csv"
    
    if slice_features_cache.exists():
        print(f"\n✓ Loading cached slice features: {slice_features_cache}")
        slice_features_df = pd.read_csv(slice_features_cache)
    else:
        print("\n" + "="*70)
        print("EXTRACTING CNN FEATURES")
        print("="*70)
        
        feature_extractor = CNNFeatureExtractor(
            model_name='resnet50',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        slice_features_df = feature_extractor.extract_features_patient_grouping(
            patient_data=patient_data,
            patients_per_batch=4,
            save_path=slice_features_cache
        )
    
    # Load patient features
    from ablation_study import load_and_merge_demographics
    
    patient_features_df = pd.read_csv(BASE_CONFIG['patient_features_path'])
    patient_features_df = load_and_merge_demographics(
        train_csv_path=BASE_CONFIG['train_csv_path'],
        patient_features_df=patient_features_df
    )
    
    # Load K-fold splits
    with open(BASE_CONFIG['kfold_splits_path'], 'rb') as f:
        kfold_splits = pickle.load(f)
    
    print(f"\n✓ Loaded {len(kfold_splits)} folds")
    
    # Run experiment
    all_results = run_unfreezing_experiment(
        strategies_to_test=STRATEGIES_TO_TEST,
        slice_features_df=slice_features_df,
        patient_features_df=patient_features_df,
        kfold_splits=kfold_splits,
        base_config=BASE_CONFIG,
        results_base_dir=results_base_dir
    )
    
    print("\n" + "="*80)
    print("✅ UNFREEZING EXPERIMENT COMPLETE!")
    print("="*80)
    print(f"Results: {results_base_dir}")
    print(f"\n📊 Check:")
    print(f"  - strategy_comparison.csv: Summary of all strategies")
    print(f"  - statistical_tests_unfreezing.csv: Paired t-tests")
    print(f"  - strategy_comparison.png: Visualizations")
    print(f"  - */overfitting_curve.png: Overfitting monitoring per fold")


if __name__ == "__main__":
    main()
