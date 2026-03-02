from pathlib import Path
import sys
from typing import Dict, List, Tuple
import pandas as pd
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, 
    average_precision_score, confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib
matplotlib.use('Agg')  # Backend non-interattivo
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
sys.path.append(str(Path(__file__).parent.parent))
from utilities import (
    create_dataloaders,
    compute_class_weights
    )

HAND_FEATURE_COLS = [
    'ApproxVol_30_60',
    'Avg_NumTissuePixel_30_60',
    'Avg_Tissue_30_60',
    'Avg_Tissue_thickness_30_60',
    'Avg_TissueByTotal_30_60',
    'Avg_TissueByLung_30_60',
    'Mean_30_60',
    'Skew_30_60',
    'Kurtosis_30_60'
]

DEMO_FEATURE_COLS = ['Age', 'Sex', 'SmokingStatus']

class ProgressionPredictionModel(nn.Module):
    """
    Complete model for IPF progression prediction
    """
    
    def __init__(
        self,
        ctfm_feature_dim: int = 512,
        hand_feature_dim: int = 0,
        demo_feature_dim: int = 0,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.5,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        
        self.ctfm_feature_dim = ctfm_feature_dim
        self.hand_feature_dim = hand_feature_dim
        self.demo_feature_dim = demo_feature_dim
        
        print(f"\nProgression Prediction Model Configuration:")
        print(f"  CT-FM features (per slice): {ctfm_feature_dim}")
        print(f"  Hand-crafted features: {hand_feature_dim}")
        print(f"  Demographic features: {demo_feature_dim}")
    
        
        # === CNN REDUCTION LAYER (only if CNN features present) ===
        if ctfm_feature_dim > 0:
            self.ctf_reduction = nn.Sequential(
                nn.Linear(ctfm_feature_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            ctfm_reduced_dim = 64
            print(f"  CT-FM reduction: {ctfm_feature_dim} -> 64")
        else:
            self.ctf_reduction = None
            ctfm_reduced_dim = 0
            print(f"  CT-FM reduction: SKIPPED (no CT-FM features)")
        
        # Fusion input: reduced CT-FM (64 or 0) + hand-crafted + demographics
        fusion_input_dim = ctfm_reduced_dim + hand_feature_dim + demo_feature_dim
        
        print(f"  Fusion layer input: {fusion_input_dim}")
        
        # === CLASSIFICATION HEAD (SMALL) ===
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    
    def forward(
        self, 
        volume_features: torch.Tensor,
        patient_features: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            volume_features: (batch_size, feature_dim) - reduced CT-FM features
            patient_features: (batch_size, hand_dim + demo_dim) - optional
        
        Returns:
            logits: (B,1) 
        """
        # 1. Reduce CT-FM embeddings
        if self.ctfm_feature_dim > 0:
            ctfm_reduced = self.ctf_reduction(volume_features)  # (B, 64)
        else:
            ctfm_reduced = None

        # 2. Concatenate with clinical features
        if ctfm_reduced is not None:
            if patient_features is not None and patient_features.shape[1] > 0:
                combined = torch.cat([ctfm_reduced, patient_features], dim=-1)
            else:
                combined = ctfm_reduced
        else:
            if patient_features is not None and patient_features.shape[1] > 0:
                combined = patient_features
            else:
                raise ValueError("No features available: both CT-FM and patient features are empty!")

        # 3. Classify
        logits = self.classifier(combined)  # (B, 1)
        return logits
    
    def predict_proba(self, 
                     volume_features: torch.Tensor,
                     patient_features: torch.Tensor = None) -> Dict:
        """
        Return probabilities instead of logits
        For max pooling architecture: returns patient-level probability and pooling info
        """

        logits = self.forward(volume_features, patient_features)
        probs = torch.sigmoid(logits).squeeze(-1)  # (B,)
        return {
            'prob': probs
        }
    



class ModelTrainer:
    """
    Trainer for progression prediction model (CT-FM backbone).

    Batches now contain 'volume_features' (B, 512) instead of
    'cnn_features' (B, max_slices, 2048) + 'lengths'.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        class_weights: Tuple[float, float] = None,
        use_scheduler: bool = True,
        label_smoothing: float = 0.0
    ):
        self.model = model.to(device)
        self.device = device
        self.label_smoothing = label_smoothing
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.use_scheduler = use_scheduler

        if use_scheduler:
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
            'train_loss': [], 'train_auc': [], 'train_acc': [],
            'val_loss': [], 'val_auc': [], 'val_acc': []
        }

    def _apply_label_smoothing(self, labels, epsilon= 0.1):
        """Apply label smoothing to binary labels"""
        return labels * (1 - epsilon) + 0.5 * epsilon
    
    def _get_patient_features(self, batch):
        """Combine hand-crafted and demographic features if present"""
        hand_features = batch.get('hand_features')
        demo_features = batch.get('demo_features')
        
        
        if hand_features is not None and demo_features is not None:
            patient_features = torch.cat([hand_features, demo_features], dim=-1).to(self.device)
        elif hand_features is not None:
            patient_features = hand_features.to(self.device)
        elif demo_features is not None:
            patient_features = demo_features.to(self.device)
        else:
            patient_features = None  # No patient-level features available
        
        return patient_features

    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        all_losses = []
        all_preds = []
        all_labels = []
        
        for i, batch in enumerate(dataloader):

            # CT-FM: volume-features come as (B, max_slices, feature_dim)
            # For patient-level CT-FM embeddings, max_slices=1, so squeeze to (B, feature_dim)
            volume_features = batch['volume_features'].to(self.device)  # (B, 1, 512)
            if volume_features.shape[1] == 1:
                volume_features = volume_features.squeeze(1)  # (B, 512)
            
            patient_features = self._get_patient_features(batch)  # (B, hand+demo) or None            
            
            if i==0:
                print(f"  volume_features: {volume_features.shape}")
                print(f"  patient_features: {patient_features.shape if patient_features is not None else 'None'}")
            
            
            labels = batch['labels'].to(self.device).float()  # (B,)
            
            # Apply label smoothing only if configured
            if self.label_smoothing > 0:
                labels = self._apply_label_smoothing(labels, self.label_smoothing)

            logits = self.model(volume_features, patient_features).squeeze(-1)  # (B,)
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
        metrics = {
            'loss': np.mean(all_losses),
            'auc': roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5,
            'acc': accuracy_score(all_labels, np.array(all_preds) > 0.5)
        }
        
        return metrics
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate on validation/test set"""
        self.model.eval()
        
        all_losses = []
        all_preds = []
        all_labels = []
        all_patient_ids = []
        
        with torch.no_grad():
            for batch in dataloader:
                volume_features = batch['volume_features'].to(self.device)
                # For patient-level CT-FM embeddings, squeeze slice dimension if needed
                if volume_features.shape[1] == 1:
                    volume_features = volume_features.squeeze(1)  # (B, 1, 512) -> (B, 512)
                
                patient_features = self._get_patient_features(batch)  # (B, hand+demo) or None
                
                labels = batch['labels'].to(self.device).float()
                patient_ids = batch['patient_ids']  # Collect patient IDs
            
                # Forward pass
                logits = self.model(volume_features, patient_features).squeeze(-1)
                loss = self.criterion(logits, labels)
                
                # Record metrics
                all_losses.append(loss.item())
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())
                all_patient_ids.extend(patient_ids)
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        metrics = {
            'loss': np.mean(all_losses),
            'auc': roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5,
            'acc': accuracy_score(all_labels, all_preds > 0.5),
            'f1': f1_score(all_labels, all_preds > 0.5, zero_division=0),
            'predictions': all_preds,
            'labels': all_labels,
            'patient_ids': all_patient_ids
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
        """Train the model with early stopping"""
        
        best_val_auc = 0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            
            if self.use_scheduler:
                self.scheduler.step(val_metrics['auc'])

            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_auc'].append(train_metrics['auc'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_acc'].append(val_metrics['acc'])
            
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}, Acc: {train_metrics['acc']:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}, Acc: {val_metrics['acc']:.4f}")
            
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
    
    def plot_training_history(self, save_path: str = None):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # AUC
        axes[1].plot(self.history['train_auc'], label='Train')
        axes[1].plot(self.history['val_auc'], label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].set_title('ROC AUC')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Target')
        
        # Accuracy
        axes[2].plot(self.history['train_acc'], label='Train')
        axes[2].plot(self.history['val_acc'], label='Val')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Accuracy')
        axes[2].set_title('Accuracy')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        



def plot_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    save_path: str = None
):
    """
    Plot comprehensive evaluation metrics with custom threshold
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    
    axes[0, 0].plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    
    # Mark the threshold point
    y_pred_binary = (y_pred >= threshold).astype(int)
    idx = np.argmin(np.abs(roc_curve(y_true, y_pred)[2] - threshold))
    axes[0, 0].plot(fpr[idx], tpr[idx], 'ro', markersize=10, 
                    label=f'Threshold = {threshold:.3f}')
    
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    
    axes[0, 1].plot(recall, precision, label=f'AP = {ap:.3f}')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Confusion Matrix
    y_pred_binary = (y_pred >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_title(f'Confusion Matrix (Threshold = {threshold:.3f})')
    
    # 4. Prediction distribution
    axes[1, 1].hist(y_pred[y_true == 0], bins=20, alpha=0.5, label='No Progression', color='blue')
    axes[1, 1].hist(y_pred[y_true == 1], bins=20, alpha=0.5, label='Progression', color='red')
    axes[1, 1].axvline(x=threshold, color='k', linestyle='--', alpha=0.7, linewidth=2,
                       label=f'Threshold = {threshold:.3f}')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Prediction Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Print classification report
    print("\n" + "="*60)
    print(f"CLASSIFICATION REPORT (Threshold = {threshold:.3f})")
    print("="*60)
    print(classification_report(y_true, y_pred_binary, 
                                target_names=['No Progression', 'Progression']))







def train_kfold_cv(features_df, splits, config, results_dir):
    """
    Train with K-fold cross-validation
    """
    print("\n" + "="*70)
    print("K-FOLD CROSS-VALIDATION")
    print("="*70)
    
    fold_results = []
    
    # Determine which folds to train
    
    fold_keys = sorted(splits.keys())
    print(f"\nTraining all {len(fold_keys)} folds")
    
    # Train each fold
    for fold_key in fold_keys:
        fold_idx = int(fold_key.split('_')[1])
        fold_data = splits[fold_key]
        
        result = train_single_fold(
            features_df=features_df,
            fold_data=fold_data,
            fold_idx=fold_idx,
            config=config,
            results_dir=results_dir
        )
        
        fold_results.append(result)
    
    # Summary statistics
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
    results_df.to_csv(results_dir / "kfold_summary.csv", index=False)
    
    # Performance interpretation
    mean_test_auc = results_df['test_auc'].mean()
    
    
    return results_df


def train_single_fold(features_df: pd.DataFrame,
                      fold_data: dict,
                      fold_idx: int,
                      config: dict,
                      results_dir: Path,
                      resume_from_checkpoint: bool = True,
                      hand_feature_cols: list = None,
                      demo_feature_cols: list = None,
                      encoding_info: dict = None):
    """
    Train model on a single fold with preprocessed demographics
    """

    if hand_feature_cols is None:
        hand_feature_cols = HAND_FEATURE_COLS

    if demo_feature_cols is None:
        demo_feature_cols = DEMO_FEATURE_COLS
    
    if encoding_info is None:
        encoding_info = {}

    fold_dir = results_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = fold_dir / "best_model.pth"

    # Check if checkpoint exists and we should resume
    if resume_from_checkpoint and checkpoint_path.exists():
        print("\n" + "="*70)
        print(f"CHECKPOINT FOUND FOR FOLD {fold_idx}")
        print("="*70)
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path,weights_only=False)

        # Check if this fold is already complete (has all test metrics)
        # A complete fold must have both default and optimal threshold test results
        is_complete = (
            'test_metrics_default' in checkpoint and 
            'test_metrics_optimal' in checkpoint and
            'val_auc' in checkpoint and
            'optimal_threshold' in checkpoint
        )
        
        if is_complete:
            print("\n✓ Fold already completed! Loading saved results...")
            print(f"  Val AUC: {checkpoint.get('val_auc', 'N/A'):.4f}")
            print(f"  Optimal Threshold: {checkpoint.get('optimal_threshold', 'N/A'):.4f}")
            print(f"  Test AUC (Default): {checkpoint['test_metrics_default'].get('auc', 'N/A'):.4f}")
            print(f"  Test AUC (Optimal): {checkpoint['test_metrics_optimal'].get('auc', 'N/A'):.4f}")
            
            # Return saved results
            return {
                'fold_idx': fold_idx,
                'val_auc': checkpoint.get('val_auc'),
                'val_metrics': checkpoint.get('val_metrics', {}),
                'test_metrics_default': checkpoint.get('test_metrics_default', {}),
                'test_metrics_optimal': checkpoint.get('test_metrics_optimal', {}),
                'optimal_threshold': checkpoint.get('optimal_threshold'),
                'threshold_analysis': checkpoint.get('threshold_analysis', {}),
                'loaded_from_checkpoint': True
            }
        else:
            print("\n⚠ Checkpoint found but fold is incomplete. Starting from scratch...")
            print("   (This may happen if previous run was interrupted during evaluation)")
    


    print("\n" + "="*70)
    print(f"TRAINING FOLD {fold_idx}")
    print("="*70)
    
    # Create dataloaders
    train_ids = fold_data['train']
    val_ids = fold_data['val']
    test_ids = fold_data['test']
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_ids)} patients")
    print(f"  Val: {len(val_ids)} patients")
    print(f"  Test: {len(test_ids)} patients")

    # Identify available features
    available_hand_cols = [c for c in hand_feature_cols if c in features_df.columns]
    
    # For demographics, DON'T filter by column names since they've been preprocessed
    # (Age -> Age_normalized, Sex -> Sex_encoded, SmokingStatus -> Smoking_0/1/2)
    # The dataset will handle mapping original names to preprocessed columns
    
    print(f"\nFeature availability:")
    print(f"  Hand-crafted: {len(available_hand_cols)}/{len(hand_feature_cols)}")
    print(f"  Demographics: {len(demo_feature_cols)}/{len(demo_feature_cols)}")
    
    # Verify demographics preprocessing
    if demo_feature_cols:
        print(f"\n  Demographics preprocessing verification:")
        if 'Age' in demo_feature_cols:
            has_age_norm = 'Age_normalized' in features_df.columns
            print(f"    Age → Age_normalized: {'✓' if has_age_norm else '✗ MISSING'}")
        if 'Sex' in demo_feature_cols:
            has_sex_enc = 'Sex_encoded' in features_df.columns
            print(f"    Sex → Sex_encoded: {'✓' if has_sex_enc else '✗ MISSING'}")
        if 'SmokingStatus' in demo_feature_cols:
            smoking_cols = encoding_info.get('smoking_columns', [])
            has_smoking = all(col in features_df.columns for col in smoking_cols)
            print(f"    SmokingStatus → {smoking_cols}: {'✓' if has_smoking else '✗ MISSING'}")
    
    # Compute class weights
    class_weights = compute_class_weights(features_df, train_ids)
    
    # Create dataloaders
    # Pass ORIGINAL demographic column names - the dataset handles preprocessing mapping
    train_loader, val_loader, test_loader = create_dataloaders(
        features_df,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        batch_size=config['batch_size'],
        num_workers=4,
        hand_feature_cols=available_hand_cols,
        demo_feature_cols=demo_feature_cols,  # Pass original names, not filtered
        encoding_info=encoding_info
    )

    # Get actual dimensions from first batch
    sample_batch = next(iter(train_loader))
    # volume_features shape: (B, max_slices, feature_dim) - for CT-FM it's (B, 1, 512)
    actual_ctfm_dim = sample_batch['volume_features'].shape[2]   # Get feature_dim (512)
    actual_hand_dim = sample_batch['hand_features'].shape[1] if sample_batch.get('hand_features') is not None else 0
    actual_demo_dim = sample_batch['demo_features'].shape[1] if sample_batch.get('demo_features') is not None else 0

    print(f"\nFeature dimensions:")
    print(f"  CT-FM volume features: {actual_ctfm_dim}")
    print(f"  Hand-crafted:          {actual_hand_dim}")
    print(f"  Demographic:           {actual_demo_dim}")
    # Create model
    print(f"\nInitializing model:")
    print(f"  Hidden dimensions: {config['hidden_dims']} (not used - fixed architecture)")
    print(f"  Dropout: {config['dropout']}")
    print(f"  Pooling: {config.get('pooling_type', 'max')}")
    print(f"  Architecture: Simplified (CNN reduction → concat → classifier)")
    print(f"  Label smoothing: {config.get('label_smoothing', 0.0)}")

    model = ProgressionPredictionModel(
        ctfm_feature_dim=actual_ctfm_dim,
        hand_feature_dim=actual_hand_dim,  # Use ACTUAL dimensions from batch
        demo_feature_dim=actual_demo_dim,   # Use ACTUAL dimensions from batch
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout'],
        use_batch_norm=config['use_batch_norm']
    )
    
    # Log trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        class_weights=class_weights,
        use_scheduler=config['use_scheduler'],
        label_smoothing=config.get('label_smoothing', 0.0)
    )
    
    # Train
    print(f"\n{'='*70}")
    print("TRAINING")
    print("="*70)
    
    best_val_auc = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        early_stopping_patience=config['early_stopping_patience'],
        verbose=True
    )
    
    # Plot training history
    fold_dir = results_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    trainer.plot_training_history(save_path=str(fold_dir / "training_history.png"))
    # ========== VALIDATION SET ANALYSIS ==========
    print(f"\n{'='*70}")
    print("VALIDATION SET THRESHOLD ANALYSIS")
    print("="*70)
    
    val_results = trainer.evaluate(val_loader)
    
    # Analyze validation ROC and find optimal thresholds
    threshold_analysis = plot_validation_roc_with_thresholds(
        y_true=val_results['labels'],
        y_pred=val_results['predictions'],
        save_path=str(fold_dir / "validation_roc_threshold_analysis.png")
    )
    
    # Print threshold analysis
    print("\nThreshold Analysis on Validation Set:")
    print("-" * 70)
    for name, metrics in threshold_analysis.items():
        print(f"\n{name}:")
        print(f"  Threshold: {metrics['threshold']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
    
    # Choose optimal threshold (using Youden's J statistic)
    optimal_threshold = threshold_analysis['Youden']['threshold']
    print(f"\n{'='*70}")
    print(f"Selected Optimal Threshold: {optimal_threshold:.4f} (Youden's J)")
    print("="*70)
    
    # Save validation metrics
    val_metrics = {
        'auc': val_results['auc'],
        'threshold_analysis': threshold_analysis
    }
    
    # ========== TEST SET EVALUATION ==========
    print(f"\n{'='*70}")
    print("TEST SET EVALUATION")
    print("="*70)
    
    test_results = trainer.evaluate(test_loader)
    
    # Evaluate with default threshold (0.5)
    print("\n1. Default Threshold (0.5):")
    print("-" * 70)
    test_metrics_default = evaluate_with_threshold(
        y_true=test_results['labels'],
        y_pred=test_results['predictions'],
        threshold=0.5
    )
    
    for metric, value in test_metrics_default.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    # Evaluate with optimal threshold from validation
    print(f"\n2. Optimal Threshold ({optimal_threshold:.4f}):")
    print("-" * 70)
    test_metrics_optimal = evaluate_with_threshold(
        y_true=test_results['labels'],
        y_pred=test_results['predictions'],
        threshold=optimal_threshold
    )
    
    for metric, value in test_metrics_optimal.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    # Plot evaluation metrics with default threshold
    plot_evaluation_metrics(
        y_true=test_results['labels'],
        y_pred=test_results['predictions'],
        threshold=0.5,
        save_path=str(fold_dir / "test_evaluation_default_threshold.png")
    )
    
    # Plot evaluation metrics with optimal threshold
    plot_evaluation_metrics(
        y_true=test_results['labels'],
        y_pred=test_results['predictions'],
        threshold=optimal_threshold,
        save_path=str(fold_dir / "test_evaluation_optimal_threshold.png")
    )
    
    # Save comprehensive results
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'fold_idx': fold_idx,
        'val_auc': best_val_auc,
        'val_metrics': val_metrics,
        'optimal_threshold': optimal_threshold,
        'threshold_analysis': threshold_analysis,
        'test_metrics_default': test_metrics_default,
        'test_metrics_optimal': test_metrics_optimal,
    }, checkpoint_path)
    
    # Save predictions with both thresholds
    predictions_df = pd.DataFrame({
        'patient_id': test_results['patient_ids'],  # Use actual patient IDs from test results
        'true_label': test_results['labels'],
        'predicted_prob': test_results['predictions'],
        'predicted_label_default': (test_results['predictions'] >= 0.5).astype(int),
        'predicted_label_optimal': (test_results['predictions'] >= optimal_threshold).astype(int)
    })
    predictions_df.to_csv(fold_dir / "test_predictions.csv", index=False)
    
    # Save metrics summary to CSV
    metrics_summary = pd.DataFrame({
        'metric': ['val_auc', 'optimal_threshold'] + 
                  [f'test_{k}_default' for k in test_metrics_default.keys()] +
                  [f'test_{k}_optimal' for k in test_metrics_optimal.keys()],
        'value': [best_val_auc, optimal_threshold] + 
                 list(test_metrics_default.values()) +
                 list(test_metrics_optimal.values())
    })
    metrics_summary.to_csv(fold_dir / "metrics_summary.csv", index=False)
    
    print(f"\n{'='*70}")
    print("FOLD TRAINING COMPLETE")
    print("="*70)
    print(f"Results saved to: {fold_dir}")
    
    return {
        'fold_idx': fold_idx,
        'val_auc': best_val_auc,
        'val_metrics': val_metrics,
        'test_metrics_default': test_metrics_default,
        'test_metrics_optimal': test_metrics_optimal,
        'optimal_threshold': optimal_threshold,
        'threshold_analysis': threshold_analysis,
        'loaded_from_checkpoint': False
    }


def compute_specificity(y_true, y_pred_binary):
    """Compute specificity (True Negative Rate)"""
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return specificity



def find_optimal_threshold(y_true, y_pred, metric='youden'):
    """
    Find optimal threshold using different strategies
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        metric: 'youden', 'f1', or 'closest_to_topleft'
    
    Returns:
        optimal_threshold, metrics_at_threshold
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    
    if metric == 'youden':
        # Youden's J statistic = Sensitivity + Specificity - 1
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
    elif metric == 'f1':
        # Find threshold that maximizes F1 score
        f1_scores = []
        for threshold in thresholds:
            y_pred_binary = (y_pred >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred_binary, zero_division=0)
            f1_scores.append(f1)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
    elif metric == 'closest_to_topleft':
        # Find point closest to top-left corner (0, 1)
        distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
        optimal_idx = np.argmin(distances)
        optimal_threshold = thresholds[optimal_idx]
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Compute metrics at optimal threshold
    y_pred_binary = (y_pred >= optimal_threshold).astype(int)
    
    metrics = {
        'threshold': optimal_threshold,
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'f1': f1_score(y_true, y_pred_binary, zero_division=0),
        'specificity': compute_specificity(y_true, y_pred_binary),
        'sensitivity': recall_score(y_true, y_pred_binary, zero_division=0),  # same as recall
        'fpr': fpr[optimal_idx],
        'tpr': tpr[optimal_idx]
    }
    
    return optimal_threshold, metrics


def plot_validation_roc_with_thresholds(y_true, y_pred, save_path=None):
    """
    Plot ROC curve with multiple threshold strategies marked
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')
    
    # Find and mark optimal thresholds
    threshold_strategies = {
        'Youden': 'youden',
        'Max F1': 'f1',
        'Closest to Top-Left': 'closest_to_topleft'
    }
    
    colors = {'Youden': 'red', 'Max F1': 'green', 'Closest to Top-Left': 'orange'}
    markers = {'Youden': 'o', 'Max F1': 's', 'Closest to Top-Left': '^'}
    
    threshold_results = {}
    
    for name, strategy in threshold_strategies.items():
        optimal_threshold, metrics = find_optimal_threshold(y_true, y_pred, strategy)
        threshold_results[name] = {'threshold': optimal_threshold, **metrics}
        
        # Mark on plot
        ax.plot(metrics['fpr'], metrics['tpr'], 
                markers[name], 
                color=colors[name], 
                markersize=12, 
                label=f"{name}: {optimal_threshold:.3f}",
                markeredgecolor='white',
                markeredgewidth=1.5)
    
    # Mark default threshold (0.5)
    default_pred = (y_pred >= 0.5).astype(int)
    default_metrics = {
        'accuracy': accuracy_score(y_true, default_pred),
        'precision': precision_score(y_true, default_pred, zero_division=0),
        'recall': recall_score(y_true, default_pred, zero_division=0),
        'f1': f1_score(y_true, default_pred, zero_division=0),
        'specificity': compute_specificity(y_true, default_pred)
    }
    
    # Find FPR and TPR for threshold 0.5
    idx_05 = np.argmin(np.abs(thresholds - 0.5))
    ax.plot(fpr[idx_05], tpr[idx_05], 'D', 
            color='purple', 
            markersize=12, 
            label=f"Default (0.5)",
            markeredgecolor='white',
            markeredgewidth=1.5)
    
    threshold_results['Default (0.5)'] = {'threshold': 0.5, **default_metrics, 
                                           'fpr': fpr[idx_05], 'tpr': tpr[idx_05]}
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Validation ROC Curve with Optimal Thresholds', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return threshold_results


def evaluate_with_threshold(y_true, y_pred, threshold=0.5):
    """
    Evaluate predictions with a specific threshold
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    metrics = {
        'threshold': threshold,
        'auc': roc_auc_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'f1': f1_score(y_true, y_pred_binary, zero_division=0),
        'specificity': compute_specificity(y_true, y_pred_binary)
    }
    
    return metrics



def aggregate_fold_results(fold_results: list, save_path: Path):
    """
    Aggregate results across all folds
    """
    print("\n" + "="*70)
    print("AGGREGATE RESULTS ACROSS ALL FOLDS")
    print("="*70)
    
    # Collect metrics
    val_aucs = [r['val_auc'] for r in fold_results]
    
    # Default threshold metrics
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
    
    optimal_thresholds = [r['optimal_threshold'] for r in fold_results]
    
    # Create summary DataFrame
    summary_data = {
        'Metric': [
            'Validation AUC',
            'Optimal Threshold',
            '',
            'Test AUC (Default)',
            'Test Accuracy (Default)',
            'Test Precision (Default)',
            'Test Recall (Default)',
            'Test F1 (Default)',
            'Test Specificity (Default)',
            '',
            'Test AUC (Optimal)',
            'Test Accuracy (Optimal)',
            'Test Precision (Optimal)',
            'Test Recall (Optimal)',
            'Test F1 (Optimal)',
            'Test Specificity (Optimal)'
        ],
        'Mean': [
            np.mean(val_aucs),
            np.mean(optimal_thresholds),
            np.nan,
            np.mean(test_auc_default),
            np.mean(test_acc_default),
            np.mean(test_prec_default),
            np.mean(test_rec_default),
            np.mean(test_f1_default),
            np.mean(test_spec_default),
            np.nan,
            np.mean(test_auc_optimal),
            np.mean(test_acc_optimal),
            np.mean(test_prec_optimal),
            np.mean(test_rec_optimal),
            np.mean(test_f1_optimal),
            np.mean(test_spec_optimal)
        ],
        'Std': [
            np.std(val_aucs),
            np.std(optimal_thresholds),
            np.nan,
            np.std(test_auc_default),
            np.std(test_acc_default),
            np.std(test_prec_default),
            np.std(test_rec_default),
            np.std(test_f1_default),
            np.std(test_spec_default),
            np.nan,
            np.std(test_auc_optimal),
            np.std(test_acc_optimal),
            np.std(test_prec_optimal),
            np.std(test_rec_optimal),
            np.std(test_f1_optimal),
            np.std(test_spec_optimal)
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Print summary
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_df.to_csv(save_path / "aggregate_metrics_summary.csv", index=False)
    
    # Create detailed fold-by-fold results
    detailed_results = []
    for r in fold_results:
        fold_data = {
            'fold': r['fold_idx'],
            'val_auc': r['val_auc'],
            'optimal_threshold': r['optimal_threshold'],
            'test_auc_default': r['test_metrics_default']['auc'],
            'test_accuracy_default': r['test_metrics_default']['accuracy'],
            'test_precision_default': r['test_metrics_default']['precision'],
            'test_recall_default': r['test_metrics_default']['recall'],
            'test_f1_default': r['test_metrics_default']['f1'],
            'test_specificity_default': r['test_metrics_default']['specificity'],
            'test_auc_optimal': r['test_metrics_optimal']['auc'],
            'test_accuracy_optimal': r['test_metrics_optimal']['accuracy'],
            'test_precision_optimal': r['test_metrics_optimal']['precision'],
            'test_recall_optimal': r['test_metrics_optimal']['recall'],
            'test_f1_optimal': r['test_metrics_optimal']['f1'],
            'test_specificity_optimal': r['test_metrics_optimal']['specificity'],
        }
        detailed_results.append(fold_data)
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(save_path / "detailed_fold_results.csv", index=False)
    
    print(f"\nResults saved to:")
    print(f"  - {save_path / 'aggregate_metrics_summary.csv'}")
    print(f"  - {save_path / 'detailed_fold_results.csv'}")
    
    # ADD THESE LINES BEFORE RETURN:
    
    # Create aggregate visualizations
    print("\n" + "="*70)
    print("CREATING AGGREGATE VISUALIZATIONS")
    print("="*70)
    
    # Plot aggregate test ROC curves
    plot_aggregate_test_roc(
        results_dir=save_path,
        fold_results=fold_results,
        save_path=save_path / "aggregate_test_roc_curves.png"
    )
    
    # Plot confusion matrices for all folds
    plot_aggregate_confusion_matrices(
        results_dir=save_path,
        fold_results=fold_results,
        save_path=save_path / "aggregate_confusion_matrices.png"
    )
    
    # Plot metrics comparison
    plot_aggregate_metrics_comparison(
        fold_results=fold_results,
        save_path=save_path / "aggregate_metrics_comparison.png"
    )
    
    print("\n" + "="*70)
    print("AGGREGATE VISUALIZATIONS COMPLETE")
    print("="*70)
    print(f"\nGenerated plots:")
    print(f"  - {save_path / 'aggregate_test_roc_curves.png'}")
    print(f"  - {save_path / 'aggregate_confusion_matrices.png'}")
    print(f"  - {save_path / 'aggregate_metrics_comparison.png'}")
    
    return summary_df, detailed_df



def plot_aggregate_test_roc(results_dir: Path, fold_results: list, save_path: Path):
    """
    Plot ROC curves for all folds on the same plot (test set)
    Loads predictions from saved CSV files
    """
    fig, ax = plt.subplots(figsize=(12, 9))
    
    all_tprs = []
    all_aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(fold_results)))
    
    # Plot each fold
    for i, result in enumerate(fold_results):
        fold_idx = result['fold_idx']
        fold_dir = results_dir / f"fold_{fold_idx}"
        predictions_file = fold_dir / "test_predictions.csv"
        
        if predictions_file.exists():
            # Load predictions
            pred_df = pd.read_csv(predictions_file)
            y_true = pred_df['true_label'].values
            y_pred = pred_df['predicted_prob'].values
            
            # Compute ROC
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            all_aucs.append(auc)
            
            # Interpolate TPR at mean FPR
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            all_tprs.append(interp_tpr)
            
            # Plot individual fold
            ax.plot(fpr, tpr, color=colors[i], alpha=0.6, linewidth=1.5,
                   label=f'Fold {fold_idx} (AUC = {auc:.3f})')
    
    # Plot random classifier
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2, label='Random')
    
    # Plot mean ROC
    if all_tprs:
        mean_tpr = np.mean(all_tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(all_aucs)
        std_auc = np.std(all_aucs)
        
        ax.plot(mean_fpr, mean_tpr, 'b-', linewidth=3,
               label=f'Mean (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
        
        # Plot std deviation
        std_tpr = np.std(all_tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='blue', alpha=0.2,
                        label='± 1 std. dev.')
    
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('Test Set ROC Curves - All Folds', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nAggregate test ROC plot saved to: {save_path}")


def plot_aggregate_confusion_matrices(results_dir: Path, fold_results: list, save_path: Path):
    """
    Plot confusion matrices for all folds in a grid
    Shows both default threshold (0.5) and optimal threshold
    Includes aggregate (summed) confusion matrices across folds
    """
    n_folds = len(fold_results)

    # Create figure with 2 rows and n_folds + 1 columns (last column = sum)
    fig, axes = plt.subplots(2, n_folds + 1, figsize=(4 * (n_folds + 1), 8))

    if n_folds == 1:
        axes = axes.reshape(2, 2)

    # Initialize aggregate confusion matrices
    cm_default_sum = np.zeros((2, 2), dtype=int)
    cm_optimal_sum = np.zeros((2, 2), dtype=int)

    for i, result in enumerate(fold_results):
        fold_idx = result['fold_idx']
        fold_dir = results_dir / f"fold_{fold_idx}"
        predictions_file = fold_dir / "test_predictions.csv"

        if predictions_file.exists():
            pred_df = pd.read_csv(predictions_file)
            y_true = pred_df['true_label'].values
            y_pred_prob = pred_df['predicted_prob'].values
            optimal_threshold = result['optimal_threshold']

            # ---- Default threshold (0.5)
            y_pred_default = (y_pred_prob >= 0.5).astype(int)
            cm_default = confusion_matrix(y_true, y_pred_default)
            cm_default_sum += cm_default

            sns.heatmap(
                cm_default, annot=True, fmt='d', cmap='Blues',
                ax=axes[0, i], cbar=False,
                xticklabels=['No Prog', 'Prog'],
                yticklabels=['No Prog', 'Prog']
            )
            axes[0, i].set_title(f'Fold {fold_idx}\nThreshold = 0.5', fontsize=11, fontweight='bold')
            if i == 0:
                axes[0, i].set_ylabel('True Label', fontsize=11)
            axes[0, i].set_xlabel('Predicted', fontsize=10)

            # ---- Optimal threshold
            y_pred_optimal = (y_pred_prob >= optimal_threshold).astype(int)
            cm_optimal = confusion_matrix(y_true, y_pred_optimal)
            cm_optimal_sum += cm_optimal

            sns.heatmap(
                cm_optimal, annot=True, fmt='d', cmap='Greens',
                ax=axes[1, i], cbar=False,
                xticklabels=['No Prog', 'Prog'],
                yticklabels=['No Prog', 'Prog']
            )
            axes[1, i].set_title(
                f'Fold {fold_idx}\nThreshold = {optimal_threshold:.3f}',
                fontsize=11, fontweight='bold'
            )
            if i == 0:
                axes[1, i].set_ylabel('True Label', fontsize=11)
            axes[1, i].set_xlabel('Predicted', fontsize=10)

    # ---- Plot aggregate (summed) matrices in last column
    sns.heatmap(
        cm_default_sum, annot=True, fmt='d', cmap='Blues',
        ax=axes[0, -1], cbar=False,
        xticklabels=['No Prog', 'Prog'],
        yticklabels=['No Prog', 'Prog']
    )
    axes[0, -1].set_title('Sum (All Folds)\nThreshold = 0.5', fontsize=11, fontweight='bold')
    axes[0, -1].set_xlabel('Predicted', fontsize=10)

    sns.heatmap(
        cm_optimal_sum, annot=True, fmt='d', cmap='Greens',
        ax=axes[1, -1], cbar=False,
        xticklabels=['No Prog', 'Prog'],
        yticklabels=['No Prog', 'Prog']
    )
    axes[1, -1].set_title('Sum (All Folds)\nOptimal Thresholds', fontsize=11, fontweight='bold')
    axes[1, -1].set_xlabel('Predicted', fontsize=10)

    plt.suptitle(
        'Confusion Matrices - All Folds\nTop: Default (0.5) | Bottom: Optimal | Last Column: Sum',
        fontsize=14, fontweight='bold', y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Aggregate confusion matrices saved to: {save_path}")



def plot_aggregate_metrics_comparison(fold_results: list, save_path: Path):
    """
    Plot comparison of metrics across folds (default vs optimal threshold)
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    metrics_to_plot = ['auc', 'accuracy', 'precision', 'recall', 'f1', 'specificity']
    titles = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    
    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        folds = [r['fold_idx'] for r in fold_results]
        default_values = [r['test_metrics_default'][metric] for r in fold_results]
        optimal_values = [r['test_metrics_optimal'][metric] for r in fold_results]
        
        x = np.arange(len(folds))
        width = 0.35
        
        ax.bar(x - width/2, default_values, width, label='Default (0.5)', 
               alpha=0.8, color='steelblue')
        ax.bar(x + width/2, optimal_values, width, label='Optimal', 
               alpha=0.8, color='seagreen')
        
        # Add mean lines
        ax.axhline(y=np.mean(default_values), color='steelblue', 
                   linestyle='--', alpha=0.5, linewidth=2)
        ax.axhline(y=np.mean(optimal_values), color='seagreen', 
                   linestyle='--', alpha=0.5, linewidth=2)
        
        ax.set_xlabel('Fold', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{title}\n(Mean: Default={np.mean(default_values):.3f}, '
                    f'Optimal={np.mean(optimal_values):.3f})', 
                    fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{f}' for f in folds])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)
    
    plt.suptitle('Test Metrics Comparison Across Folds\nDefault vs Optimal Threshold', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics comparison plot saved to: {save_path}")