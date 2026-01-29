from pathlib import Path
import sys
from typing import Dict, List, Tuple
import pandas as pd
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
sys.path.append(str(Path(__file__).parent.parent))
from utilities import (
    create_dataloaders,
    compute_class_weights
    )

class SliceAggregator(nn.Module):
    """Base class for slice aggregation strategies"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
    
    def forward(self, slice_features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slice_features: (batch_size, max_slices, feature_dim)
            lengths: (batch_size,) - actual number of slices per patient
        Returns:
            aggregated_features: (batch_size, output_dim)
        """
        raise NotImplementedError
    
class MaxPoolAggregator(SliceAggregator):
    """Max pooling over slices"""
    
    def forward(self, slice_features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, max_slices, _ = slice_features.shape
        mask = torch.arange(max_slices, device=slice_features.device)[None, :] < lengths[:, None]
        mask = mask.unsqueeze(-1)  # (batch_size, max_slices, 1)
        
        # Set padding to very negative value
        masked_features = slice_features.clone()
        masked_features[~mask.expand_as(slice_features)] = -1e9
        
        # Max pooling
        max_features, _ = masked_features.max(dim=1)  # (batch_size, feature_dim)
        
        return max_features

class ProgressionPredictionModel(nn.Module):
    """
    Complete model for IPF progression prediction
    """
    
    def __init__(
        self,
        feature_dim: int = 1280,
        aggregation: str = 'max',  # 'mean', 'max', 'attention', 'combined'
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.5,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.aggregation_type = aggregation
        
        # Aggregator
        if aggregation == 'max':
            self.aggregator = MaxPoolAggregator(feature_dim)
            agg_output_dim = feature_dim
        
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        # Classification head
        layers = []
        input_dim = agg_output_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, slice_features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slice_features: (batch_size, max_slices, feature_dim)
            lengths: (batch_size,) - actual number of slices per patient
        Returns:
            logits: (batch_size, 1)
        """
        # Aggregate slices
        aggregated = self.aggregator(slice_features, lengths)
        
        # Classify
        logits = self.classifier(aggregated)
        
        return logits
    
    def predict_proba(self, slice_features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return probabilities instead of logits"""
        logits = self.forward(slice_features, lengths)
        probs = torch.sigmoid(logits)
        return probs
    
class ModelTrainer:
    """
    Trainer for progression prediction model
    """
    
    def __init__(
        self,
        model: nn.Module,
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
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        all_losses = []
        all_preds = []
        all_labels = []
        
        for batch in dataloader:
            slice_features = batch['features'].to(self.device)  # (B, max_slices, feat_dim)
            lengths = batch['lengths'].to(self.device)  # (B,)
            labels = batch['labels'].to(self.device).float()  # (B,)
            
            # Forward pass
            logits = self.model(slice_features, lengths).squeeze(-1)  # (B,)
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
        
        with torch.no_grad():
            for batch in dataloader:
                slice_features = batch['features'].to(self.device)
                lengths = batch['lengths'].to(self.device)
                labels = batch['labels'].to(self.device).float()
                
                # Forward pass
                logits = self.model(slice_features, lengths).squeeze(-1)
                loss = self.criterion(logits, labels)
                
                # Record metrics
                all_losses.append(loss.item())
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        metrics = {
            'loss': np.mean(all_losses),
            'auc': roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5,
            'acc': accuracy_score(all_labels, all_preds > 0.5),
            'f1': f1_score(all_labels, all_preds > 0.5, zero_division=0),
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
        """Train the model with early stopping"""
        
        best_val_auc = 0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            
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
        plt.show() 



def plot_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = None
):
    """
    Plot comprehensive evaluation metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    
    axes[0, 0].plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
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
    y_pred_binary = (y_pred > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_title('Confusion Matrix')
    
    # 4. Prediction distribution
    axes[1, 1].hist(y_pred[y_true == 0], bins=20, alpha=0.5, label='No Progression', color='blue')
    axes[1, 1].hist(y_pred[y_true == 1], bins=20, alpha=0.5, label='Progression', color='red')
    axes[1, 1].axvline(x=0.5, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Prediction Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred_binary, target_names=['No Progression', 'Progression']))








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


def train_single_fold(features_df: pd.DataFrame,fold_data: dict,fold_idx: int,config: dict,results_dir: Path):
    """
    Train model on a single fold
    """
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
    
    # Compute class weights
    class_weights = compute_class_weights(features_df, train_ids)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        features_df,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        batch_size=config['batch_size'],
        num_workers=4
    )
    
    # Create model
    print(f"\nInitializing model:")
    print(f"  Aggregation: {config['aggregation']}")
    print(f"  Hidden dimensions: {config['hidden_dims']}")
    print(f"  Dropout: {config['dropout']}")
    
    model = ProgressionPredictionModel(
        feature_dim=config['feature_dim'],
        aggregation=config['aggregation'],
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout']
    )
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        class_weights=class_weights
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
    
    # Evaluate on test set
    print(f"\n{'='*70}")
    print("TEST SET EVALUATION")
    print("="*70)
    
    test_metrics = trainer.evaluate(test_loader)
    
    print(f"\nTest Metrics:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  Accuracy: {test_metrics['acc']:.4f}")
    print(f"  F1-Score: {test_metrics['f1']:.4f}")
    
    # Plot evaluation metrics
    plot_evaluation_metrics(
        y_true=test_metrics['labels'],
        y_pred=test_metrics['predictions'],
        save_path=str(fold_dir / "test_evaluation.png")
    )
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'test_metrics': test_metrics,
        'fold_idx': fold_idx
    }, fold_dir / "best_model.pth")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'patient_id': test_ids,
        'true_label': test_metrics['labels'],
        'predicted_prob': test_metrics['predictions']
    })
    predictions_df.to_csv(fold_dir / "test_predictions.csv", index=False)
    
    return {
        'fold_idx': fold_idx,
        'val_auc': best_val_auc,
        'test_auc': test_metrics['auc'],
        'test_acc': test_metrics['acc'],
        'test_f1': test_metrics['f1']
    }



