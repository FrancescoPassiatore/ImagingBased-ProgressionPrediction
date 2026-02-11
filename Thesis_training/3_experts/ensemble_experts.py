# Three-Expert Ensemble Architecture for IPF Progression Prediction
# Expert 1: CNN + Demographics -> Risk
# Expert 2: Handcrafted Features + Demographics (LightGBM) -> Risk
# Expert 3: Fusion (weighted combination)

from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# EXPERT 1: CNN + DEMOGRAPHICS NEURAL NETWORK
# =============================================================================

class CNNExpert(nn.Module):
    """
    Expert 1: CT Scans -> ResNet50 Features -> Pooling -> MLP(Features, Demographics) -> Risk
    """
    
    def __init__(
        self,
        cnn_feature_dim: int = 2048,
        demo_feature_dim: int = 0,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.5,
        pooling_type: str = 'max'
    ):
        super().__init__()
        
        self.cnn_feature_dim = cnn_feature_dim
        self.demo_feature_dim = demo_feature_dim
        self.pooling_type = pooling_type
        
        print(f"\nCNN Expert Configuration:")
        print(f"  CNN features (per slice): {cnn_feature_dim}")
        print(f"  Demographic features: {demo_feature_dim}")
        print(f"  Pooling: {pooling_type}")
        
        # CNN processing branch
        self.cnn_branch = nn.Sequential(
            nn.Linear(cnn_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Demographics branch (if used)
        if demo_feature_dim > 0:
            self.demo_branch = nn.Sequential(
                nn.Linear(demo_feature_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            )
            fusion_input_dim = 512 + 64
        else:
            self.demo_branch = None
            fusion_input_dim = 512
        
        # Classification head
        layers = []
        input_dim = fusion_input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout if i == 0 else dropout * 0.8))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))  # Output: risk score (logit)
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(
        self,
        slice_features: torch.Tensor,
        lengths: torch.Tensor,
        demo_features: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            slice_features: (batch_size, max_slices, cnn_feature_dim)
            lengths: (batch_size,) - actual number of slices per patient
            demo_features: (batch_size, demo_dim) - optional
        
        Returns:
            logits: (batch_size, 1) - risk score logits
        """
        batch_size, max_slices, _ = slice_features.shape
        
        # Handle batch size 1 with BatchNorm
        is_training = self.training
        if batch_size == 1 and is_training:
            self.eval()
        
        # POOLING (directly on all slices)
        if self.pooling_type == 'max':
            mask = torch.arange(max_slices, device=slice_features.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1)
            masked_features = slice_features.clone()
            masked_features[~mask.expand_as(slice_features)] = -1e9
            pooled_cnn, _ = masked_features.max(dim=1)  # (batch_size, cnn_dim)
            
        elif self.pooling_type == 'mean':
            mask = torch.arange(max_slices, device=slice_features.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1).float()
            sum_features = (slice_features * mask).sum(dim=1)
            pooled_cnn = sum_features / lengths.unsqueeze(-1).float()
        
        # CNN BRANCH
        cnn_features = self.cnn_branch(pooled_cnn)  # (batch, 512)
        
        # DEMOGRAPHICS BRANCH (if used)
        if self.demo_branch is not None and demo_features is not None and demo_features.shape[1] > 0:
            demo_feat = self.demo_branch(demo_features)  # (batch, 64)
            combined = torch.cat([cnn_features, demo_feat], dim=-1)
        else:
            combined = cnn_features
        
        # CLASSIFICATION
        logits = self.classifier(combined)  # (batch_size, 1)
        
        # Restore training mode if changed
        if batch_size == 1 and is_training:
            self.train()
        
        return logits


# =============================================================================
# EXPERT 2: LIGHTGBM WITH HANDCRAFTED FEATURES + DEMOGRAPHICS
# =============================================================================

class LightGBMExpert:
    """
    Expert 2: Handcrafted Features + Demographics -> LightGBM -> Risk
    """
    
    def __init__(
        self,
        params: dict = None,
        feature_names: List[str] = None
    ):
        """
        Args:
            params: LightGBM parameters
            feature_names: Names of features (for interpretability)
        """
        self.params = params or {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        }
        
        self.feature_names = feature_names
        self.model = None
        self.is_trained = False
        
        print(f"\nLightGBM Expert Configuration:")
        print(f"  Features: {len(feature_names) if feature_names else 'Not specified'}")
        print(f"  Params: {self.params}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50
    ):
        """
        Train LightGBM model
        """
        print("\nTraining LightGBM Expert...")
        
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=self.feature_names
        )
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                feature_name=self.feature_names,
                reference=train_data
            )
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        callbacks = []
        if early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        self.is_trained = True
        
        print(f"✓ LightGBM training complete!")
        print(f"  Best iteration: {self.model.best_iteration}")
        print(f"  Best score: {self.model.best_score}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict risk scores
        
        Returns:
            probabilities: (n_samples,) - risk probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        importance = self.model.feature_importance(importance_type=importance_type)
        
        df = pd.DataFrame({
            'feature': self.feature_names or [f'feature_{i}' for i in range(len(importance))],
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def save(self, filepath: Path):
        """Save model"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        self.model.save_model(str(filepath))
        print(f"LightGBM model saved to: {filepath}")
    
    def load(self, filepath: Path):
        """Load model"""
        self.model = lgb.Booster(model_file=str(filepath))
        self.is_trained = True
        print(f"LightGBM model loaded from: {filepath}")


# =============================================================================
# EXPERT 3: FUSION LAYER
# =============================================================================

class FusionExpert:
    """
    Expert 3: Weighted fusion of CNN Expert and LightGBM Expert
    
    Fusion methods:
    - 'fixed': Fixed weights (a, b) where a + b = 1
    - 'learned': Learn optimal weights on validation set
    - 'stacking': Train a meta-model (logistic regression) on validation predictions
    """
    
    def __init__(self, fusion_method: str = 'learned'):
        """
        Args:
            fusion_method: 'fixed', 'learned', or 'stacking'
        """
        self.fusion_method = fusion_method
        self.weights = None
        self.meta_model = None
        
        print(f"\nFusion Expert Configuration:")
        print(f"  Method: {fusion_method}")
    
    def fit(
        self,
        cnn_preds: np.ndarray,
        lgb_preds: np.ndarray,
        y_true: np.ndarray,
        initial_weights: Tuple[float, float] = (0.5, 0.5)
    ):
        """
        Learn fusion weights on validation set
        
        Args:
            cnn_preds: (n_samples,) - CNN expert predictions
            lgb_preds: (n_samples,) - LightGBM expert predictions
            y_true: (n_samples,) - true labels
            initial_weights: (a, b) - initial weights for CNN and LGB
        """
        print("\nFitting Fusion Expert...")
        
        if self.fusion_method == 'fixed':
            # Use fixed weights
            a, b = initial_weights
            if abs(a + b - 1.0) > 1e-6:
                print(f"  Warning: Weights don't sum to 1. Normalizing...")
                total = a + b
                a, b = a / total, b / total
            
            self.weights = (a, b)
            print(f"  Fixed weights: CNN={a:.3f}, LGB={b:.3f}")
        
        elif self.fusion_method == 'learned':
            # Grid search for optimal weights
            best_auc = 0
            best_weights = initial_weights
            
            for a in np.arange(0, 1.01, 0.05):
                b = 1 - a
                fused = a * cnn_preds + b * lgb_preds
                auc = roc_auc_score(y_true, fused)
                
                if auc > best_auc:
                    best_auc = auc
                    best_weights = (a, b)
            
            self.weights = best_weights
            print(f"  Learned weights: CNN={best_weights[0]:.3f}, LGB={best_weights[1]:.3f}")
            print(f"  Validation AUC: {best_auc:.4f}")
        
        elif self.fusion_method == 'stacking':
            # Train logistic regression meta-model
            from sklearn.linear_model import LogisticRegression
            
            X_meta = np.column_stack([cnn_preds, lgb_preds])
            self.meta_model = LogisticRegression()
            self.meta_model.fit(X_meta, y_true)
            
            # Extract weights for interpretation
            coeffs = self.meta_model.coef_[0]
            total = np.abs(coeffs).sum()
            normalized_weights = np.abs(coeffs) / total
            
            self.weights = tuple(normalized_weights)
            print(f"  Stacking weights (normalized): CNN={self.weights[0]:.3f}, LGB={self.weights[1]:.3f}")
            
            # Validation performance
            fused = self.meta_model.predict_proba(X_meta)[:, 1]
            auc = roc_auc_score(y_true, fused)
            print(f"  Validation AUC: {auc:.4f}")
    
    def predict(
        self,
        cnn_preds: np.ndarray,
        lgb_preds: np.ndarray
    ) -> np.ndarray:
        """
        Fuse predictions from both experts
        
        Returns:
            fused_preds: (n_samples,) - final risk predictions
        """
        if self.weights is None and self.meta_model is None:
            raise ValueError("Fusion expert not fitted yet!")
        
        if self.fusion_method in ['fixed', 'learned']:
            a, b = self.weights
            return a * cnn_preds + b * lgb_preds
        
        elif self.fusion_method == 'stacking':
            X_meta = np.column_stack([cnn_preds, lgb_preds])
            return self.meta_model.predict_proba(X_meta)[:, 1]
    
    def get_weights(self) -> Tuple[float, float]:
        """Get fusion weights"""
        if self.weights is None:
            raise ValueError("Fusion expert not fitted yet!")
        return self.weights


# =============================================================================
# ENSEMBLE COORDINATOR
# =============================================================================

class ThreeExpertEnsemble:
    """
    Coordinates all three experts:
    1. CNN Expert (Neural Network)
    2. LightGBM Expert (Gradient Boosting)
    3. Fusion Expert (Weighted combination)
    """
    
    def __init__(
        self,
        cnn_expert: CNNExpert,
        lgb_expert: LightGBMExpert,
        fusion_expert: FusionExpert,
        device: str = 'cuda'
    ):
        self.cnn_expert = cnn_expert.to(device)
        self.lgb_expert = lgb_expert
        self.fusion_expert = fusion_expert
        self.device = device
        
        print("\n" + "="*70)
        print("THREE-EXPERT ENSEMBLE INITIALIZED")
        print("="*70)
        print("Expert 1: CNN + Demographics (Neural Network)")
        print("Expert 2: Handcrafted Features + Demographics (LightGBM)")
        print("Expert 3: Fusion Layer")
        print("="*70)
    
    def predict_cnn(
        self,
        dataloader,
        return_probs: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions from CNN expert
        
        Returns:
            predictions: (n_samples,) - risk scores
            labels: (n_samples,) - true labels
        """
        self.cnn_expert.eval()
        
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
                logits = self.cnn_expert(cnn_features, lengths, demo_features).squeeze(-1)
                
                if return_probs:
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_preds.extend(probs)
                else:
                    all_preds.extend(logits.cpu().numpy())
                
                all_labels.extend(labels.cpu().numpy())
        
        return np.array(all_preds), np.array(all_labels)
    
    def predict_lgb(
        self,
        X: np.ndarray,
        y: np.ndarray = None
    ) -> np.ndarray:
        """
        Get predictions from LightGBM expert
        
        Returns:
            predictions: (n_samples,) - risk scores
        """
        return self.lgb_expert.predict(X)
    
    def predict_ensemble(
        self,
        cnn_dataloader,
        lgb_features: np.ndarray,
        y_true: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """
        Get predictions from all experts and fused prediction
        
        Returns:
            dict with keys: 'cnn', 'lgb', 'fusion', 'labels'
        """
        # CNN predictions
        cnn_preds, labels = self.predict_cnn(cnn_dataloader, return_probs=True)
        
        # LightGBM predictions
        lgb_preds = self.predict_lgb(lgb_features)
        
        # Fusion
        fused_preds = self.fusion_expert.predict(cnn_preds, lgb_preds)
        
        return {
            'cnn': cnn_preds,
            'lgb': lgb_preds,
            'fusion': fused_preds,
            'labels': labels if y_true is None else y_true
        }
    
    def evaluate(
        self,
        cnn_dataloader,
        lgb_features: np.ndarray,
        y_true: np.ndarray = None,
        threshold: float = 0.5
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all experts
        
        Returns:
            dict with metrics for each expert
        """
        preds = self.predict_ensemble(cnn_dataloader, lgb_features, y_true)
        
        results = {}
        
        for expert_name in ['cnn', 'lgb', 'fusion']:
            pred = preds[expert_name]
            labels = preds['labels']
            
            pred_binary = (pred >= threshold).astype(int)
            
            results[expert_name] = {
                'auc': roc_auc_score(labels, pred),
                'accuracy': accuracy_score(labels, pred_binary),
                'f1': f1_score(labels, pred_binary, zero_division=0)
            }
        
        return results


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def plot_expert_comparison(results: Dict[str, Dict[str, float]], save_path: Path = None):
    """
    Plot comparison of all three experts
    """
    experts = list(results.keys())
    metrics = list(results[experts[0]].keys())
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        values = [results[expert][metric] for expert in experts]
        
        axes[i].bar(experts, values, alpha=0.7, color=['steelblue', 'seagreen', 'coral'])
        axes[i].set_ylabel(metric.upper())
        axes[i].set_title(f'{metric.upper()} Comparison')
        axes[i].set_ylim(0, 1.0)
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.suptitle('Expert Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_fusion_weights(fusion_expert: FusionExpert, save_path: Path = None):
    """
    Plot fusion weights
    """
    weights = fusion_expert.get_weights()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    experts = ['CNN Expert', 'LightGBM Expert']
    colors = ['steelblue', 'seagreen']
    
    bars = ax.bar(experts, weights, alpha=0.7, color=colors)
    ax.set_ylabel('Weight', fontsize=12)
    ax.set_title('Fusion Weights', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, w in zip(bars, weights):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{w:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


print("✓ Three-Expert Ensemble module loaded successfully!")