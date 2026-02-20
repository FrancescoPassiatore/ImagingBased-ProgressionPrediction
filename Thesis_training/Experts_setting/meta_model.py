"""
Meta-Model for Fusion of Expert Predictions
Combines CNN and LightGBM predictions using Logistic Regression
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, precision_recall_curve
)
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class MetaModel:
    """
    Meta-model for fusing expert predictions
    Supports both Logistic Regression and Weighted Fusion
    """
    
    def __init__(self, fusion_type: str = 'weighted', C: float = 1.0, max_iter: int = 1000):
        """
        Args:
            fusion_type: 'weighted' or 'logistic'
            C: Regularization parameter for Logistic Regression
            max_iter: Maximum iterations for solver
        """
        self.fusion_type = fusion_type
        self.optimal_weight = None
        self.is_fitted = False
        
        if fusion_type == 'logistic':
            self.model = LogisticRegression(
                C=C,
                max_iter=max_iter,
                random_state=42,
                solver='lbfgs'
            )
        else:
            self.model = None
    
    def fit(
        self,
        p_cnn: np.ndarray,
        p_lgbm: np.ndarray,
        y_true: np.ndarray
    ):
        """
        Train meta-model on validation predictions
        
        Args:
            p_cnn: CNN predictions (N,)
            p_lgbm: LightGBM predictions (N,)
            y_true: True labels (N,)
        """
        if self.fusion_type == 'weighted':
            # Find optimal weight by grid search
            self.optimal_weight = find_optimal_weight(
                p_cnn, p_lgbm, y_true
            )
            self.is_fitted = True
            
            print("\nWeighted Fusion Optimal Weight:")
            print(f"  CNN weight: {self.optimal_weight:.4f}")
            print(f"  LGBM weight: {1 - self.optimal_weight:.4f}")
            
        elif self.fusion_type == 'logistic':
            # Stack predictions as features
            X_meta = np.column_stack([p_cnn, p_lgbm])
            
            # Train logistic regression
            self.model.fit(X_meta, y_true)
            self.is_fitted = True
            
            # Print learned weights
            print("\nMeta-Model Learned Weights:")
            print(f"  CNN coefficient: {self.model.coef_[0][0]:.4f}")
            print(f"  LGBM coefficient: {self.model.coef_[0][1]:.4f}")
            print(f"  Intercept: {self.model.intercept_[0]:.4f}")
    
    def predict_proba(
        self,
        p_cnn: np.ndarray,
        p_lgbm: np.ndarray
    ) -> np.ndarray:
        """
        Get fused probability predictions
        
        Args:
            p_cnn: CNN predictions (N,)
            p_lgbm: LightGBM predictions (N,)
        
        Returns:
            p_fused: Fused predictions (N,)
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call fit() first.")
        
        if self.fusion_type == 'weighted':
            # Weighted average
            return self.optimal_weight * p_cnn + (1 - self.optimal_weight) * p_lgbm
        elif self.fusion_type == 'logistic':
            X_meta = np.column_stack([p_cnn, p_lgbm])
            return self.model.predict_proba(X_meta)[:, 1]
    
    def predict(
        self,
        p_cnn: np.ndarray,
        p_lgbm: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Get binary predictions
        
        Args:
            p_cnn: CNN predictions (N,)
            p_lgbm: LightGBM predictions (N,)
            threshold: Classification threshold
        
        Returns:
            predictions: Binary predictions (N,)
        """
        p_fused = self.predict_proba(p_cnn, p_lgbm)
        return (p_fused >= threshold).astype(int)


def find_optimal_weight(
    p_cnn: np.ndarray,
    p_lgbm: np.ndarray,
    y_true: np.ndarray
) -> float:
    """
    Find optimal weight for weighted fusion using grid search
    
    Args:
        p_cnn: CNN predictions (N,)
        p_lgbm: LightGBM predictions (N,)
        y_true: True labels (N,)
    
    Returns:
        optimal_weight: Weight for CNN (LightGBM weight = 1 - optimal_weight)
    """
    weights = np.linspace(0, 1, 101)
    best_auc = 0
    best_weight = 0.5
    
    auc_scores = []
    
    for w in weights:
        p_fused = w * p_cnn + (1 - w) * p_lgbm
        auc = roc_auc_score(y_true, p_fused)
        auc_scores.append(auc)
        
        if auc > best_auc:
            best_auc = auc
            best_weight = w
    
    print(f"\nWeighted Fusion Grid Search:")
    print(f"  Tested weights: {len(weights)}")
    print(f"  Best AUC: {best_auc:.4f}")
    print(f"  Best weight (CNN): {best_weight:.4f}")
    print(f"  AUC at w=0.0 (LGBM only): {auc_scores[0]:.4f}")
    print(f"  AUC at w=0.5 (equal): {auc_scores[50]:.4f}")
    print(f"  AUC at w=1.0 (CNN only): {auc_scores[-1]:.4f}")
    
    return best_weight


def compute_correlation(
    p_cnn: np.ndarray,
    p_lgbm: np.ndarray
) -> Dict[str, float]:
    """
    Compute correlation between expert predictions
    
    Args:
        p_cnn: CNN predictions (N,)
        p_lgbm: LightGBM predictions (N,)
    
    Returns:
        correlations: Dict with Pearson and Spearman correlations
    """
    from scipy.stats import pearsonr, spearmanr
    
    pearson_r, pearson_p = pearsonr(p_cnn, p_lgbm)
    spearman_r, spearman_p = spearmanr(p_cnn, p_lgbm)
    
    print("\n" + "="*60)
    print("EXPERT CORRELATION ANALYSIS")
    print("="*60)
    print(f"Pearson correlation:  {pearson_r:.4f} (p={pearson_p:.4e})")
    print(f"Spearman correlation: {spearman_r:.4f} (p={spearman_p:.4e})")
    
    if pearson_r < 0.7:
        print("✓ Low correlation indicates complementary information")
    else:
        print("⚠ High correlation suggests redundant information")
    
    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    strategy: str = 'youden'
) -> Tuple[float, Dict]:
    """
    Find optimal classification threshold
    
    Args:
        y_true: True labels (N,)
        y_pred_proba: Predicted probabilities (N,)
        strategy: 'youden', 'f1', or 'precision_recall'
    
    Returns:
        threshold: Optimal threshold
        metrics: Dict with threshold and associated metrics
    """
    if strategy == 'youden':
        # Youden's J statistic = Sensitivity + Specificity - 1
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        threshold = thresholds[optimal_idx]
        
        metrics = {
            'threshold': threshold,
            'sensitivity': tpr[optimal_idx],
            'specificity': 1 - fpr[optimal_idx],
            'youden_j': youden_j[optimal_idx]
        }
    
    elif strategy == 'f1':
        # Maximize F1 score
        thresholds = np.linspace(0, 1, 101)
        f1_scores = []
        
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            f1_scores.append(f1)
        
        optimal_idx = np.argmax(f1_scores)
        threshold = thresholds[optimal_idx]
        
        metrics = {
            'threshold': threshold,
            'f1': f1_scores[optimal_idx]
        }
    
    elif strategy == 'precision_recall':
        # Balance precision and recall
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        # F1 = 2 * (precision * recall) / (precision + recall)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        threshold = thresholds[optimal_idx]
        
        metrics = {
            'threshold': threshold,
            'precision': precision[optimal_idx],
            'recall': recall[optimal_idx],
            'f1': f1_scores[optimal_idx]
        }
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    print(f"\nOptimal threshold ({strategy}): {threshold:.4f}")
    for key, value in metrics.items():
        if key != 'threshold':
            print(f"  {key}: {value:.4f}")
    
    return threshold, metrics


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics
    
    Args:
        y_true: True labels (N,)
        y_pred_proba: Predicted probabilities (N,)
        threshold: Classification threshold
    
    Returns:
        metrics: Dict with all metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Compute metrics
    metrics = {
        'auc': roc_auc_score(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'threshold': threshold,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """Pretty print metrics"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print("="*60)
    print(f"AUC:         {metrics['auc']:.4f}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1 Score:    {metrics['f1']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    if 'threshold' in metrics:
        print(f"Threshold:   {metrics['threshold']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {metrics['tn']:3d}  FP: {metrics['fp']:3d}")
    print(f"  FN: {metrics['fn']:3d}  TP: {metrics['tp']:3d}")


def plot_roc_curves(
    y_true: np.ndarray,
    p_cnn: np.ndarray,
    p_lgbm: np.ndarray,
    p_fused: np.ndarray,
    save_path: str = None
):
    """
    Plot ROC curves for all models
    
    Args:
        y_true: True labels
        p_cnn: CNN predictions
        p_lgbm: LightGBM predictions
        p_fused: Fused predictions
        save_path: Path to save plot
    """
    from sklearn.metrics import roc_curve, auc
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # CNN
    fpr_cnn, tpr_cnn, _ = roc_curve(y_true, p_cnn)
    auc_cnn = auc(fpr_cnn, tpr_cnn)
    ax.plot(fpr_cnn, tpr_cnn, label=f'CNN (AUC={auc_cnn:.3f})', linewidth=2)
    
    # LGBM
    fpr_lgbm, tpr_lgbm, _ = roc_curve(y_true, p_lgbm)
    auc_lgbm = auc(fpr_lgbm, tpr_lgbm)
    ax.plot(fpr_lgbm, tpr_lgbm, label=f'LightGBM (AUC={auc_lgbm:.3f})', linewidth=2)
    
    # Fused
    fpr_fused, tpr_fused, _ = roc_curve(y_true, p_fused)
    auc_fused = auc(fpr_fused, tpr_fused)
    ax.plot(fpr_fused, tpr_fused, label=f'Fusion (AUC={auc_fused:.3f})', linewidth=2, linestyle='--')
    
    # Diagonal
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Expert Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curve to: {save_path}")
    
    plt.close()


def plot_prediction_distribution(
    p_cnn: np.ndarray,
    p_lgbm: np.ndarray,
    p_fused: np.ndarray,
    y_true: np.ndarray,
    save_path: str = None
):
    """
    Plot distribution of predictions by class
    
    Args:
        p_cnn: CNN predictions
        p_lgbm: LightGBM predictions
        p_fused: Fused predictions
        y_true: True labels
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    models = [
        ('CNN', p_cnn),
        ('LightGBM', p_lgbm),
        ('Fusion', p_fused)
    ]
    
    for ax, (name, preds) in zip(axes, models):
        # Plot by class
        ax.hist(preds[y_true == 0], bins=30, alpha=0.6, label='No Progression', color='blue')
        ax.hist(preds[y_true == 1], bins=30, alpha=0.6, label='Progression', color='red')
        
        ax.set_xlabel('Predicted Probability', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{name} Predictions', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved distribution plot to: {save_path}")
    
    plt.close()
