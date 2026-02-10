"""
Attention-based MIL (Multiple Instance Learning) Model for IPF Progression Prediction
=====================================================================================

This module implements an attention-based pooling mechanism as an alternative to max pooling.
The attention mechanism learns to weight slices based on their relevance to the prediction task.

Key improvements:
1. Gated Attention: Uses both tanh and sigmoid gates for more expressive attention
2. Multi-head Attention: Optional multi-head mechanism for capturing different aspects
3. Interpretability: Attention weights show which slices are most important
4. Flexibility: Supports both single-head and multi-head configurations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np


class GatedAttentionMIL(nn.Module):
    """
    Gated Attention mechanism for MIL
    
    Based on "Attention-based Deep Multiple Instance Learning" (Ilse et al., 2018)
    Uses both tanh and sigmoid gates for more expressive attention weights.
    
    Args:
        input_dim: Dimension of input features (e.g., 2048 for ResNet50)
        hidden_dim: Dimension of attention hidden layer
        dropout: Dropout rate for regularization
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Attention network with gating
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.attention_w = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (batch_size, max_slices, input_dim)
            lengths: (batch_size,) - actual number of slices per patient
            
        Returns:
            pooled: (batch_size, input_dim) - attention-weighted features
            attention_weights: (batch_size, max_slices) - normalized attention weights
        """
        batch_size, max_slices, _ = features.shape
        
        # Compute gated attention
        A_V = self.attention_V(features)  # (B, max_slices, hidden_dim)
        A_U = self.attention_U(features)  # (B, max_slices, hidden_dim)
        A = self.attention_w(A_V * A_U)   # (B, max_slices, 1), element-wise gating
        A = A.squeeze(-1)                 # (B, max_slices)
        
        # Create attention mask (mask out padding)
        mask = torch.arange(max_slices, device=features.device)[None, :] < lengths[:, None]
        A = A.masked_fill(~mask, -1e9)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(A, dim=1)  # (B, max_slices)
        
        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum of features
        pooled = torch.bmm(attention_weights.unsqueeze(1), features).squeeze(1)  # (B, input_dim)
        
        return pooled, attention_weights


class MultiHeadAttentionMIL(nn.Module):
    """
    Multi-head Attention mechanism for MIL
    
    Uses multiple attention heads to capture different aspects of the data.
    Each head learns different attention patterns.
    
    Args:
        input_dim: Dimension of input features
        hidden_dim: Dimension of attention hidden layer per head
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.25):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multiple attention heads
        self.attention_heads = nn.ModuleList([
            GatedAttentionMIL(input_dim, hidden_dim, dropout)
            for _ in range(num_heads)
        ])
        
        # Projection layer to combine heads
        self.projection = nn.Linear(input_dim * num_heads, input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (batch_size, max_slices, input_dim)
            lengths: (batch_size,)
            
        Returns:
            pooled: (batch_size, input_dim) - combined multi-head features
            attention_weights: (batch_size, num_heads, max_slices) - attention weights per head
        """
        head_outputs = []
        head_attentions = []
        
        # Process each head
        for head in self.attention_heads:
            pooled, attn = head(features, lengths)
            head_outputs.append(pooled)
            head_attentions.append(attn)
        
        # Concatenate head outputs
        combined = torch.cat(head_outputs, dim=-1)  # (B, input_dim * num_heads)
        
        # Project back to input_dim
        pooled = self.projection(combined)  # (B, input_dim)
        pooled = self.dropout(pooled)
        
        # Stack attention weights
        attention_weights = torch.stack(head_attentions, dim=1)  # (B, num_heads, max_slices)
        
        return pooled, attention_weights


class AttentionMILProgressionModel(nn.Module):
    """
    Complete Attention-based MIL Model for IPF Progression Prediction
    
    Architecture:
    1. CNN feature extraction per slice (pre-computed)
    2. Attention-based pooling across slices
    3. Optional patient-level features (hand-crafted + demographics)
    4. Multi-layer classification head
    
    Args:
        cnn_feature_dim: Dimension of CNN features per slice
        hand_feature_dim: Number of hand-crafted features
        demo_feature_dim: Number of demographic features
        attention_hidden_dim: Hidden dimension for attention mechanism
        attention_type: 'gated' or 'multihead'
        num_attention_heads: Number of heads (if multihead)
        hidden_dims: List of hidden layer dimensions for classifier
        dropout: Dropout rate
        use_batch_norm: Whether to use batch normalization
        use_feature_branches: Whether to use separate processing branches
    """
    
    def __init__(
        self,
        cnn_feature_dim: int = 2048,
        hand_feature_dim: int = 0,
        demo_feature_dim: int = 0,
        attention_hidden_dim: int = 128,
        attention_type: str = 'gated',  # 'gated' or 'multihead'
        num_attention_heads: int = 4,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.5,
        use_batch_norm: bool = True,
        use_feature_branches: bool = True
    ):
        super().__init__()
        
        self.cnn_feature_dim = cnn_feature_dim
        self.hand_feature_dim = hand_feature_dim
        self.demo_feature_dim = demo_feature_dim
        self.attention_type = attention_type
        
        print(f"\n{'='*70}")
        print("ATTENTION-BASED MIL MODEL CONFIGURATION")
        print(f"{'='*70}")
        print(f"  CNN features (per slice): {cnn_feature_dim}")
        print(f"  Hand-crafted features: {hand_feature_dim}")
        print(f"  Demographic features: {demo_feature_dim}")
        print(f"  Attention type: {attention_type}")
        if attention_type == 'multihead':
            print(f"  Number of attention heads: {num_attention_heads}")
        print(f"  Attention hidden dim: {attention_hidden_dim}")
        print(f"  Feature branches: {use_feature_branches}")
        print(f"  Hidden dimensions: {hidden_dims}")
        print(f"  Dropout: {dropout}")
        
        # === ATTENTION POOLING ===
        if attention_type == 'gated':
            self.attention = GatedAttentionMIL(
                input_dim=cnn_feature_dim,
                hidden_dim=attention_hidden_dim,
                dropout=dropout * 0.5  # Less dropout in attention
            )
        elif attention_type == 'multihead':
            self.attention = MultiHeadAttentionMIL(
                input_dim=cnn_feature_dim,
                hidden_dim=attention_hidden_dim,
                num_heads=num_attention_heads,
                dropout=dropout * 0.5
            )
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")
        
        # === FEATURE PROCESSING BRANCHES ===
        if use_feature_branches:
            # CNN features branch
            self.cnn_branch = nn.Sequential(
                nn.Linear(cnn_feature_dim, 512),
                nn.BatchNorm1d(512) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            cnn_output_dim = 512
            
            # Patient features branch (hand-crafted + demographics)
            patient_feature_dim = hand_feature_dim + demo_feature_dim
            if patient_feature_dim > 0:
                self.patient_branch = nn.Sequential(
                    nn.Linear(patient_feature_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.7)
                )
                patient_output_dim = 64
            else:
                self.patient_branch = None
                patient_output_dim = 0
            
            fusion_input_dim = cnn_output_dim + patient_output_dim
            
        else:
            # Simple concatenation (no branches)
            self.cnn_branch = None
            self.patient_branch = None
            fusion_input_dim = cnn_feature_dim + hand_feature_dim + demo_feature_dim
        
        self.use_feature_branches = use_feature_branches
        
        print(f"  Fusion layer input: {fusion_input_dim}")
        
        # === CLASSIFICATION HEAD ===
        layers = []
        input_dim = fusion_input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout if i == 0 else dropout * 0.8))
            input_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.classifier = nn.Sequential(*layers)
        
        print(f"{'='*70}\n")
    
    def forward(
        self,
        slice_features: torch.Tensor,
        lengths: torch.Tensor,
        patient_features: torch.Tensor = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Args:
            slice_features: (batch_size, max_slices, cnn_feature_dim)
            lengths: (batch_size,) - actual number of slices per patient
            patient_features: (batch_size, hand_dim + demo_dim) - optional
            return_attention: Whether to return attention weights
            
        Returns:
            logits: (batch_size, 1) - progression probability logits
            attention_weights: (batch_size, max_slices) or (batch_size, num_heads, max_slices)
                               - only if return_attention=True
        """
        batch_size = slice_features.shape[0]
        
        # Handle batch size 1 with BatchNorm: temporarily set to eval mode
        is_training = self.training
        if batch_size == 1 and is_training:
            self.eval()
        
        # === 1. ATTENTION POOLING ===
        pooled_cnn, attention_weights = self.attention(slice_features, lengths)
        
        # === 2. PROCESS THROUGH BRANCHES ===
        if self.use_feature_branches:
            # CNN branch
            if self.cnn_branch is not None:
                cnn_features = self.cnn_branch(pooled_cnn)
            else:
                cnn_features = pooled_cnn
            
            # Patient features branch
            if self.patient_branch is not None and patient_features is not None:
                patient_feat = self.patient_branch(patient_features)
                combined = torch.cat([cnn_features, patient_feat], dim=-1)
            else:
                combined = cnn_features
        else:
            # Simple concatenation
            if patient_features is not None and patient_features.shape[1] > 0:
                combined = torch.cat([pooled_cnn, patient_features], dim=-1)
            else:
                combined = pooled_cnn
        
        # === 3. CLASSIFICATION ===
        logits = self.classifier(combined)
        
        # Restore training mode if it was changed
        if batch_size == 1 and is_training:
            self.train()
        
        if return_attention:
            return logits, attention_weights
        return logits
    
    def predict_proba(
        self,
        slice_features: torch.Tensor,
        lengths: torch.Tensor,
        patient_features: torch.Tensor = None
    ) -> Dict:
        """
        Return probabilities with attention weights for interpretation
        """
        logits, attention_weights = self.forward(
            slice_features, lengths, patient_features, return_attention=True
        )
        
        probs = torch.sigmoid(logits).squeeze(-1)  # (batch_size,)
        
        # Get most important slice (highest attention weight)
        if self.attention_type == 'multihead':
            # Average attention across heads
            avg_attention = attention_weights.mean(dim=1)  # (batch_size, max_slices)
            most_important_slice = avg_attention.argmax(dim=1)
        else:
            most_important_slice = attention_weights.argmax(dim=1)
        
        return {
            'prob': probs,
            'attention_weights': attention_weights,
            'important_slice_idx': most_important_slice,
            'num_slices': lengths,
            'attention_type': self.attention_type
        }


# =============================================================================
# COMPARISON: Attention MIL vs Max Pooling
# =============================================================================

class PoolingComparisonModel(nn.Module):
    """
    Model that supports both attention and max pooling for comparison
    """
    
    def __init__(
        self,
        cnn_feature_dim: int = 2048,
        hand_feature_dim: int = 0,
        demo_feature_dim: int = 0,
        pooling_type: str = 'attention',  # 'attention', 'max', 'mean'
        attention_hidden_dim: int = 128,
        attention_type: str = 'gated',
        num_attention_heads: int = 4,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.5,
        use_batch_norm: bool = True,
        use_feature_branches: bool = True
    ):
        super().__init__()
        
        self.pooling_type = pooling_type
        self.cnn_feature_dim = cnn_feature_dim
        
        print(f"\nPooling Comparison Model - Using: {pooling_type}")
        
        # Initialize attention if needed
        if pooling_type == 'attention':
            if attention_type == 'gated':
                self.attention = GatedAttentionMIL(
                    input_dim=cnn_feature_dim,
                    hidden_dim=attention_hidden_dim,
                    dropout=dropout * 0.5
                )
            elif attention_type == 'multihead':
                self.attention = MultiHeadAttentionMIL(
                    input_dim=cnn_feature_dim,
                    hidden_dim=attention_hidden_dim,
                    num_heads=num_attention_heads,
                    dropout=dropout * 0.5
                )
        else:
            self.attention = None
        
        # Feature branches and classifier
        if use_feature_branches:
            self.cnn_branch = nn.Sequential(
                nn.Linear(cnn_feature_dim, 512),
                nn.BatchNorm1d(512) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            cnn_output_dim = 512
            
            patient_feature_dim = hand_feature_dim + demo_feature_dim
            if patient_feature_dim > 0:
                self.patient_branch = nn.Sequential(
                    nn.Linear(patient_feature_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.7)
                )
                patient_output_dim = 64
            else:
                self.patient_branch = None
                patient_output_dim = 0
            
            fusion_input_dim = cnn_output_dim + patient_output_dim
        else:
            self.cnn_branch = None
            self.patient_branch = None
            fusion_input_dim = cnn_feature_dim + hand_feature_dim + demo_feature_dim
        
        self.use_feature_branches = use_feature_branches
        
        # Classifier
        layers = []
        input_dim = fusion_input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout if i == 0 else dropout * 0.8))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        
        self.classifier = nn.Sequential(*layers)
    
    def pool_features(self, features: torch.Tensor, lengths: torch.Tensor):
        """Apply pooling based on pooling_type"""
        batch_size, max_slices, _ = features.shape
        
        if self.pooling_type == 'attention':
            pooled, attention_weights = self.attention(features, lengths)
            return pooled, attention_weights
        
        elif self.pooling_type == 'max':
            # Max pooling with masking
            mask = torch.arange(max_slices, device=features.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1)
            masked_features = features.clone()
            masked_features[~mask.expand_as(features)] = -1e9
            pooled, _ = masked_features.max(dim=1)
            return pooled, None
        
        elif self.pooling_type == 'mean':
            # Mean pooling with masking
            mask = torch.arange(max_slices, device=features.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1).float()
            sum_features = (features * mask).sum(dim=1)
            pooled = sum_features / lengths.unsqueeze(-1).float()
            return pooled, None
        
        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}")
    
    def forward(self, slice_features, lengths, patient_features=None, return_attention=False):
        # Pooling
        pooled_cnn, attention_weights = self.pool_features(slice_features, lengths)
        
        # Feature branches
        if self.use_feature_branches:
            if self.cnn_branch is not None:
                cnn_features = self.cnn_branch(pooled_cnn)
            else:
                cnn_features = pooled_cnn
            
            if self.patient_branch is not None and patient_features is not None:
                patient_feat = self.patient_branch(patient_features)
                combined = torch.cat([cnn_features, patient_feat], dim=-1)
            else:
                combined = cnn_features
        else:
            if patient_features is not None and patient_features.shape[1] > 0:
                combined = torch.cat([pooled_cnn, patient_features], dim=-1)
            else:
                combined = pooled_cnn
        
        # Classification
        logits = self.classifier(combined)
        
        if return_attention:
            return logits, attention_weights
        return logits


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def visualize_attention_weights(
    attention_weights: np.ndarray,
    patient_id: str,
    num_slices: int,
    save_path: str = None
):
    """
    Visualize attention weights for a single patient
    
    Args:
        attention_weights: (max_slices,) or (num_heads, max_slices)
        patient_id: Patient identifier
        num_slices: Actual number of slices
        save_path: Optional path to save plot
    """
    import matplotlib.pyplot as plt
    
    if len(attention_weights.shape) == 1:
        # Single-head attention
        fig, ax = plt.subplots(figsize=(12, 4))
        
        weights = attention_weights[:num_slices]
        slices = np.arange(num_slices)
        
        ax.bar(slices, weights, alpha=0.7, color='steelblue')
        ax.set_xlabel('Slice Index')
        ax.set_ylabel('Attention Weight')
        ax.set_title(f'Attention Weights - Patient {patient_id}')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Mark highest attention slice
        max_idx = weights.argmax()
        ax.bar(max_idx, weights[max_idx], color='red', alpha=0.8, 
               label=f'Max attention (slice {max_idx})')
        ax.legend()
        
    else:
        # Multi-head attention
        num_heads = attention_weights.shape[0]
        fig, axes = plt.subplots(num_heads, 1, figsize=(12, 3*num_heads))
        
        if num_heads == 1:
            axes = [axes]
        
        for head_idx in range(num_heads):
            weights = attention_weights[head_idx, :num_slices]
            slices = np.arange(num_slices)
            
            axes[head_idx].bar(slices, weights, alpha=0.7)
            axes[head_idx].set_xlabel('Slice Index')
            axes[head_idx].set_ylabel('Attention Weight')
            axes[head_idx].set_title(f'Head {head_idx+1} - Patient {patient_id}')
            axes[head_idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compare_pooling_methods_on_batch(
    models: Dict[str, nn.Module],
    batch: Dict,
    device: str = 'cuda'
) -> Dict:
    """
    Compare different pooling methods on the same batch
    
    Args:
        models: Dict of {method_name: model}
        batch: Batch from dataloader
        device: 'cuda' or 'cpu'
        
    Returns:
        results: Dict with predictions and metrics for each method
    """
    results = {}
    
    cnn_features = batch['cnn_features'].to(device)
    patient_features = batch.get('patient_features')
    if patient_features is not None:
        patient_features = patient_features.to(device)
    lengths = batch['lengths'].to(device)
    labels = batch['labels'].to(device)
    
    for method_name, model in models.items():
        model.eval()
        with torch.no_grad():
            logits = model(cnn_features, lengths, patient_features)
            probs = torch.sigmoid(logits).squeeze(-1)
            
            results[method_name] = {
                'predictions': probs.cpu().numpy(),
                'labels': labels.cpu().numpy()
            }
    
    return results


if __name__ == "__main__":
    print("Attention MIL Model Module")
    print("="*70)
    print("\nAvailable models:")
    print("  1. AttentionMILProgressionModel - Pure attention-based model")
    print("  2. PoolingComparisonModel - Flexible model for comparing pooling methods")
    print("\nAttention types:")
    print("  - 'gated': Gated attention mechanism (Ilse et al., 2018)")
    print("  - 'multihead': Multi-head attention for diverse attention patterns")