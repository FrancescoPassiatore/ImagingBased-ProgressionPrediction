"""
CNN Expert Model for IPF Progression Prediction
Processes CT slices with attention-based aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np


class AttentionAggregator(nn.Module):
    """Attention-based aggregation of slice-level features"""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, max_slices, feature_dim)
            lengths: (batch_size,) actual number of slices per patient
        
        Returns:
            aggregated: (batch_size, feature_dim)
        """
        batch_size, max_slices, feature_dim = x.shape
        
        # Compute attention scores
        attn_scores = self.attention(x).squeeze(-1)  # (batch_size, max_slices)
        
        # Create mask for padding
        mask = torch.arange(max_slices, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch_size, max_slices)
        
        # Weighted sum
        aggregated = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (batch_size, feature_dim)
        
        return aggregated


class CNNExpert(nn.Module):
    """
    CNN Expert for progression prediction
    Uses pre-extracted CNN features + optional hand-crafted and demographic features
    """
    
    def __init__(
        self,
        cnn_feature_dim: int,
        hand_feature_dim: int = 0,
        demo_feature_dim: int = 0,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        pooling_type: str = 'mean'  # 'mean', 'max', 'max_mean', or 'attention'
    ):
        super().__init__()
        
        self.cnn_feature_dim = cnn_feature_dim
        self.hand_feature_dim = hand_feature_dim
        self.demo_feature_dim = demo_feature_dim
        self.pooling_type = pooling_type
        
        # Aggregation layer for slice-level CNN features
        if pooling_type == 'attention':
            self.aggregator = AttentionAggregator(cnn_feature_dim, hidden_dim=128)
        
        # Calculate total feature dimension after aggregation
        # For max_mean, cnn features are doubled (concatenated max and mean)
        cnn_aggregated_dim = cnn_feature_dim * 2 if pooling_type == 'max_mean' else cnn_feature_dim
        total_dim = cnn_aggregated_dim + hand_feature_dim + demo_feature_dim
        
        # Simplified classification head (using LayerNorm to handle batch_size=1)
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, batch: Dict) -> torch.Tensor:
        """
        Args:
            batch: Dict with keys:
                - cnn_features: (batch_size, max_slices, cnn_dim)
                - hand_features: (batch_size, hand_dim) or None
                - demo_features: (batch_size, demo_dim) or None
                - lengths: (batch_size,)
        
        Returns:
            logits: (batch_size, 1)
        """
        cnn_features = batch['cnn_features']
        hand_features = batch.get('hand_features')
        demo_features = batch.get('demo_features')
        lengths = batch['lengths']
        
        # Aggregate slice-level CNN features
        if self.pooling_type == 'attention':
            aggregated_cnn = self.aggregator(cnn_features, lengths)
        elif self.pooling_type == 'max':
            # Max pooling
            batch_size, max_slices, _ = cnn_features.shape
            mask = torch.arange(max_slices, device=cnn_features.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1)
            masked_features = cnn_features.clone()
            masked_features[~mask.expand_as(cnn_features)] = -1e9
            aggregated_cnn = masked_features.max(dim=1)[0]
        elif self.pooling_type == 'mean':
            # Mean pooling
            batch_size, max_slices, _ = cnn_features.shape
            mask = torch.arange(max_slices, device=cnn_features.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1).float()
            sum_features = (cnn_features * mask).sum(dim=1)
            aggregated_cnn = sum_features / lengths.unsqueeze(-1).float()
        elif self.pooling_type == 'max_mean':
            # Concatenate max and mean pooling
            batch_size, max_slices, _ = cnn_features.shape
            mask = torch.arange(max_slices, device=cnn_features.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1)
            
            # Max pooling
            masked_features = cnn_features.clone()
            masked_features[~mask.expand_as(cnn_features)] = -1e9
            pooled_max = masked_features.max(dim=1)[0]
            
            # Mean pooling
            mask_float = mask.float()
            sum_features = (cnn_features * mask_float).sum(dim=1)
            pooled_mean = sum_features / lengths.unsqueeze(-1).float()
            
            # Concatenate
            aggregated_cnn = torch.cat([pooled_max, pooled_mean], dim=-1)
        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}. Use 'mean', 'max', 'max_mean', or 'attention'.")
        
        # Concatenate all features
        feature_list = [aggregated_cnn]
        
        if hand_features is not None:
            feature_list.append(hand_features)
        
        if demo_features is not None:
            feature_list.append(demo_features)
        
        combined_features = torch.cat(feature_list, dim=1)
        
        # Classification
        logits = self.classifier(combined_features)
        
        return logits
    
    def predict_proba(self, batch: Dict) -> np.ndarray:
        """
        Get probability predictions
        
        Returns:
            probabilities: (batch_size,) numpy array
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(batch)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        return probs


class CNNExpertTrainer:
    """Trainer for CNN Expert model"""
    
    def __init__(
        self,
        model: CNNExpert,
        device: str = 'cuda',
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        class_weights: torch.Tensor = None
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Loss with class weights
        if class_weights is not None:
            pos_weight = class_weights[1] / class_weights[0]
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
        else:
            self.criterion = nn.BCEWithLogitsLoss()
    
    def train_epoch(self, dataloader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            # Move data to device
            batch = self._batch_to_device(batch)
            
            # Forward pass
            logits = self.model(batch)
            labels = batch['labels'].unsqueeze(-1)
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(self, dataloader) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate model
        
        Returns:
            loss: average loss
            y_true: true labels
            y_pred_proba: predicted probabilities
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = self._batch_to_device(batch)
                
                logits = self.model(batch)
                labels = batch['labels'].unsqueeze(-1)
                
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                num_batches += 1
                
                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.squeeze(-1).cpu().numpy())
        
        return total_loss / num_batches, np.array(all_labels), np.array(all_probs)
    
    def predict(self, dataloader) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Get predictions for a dataset
        
        Returns:
            y_true: true labels
            y_pred_proba: predicted probabilities
            patient_ids: list of patient IDs
        """
        self.model.eval()
        all_probs = []
        all_labels = []
        all_patient_ids = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = self._batch_to_device(batch)
                
                logits = self.model(batch)
                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                
                all_probs.extend(probs)
                all_labels.extend(batch['labels'].cpu().numpy())
                all_patient_ids.extend(batch['patient_ids'])
        
        return np.array(all_labels), np.array(all_probs), all_patient_ids
    
    def _batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device"""
        batch_device = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_device[key] = value.to(self.device)
            else:
                batch_device[key] = value
        return batch_device
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
