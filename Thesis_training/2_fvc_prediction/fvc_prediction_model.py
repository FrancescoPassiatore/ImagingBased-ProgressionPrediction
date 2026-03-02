from pathlib import Path
import sys
from typing import Dict, List, Tuple
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))
from fvc_utilities import create_dataloaders, compute_class_weights

# Feature column definitions
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


class FVCPredictionModel(nn.Module):
    """
    Model for predicting FVC at 52 weeks from:
    - CNN features from CT slices
    - Baseline FVC (FVC at week 0)
    - Optional: Hand-crafted features
    - Optional: Demographic features
    
    Architecture:
    1. CNN features per slice -> Max pooling across slices
    2. Concatenate: [pooled_CNN, FVC_baseline, hand_features, demographics]
    3. MLP -> FVC prediction at 52 weeks
    """
    
    def __init__(
        self,
        cnn_feature_dim: int = 2048,
        hand_feature_dim: int = 0,
        demo_feature_dim: int = 0,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.5,
        use_batch_norm: bool = True,
        pooling_type: str = 'max',  # 'max', 'mean', or 'attention'
        use_fvc_branch: bool = True
    ):
        super().__init__()
        
        self.cnn_feature_dim = cnn_feature_dim
        self.hand_feature_dim = hand_feature_dim
        self.demo_feature_dim = demo_feature_dim
        self.pooling_type = pooling_type
        self.use_fvc_branch = use_fvc_branch

        print(f"\nFVC Prediction Model Configuration:")
        print(f"  CNN features (per slice): {cnn_feature_dim}")
        print(f"  Baseline FVC: 1 (with dedicated branch: {use_fvc_branch})")
        print(f"  Hand-crafted features: {hand_feature_dim}")
        print(f"  Demographic features: {demo_feature_dim}")
        print(f"  Pooling strategy: {pooling_type}")
        print(f"Architecture : separate branches (hand+demo processed independently)")
        
        # Attention pooling (if selected)
        if pooling_type == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(cnn_feature_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )

        # === SEPARATE FEATURE PROCESSING BRANCHES ===
        
        # 1. FVC baseline branch (most important!)
        if use_fvc_branch:
            self.fvc_branch = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Dropout(dropout * 0.3),  # Less dropout for important feature
                nn.Linear(64, 128),
                nn.ReLU()
            )
            fvc_output_dim = 128
        else:
            fvc_output_dim = 1
        
        # 2. CNN features branch (IMAGE features)
        # For max_mean pooling, input dimension is doubled (max + mean concatenated)
        cnn_input_dim = cnn_feature_dim * 2 if pooling_type == 'max_mean' else cnn_feature_dim
        self.cnn_branch = nn.Sequential(
            nn.Linear(cnn_input_dim, 512),
            nn.BatchNorm1d(512) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256) if use_batch_norm else nn.Identity(),
            nn.ReLU()
        )
        cnn_output_dim = 256

        #3. Hand-crafted radiomics branch 
        if hand_feature_dim > 0:
            self.hand_branch = nn.Sequential(
                nn.Linear(hand_feature_dim, 32),
                nn.BatchNorm1d(32) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),  # Moderate dropout
                nn.Linear(32, 64),
                nn.ReLU()
            )
            hand_output_dim = 64
        else:
            self.hand_branch = None
            hand_output_dim = 0

        #4. Demographics branch 
        if demo_feature_dim > 0:
            self.demo_branch = nn.Sequential(
                nn.Linear(demo_feature_dim, 16),
                nn.ReLU(),
                nn.Dropout(dropout * 0.3),  # Low dropout - simple features
                nn.Linear(16, 32),
                nn.ReLU()
            )
            demo_output_dim = 32
        else:
            self.demo_branch = None
            demo_output_dim = 0

        # === FUSION LAYER with ATTENTION ===
        fusion_input_dim = fvc_output_dim + cnn_output_dim + hand_output_dim + demo_output_dim
        
        print(f"\n  Fusion layer input: {fusion_input_dim}")
        print(f"    - FVC branch: {fvc_output_dim}")
        print(f"    - CNN branch: {cnn_output_dim}")
        print(f"    - Hand-crafted branch: {hand_output_dim}")
        print(f"    - Demographics branch: {demo_output_dim}")

        # Optional: Feature attention weights
        # Learns which features are most important
        self.feature_attention = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_input_dim // 4),
            nn.ReLU(),
            nn.Linear(fusion_input_dim // 4, fusion_input_dim),
            nn.Sigmoid()
        )

        # Fusion MLP
        fusion_layers = []
        input_dim = fusion_input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            fusion_layers.append(nn.Linear(input_dim, hidden_dim))
            if use_batch_norm:
                fusion_layers.append(nn.BatchNorm1d(hidden_dim))
            fusion_layers.append(nn.ReLU())
            fusion_layers.append(nn.Dropout(dropout if i == 0 else dropout * 0.7))
            input_dim = hidden_dim
        
        # Final prediction layer
        fusion_layers.append(nn.Linear(input_dim, 1))
        
        self.fusion_head = nn.Sequential(*fusion_layers)
        

        
    
    def forward(
        self, 
        slice_features: torch.Tensor,
        lengths: torch.Tensor,
        fvc_baseline: torch.Tensor,
        hand_features: torch.Tensor = None,
        demo_features : torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            slice_features: (batch_size, max_slices, cnn_feature_dim)
            lengths: (batch_size,) - actual number of slices per patient
            fvc_baseline: (batch_size, 1) - FVC at week 0
            patient_features: (batch_size, hand_dim + demo_dim) - optional
        
        Returns:
            predictions: (batch_size, 1) - predicted FVC at 52 weeks (normalized)
        """
        batch_size, max_slices, _ = slice_features.shape
        
        # Pool CNN features across slices
        if self.pooling_type == 'max':
            # Create mask for valid slices
            mask = torch.arange(max_slices, device=slice_features.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1)  # (batch_size, max_slices, 1)
            
            # Set invalid slices to very negative value
            masked_features = slice_features.clone()
            masked_features[~mask.expand_as(slice_features)] = -1e9
            
            # Max pooling
            pooled_cnn, _ = masked_features.max(dim=1)  # (batch_size, cnn_dim)
            
        elif self.pooling_type == 'mean':
            # Average pooling (only over valid slices)
            mask = torch.arange(max_slices, device=slice_features.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1).float()  # (batch_size, max_slices, 1)
            
            sum_features = (slice_features * mask).sum(dim=1)  # (batch_size, cnn_dim)
            pooled_cnn = sum_features / lengths.unsqueeze(-1).float()
            
        elif self.pooling_type == 'max_mean':
            # Concatenate max and mean pooling
            mask = torch.arange(max_slices, device=slice_features.device)[None, :] < lengths[:, None]
            mask_expanded = mask.unsqueeze(-1)  # (batch_size, max_slices, 1)
            
            # Max pooling
            masked_features = slice_features.clone()
            masked_features[~mask_expanded.expand_as(slice_features)] = -1e9
            max_pooled, _ = masked_features.max(dim=1)  # (batch_size, cnn_dim)
            
            # Mean pooling
            mask_float = mask_expanded.float()
            sum_features = (slice_features * mask_float).sum(dim=1)  # (batch_size, cnn_dim)
            mean_pooled = sum_features / lengths.unsqueeze(-1).float()
            
            # Concatenate
            pooled_cnn = torch.cat([max_pooled, mean_pooled], dim=1)  # (batch_size, 2*cnn_dim)
            
        elif self.pooling_type == 'attention':
            # Attention-based pooling
            # Compute attention scores
            attn_scores = self.attention(slice_features)  # (batch_size, max_slices, 1)
            
            # Mask invalid slices
            mask = torch.arange(max_slices, device=slice_features.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1)  # (batch_size, max_slices, 1)
            attn_scores = attn_scores.masked_fill(~mask, -1e9)
            
            # Softmax normalization
            attn_weights = torch.softmax(attn_scores, dim=1)  # (batch_size, max_slices, 1)
            
            # Weighted sum
            pooled_cnn = (slice_features * attn_weights).sum(dim=1)  # (batch_size, cnn_dim)
        
        # === 2. PROCESS THROUGH BRANCHES ===
        branch_outputs = []
        
        # FVC branch 
        if self.use_fvc_branch:
            fvc_features = self.fvc_branch(fvc_baseline)  # (batch, 64)
        else:
            fvc_features = fvc_baseline  # (batch, 1)

        branch_outputs.append(fvc_features)
        
        # CNN branch
        cnn_features = self.cnn_branch(pooled_cnn)  # (batch, 512)
        branch_outputs.append(cnn_features)

        if self.hand_branch is not None and hand_features is not None:
            hand_feat = self.hand_branch(hand_features)
            branch_outputs.append(hand_feat)

        if self.demo_branch is not None and demo_features is not None:
            demo_feat = self.demo_branch(demo_features)
            branch_outputs.append(demo_feat)

        #Fusion
        combined = torch.cat(branch_outputs,dim=1)

        attention_weights = self.feature_attention(combined)

        combined = combined * attention_weights

        #Final prediction
        predictions = self.fusion_head(combined)

        return predictions
        


class FVCModelTrainer:
    """
    Trainer for FVC prediction model
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        use_scheduler: bool = True
    ):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.use_scheduler = use_scheduler
        if use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',  # Minimize validation loss
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        
        # Loss function (MSE for regression)
        self.criterion = nn.MSELoss()
        
        self.history = {
            'train_loss': [], 'train_mae': [], 'train_r2': [],
            'val_loss': [], 'val_mae': [], 'val_r2': []
        }
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        all_losses = []
        all_preds = []
        all_targets = []
        
        for batch in dataloader:
            cnn_features = batch['cnn_features'].to(self.device)
            hand_features = batch.get('hand_features')
            if hand_features is not None:
                hand_features = hand_features.to(self.device)
            demo_features = batch.get('demo_features')
            if demo_features is not None:
                demo_features = demo_features.to(self.device)
            
            lengths = batch['lengths'].to(self.device)
            fvc_baseline = batch['fvc_baseline'].to(self.device)
            fvc_target = batch['fvc_52weeks'].to(self.device)
            
            # Forward pass with separate hand and demo features
            predictions = self.model(cnn_features, lengths, fvc_baseline, hand_features, demo_features)
            
            # Ensure predictions and targets have same shape
            predictions = predictions.squeeze(-1)  # Remove last dimension if present, keeps batch dim
            loss = self.criterion(predictions, fvc_target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Record metrics - convert to list to handle both scalars and arrays
            all_losses.append(loss.item())
            all_preds.extend(predictions.detach().cpu().numpy().tolist())
            all_targets.extend(fvc_target.cpu().numpy().tolist())
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        metrics = {
            'loss': np.mean(all_losses),
            'mae': mean_absolute_error(all_targets, all_preds),
            'r2': r2_score(all_targets, all_preds)
        }
        
        return metrics
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate on validation/test set"""
        self.model.eval()
        
        all_losses = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                cnn_features = batch['cnn_features'].to(self.device)
                hand_features = batch.get('hand_features')
                if hand_features is not None:
                    hand_features = hand_features.to(self.device)
                demo_features = batch.get('demo_features')
                if demo_features is not None:
                    demo_features = demo_features.to(self.device)
                
                lengths = batch['lengths'].to(self.device)
                fvc_baseline = batch['fvc_baseline'].to(self.device)
                fvc_target = batch['fvc_52weeks'].to(self.device)
                
                # Forward pass with separate hand and demo features
                predictions = self.model(cnn_features, lengths, fvc_baseline, hand_features, demo_features)
                
                # Ensure predictions and targets have same shape
                predictions = predictions.squeeze(-1)  # Remove last dimension if present, keeps batch dim
                loss = self.criterion(predictions, fvc_target)
                
                # Record metrics - convert to list to handle both scalars and arrays
                all_losses.append(loss.item())
                all_preds.extend(predictions.cpu().numpy().tolist())  # tolist() handles 0-d arrays
                all_targets.extend(fvc_target.cpu().numpy().tolist())
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        metrics = {
            'loss': np.mean(all_losses),
            'mae': mean_absolute_error(all_targets, all_preds),
            'rmse': np.sqrt(mean_squared_error(all_targets, all_preds)),
            'r2': r2_score(all_targets, all_preds),
            'predictions': all_preds,
            'targets': all_targets
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
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            
            if self.use_scheduler:
                self.scheduler.step(val_metrics['loss'])
            
            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_mae'].append(train_metrics['mae'])
            self.history['train_r2'].append(train_metrics['r2'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['val_r2'].append(val_metrics['r2'])
            
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.2f}, R²: {train_metrics['r2']:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.2f}, R²: {val_metrics['r2']:.4f}")
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return best_val_loss
    
    def plot_training_history(self, save_path: str = None):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE
        axes[1].plot(self.history['train_mae'], label='Train')
        axes[1].plot(self.history['val_mae'], label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE (mL)')
        axes[1].set_title('Mean Absolute Error')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # R²
        axes[2].plot(self.history['train_r2'], label='Train')
        axes[2].plot(self.history['val_r2'], label='Val')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('R²')
        axes[2].set_title('R² Score')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()


def plot_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = None
):
    """
    Plot comprehensive evaluation metrics for FVC prediction
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Scatter plot with regression line
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=50)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Compute metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    axes[0, 0].set_xlabel('True FVC at 52 weeks (mL)', fontsize=11)
    axes[0, 0].set_ylabel('Predicted FVC at 52 weeks (mL)', fontsize=11)
    axes[0, 0].set_title(f'Predictions vs True Values\nMAE: {mae:.2f} mL, RMSE: {rmse:.2f} mL, R²: {r2:.3f}',
                        fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residual plot
    residuals = y_pred - y_true
    axes[0, 1].scatter(y_true, residuals, alpha=0.6, s=50)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('True FVC at 52 weeks (mL)', fontsize=11)
    axes[0, 1].set_ylabel('Residuals (mL)', fontsize=11)
    axes[0, 1].set_title('Residual Plot', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribution of residuals
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Residuals (mL)', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title(f'Distribution of Residuals\nMean: {residuals.mean():.2f} mL, Std: {residuals.std():.2f} mL',
                        fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Bland-Altman plot
    mean_values = (y_true + y_pred) / 2
    diff_values = y_pred - y_true
    mean_diff = diff_values.mean()
    std_diff = diff_values.std()
    
    axes[1, 1].scatter(mean_values, diff_values, alpha=0.6, s=50)
    axes[1, 1].axhline(y=mean_diff, color='b', linestyle='-', lw=2, label='Mean Difference')
    axes[1, 1].axhline(y=mean_diff + 1.96*std_diff, color='r', linestyle='--', lw=2, label='±1.96 SD')
    axes[1, 1].axhline(y=mean_diff - 1.96*std_diff, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Mean of True and Predicted FVC (mL)', fontsize=11)
    axes[1, 1].set_ylabel('Difference (Predicted - True) (mL)', fontsize=11)
    axes[1, 1].set_title('Bland-Altman Plot', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Print detailed metrics
    print("\n" + "="*60)
    print("REGRESSION METRICS")
    print("="*60)
    print(f"MAE:  {mae:.2f} mL")
    print(f"RMSE: {rmse:.2f} mL")
    print(f"R²:   {r2:.3f}")
    print(f"\nResidual Statistics:")
    print(f"  Mean: {residuals.mean():.2f} mL")
    print(f"  Std:  {residuals.std():.2f} mL")
    print(f"  Min:  {residuals.min():.2f} mL")
    print(f"  Max:  {residuals.max():.2f} mL")