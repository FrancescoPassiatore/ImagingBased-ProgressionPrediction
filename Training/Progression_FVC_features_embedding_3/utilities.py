from datetime import datetime
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import glob
import cv2
import copy
import random
from collections import defaultdict
from typing import Dict, List
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, Sampler
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_curve

HAND_FEATURE_ORDER = [
    'approx_vol',
    'avg_num_tissue_pixel',
    'avg_tissue',
    'avg_tissue_thickness',
    'avg_tissue_by_total',
    'avg_tissue_by_lung',
    'mean',
    'skew',
    'kurtosis',
    'age',
    'sex',
    'smoking_status'
]

NORMALIZE_FEATURES = [
    'approx_vol',
    'avg_num_tissue_pixel',
    'avg_tissue',
    'avg_tissue_thickness',
    'avg_tissue_by_total',
    'avg_tissue_by_lung',
    'mean',
    'skew',
    'kurtosis',
    'age'
]
class FusionMLPOptuna(nn.Module):
    """Flexible MLP architecture for Optuna optimization"""
    
    def __init__(self, img_dim, hand_dim, trial):
        super().__init__()
        
        # Sample architecture parameters
        img_hidden = trial.suggest_categorical('img_hidden', [128, 256, 512])
        hand_hidden = trial.suggest_categorical('hand_hidden', [32, 64, 128])
        n_fusion_layers = trial.suggest_int('n_fusion_layers', 1, 3)
        dropout = trial.suggest_float('dropout', 0.3, 0.7, step=0.1)
        activation_name = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu'])
        use_layer_norm = trial.suggest_categorical('use_layer_norm', [True, False])
        
        # Activation function
        if activation_name == 'relu':
            activation = nn.ReLU()
        elif activation_name == 'leaky_relu':
            activation = nn.LeakyReLU(0.2)
        else:
            activation = nn.ELU()
        
        norm_layer = nn.LayerNorm if use_layer_norm else nn.BatchNorm1d
        
        # Image embedding branch
        self.img_fc = nn.Sequential(
            nn.Linear(img_dim, img_hidden),
            norm_layer(img_hidden),
            activation,
            nn.Dropout(dropout)
        )
        
        # Handcrafted features branch
        self.hand_fc = nn.Sequential(
            nn.Linear(hand_dim, hand_hidden),
            norm_layer(hand_hidden),
            activation,
            nn.Dropout(dropout)
        )
        
        # Fusion layers
        fusion_input = img_hidden + hand_hidden
        fusion_layers = []
        prev_dim = fusion_input
        
        for i in range(n_fusion_layers):
            hidden_dim = trial.suggest_categorical(f'fusion_layer_{i}', [64, 128, 256, 512])
            fusion_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                norm_layer(hidden_dim),
                activation,
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        fusion_layers.append(nn.Linear(prev_dim, 1))
        self.fusion = nn.Sequential(*fusion_layers)
        
        # Weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x_img, x_hand):
        img_feat = self.img_fc(x_img)
        hand_feat = self.hand_fc(x_hand)
        fused = torch.cat([img_feat, hand_feat], dim=1)
        return self.fusion(fused)


def objective(trial, train_loader, val_loader, device, img_dim, hand_dim, max_epochs=50):
    """Optuna objective function"""
    
    # Sample hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw'])
    scheduler_name = trial.suggest_categorical('scheduler', ['plateau', 'cosine', 'none'])
    use_class_weights = trial.suggest_categorical('use_class_weights', [True, False])
    grad_clip = trial.suggest_float('grad_clip', 0.5, 2.0, step=0.5)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    # Recreate dataloaders with new batch size if needed
    if batch_size != train_loader.batch_size:
        train_loader = DataLoader(
            train_loader.dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0
        )
        val_loader = DataLoader(
            val_loader.dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0
        )
    
    # Create model
    model = FusionMLPOptuna(img_dim, hand_dim, trial).to(device)
    
    # Setup loss function
    if use_class_weights:
        labels = [train_loader.dataset[i]['y'].item() for i in range(len(train_loader.dataset))]
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        pos_weight = torch.tensor([neg_count / max(pos_count, 1)]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Setup optimizer
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Setup scheduler
    if scheduler_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
    elif scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    else:
        scheduler = None
    
    # Compute normalization statistics
    all_x_img, all_x_hand = [], []
    for batch in train_loader:
        all_x_img.append(batch['x_img'])
        all_x_hand.append(batch['x_hand'])
    all_x_img = torch.cat(all_x_img, dim=0)
    all_x_hand = torch.cat(all_x_hand, dim=0)
    
    norm_stats = {
        'img_mean': all_x_img.mean(dim=0).to(device),
        'img_std': (all_x_img.std(dim=0) + 1e-8).to(device),
        'hand_mean': all_x_hand.mean(dim=0).to(device),
        'hand_std': (all_x_hand.std(dim=0) + 1e-8).to(device)
    }
    
    best_val_auc = 0
    patience_counter = 0
    early_stop_patience = 15
    
    # Training loop
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        
        for batch in train_loader:
            x_img = batch['x_img'].to(device)
            x_hand = batch['x_hand'].to(device)
            y = batch['y'].to(device)
            
            # Normalize
            x_img = (x_img - norm_stats['img_mean']) / norm_stats['img_std']
            x_hand = (x_hand - norm_stats['hand_mean']) / norm_stats['hand_std']
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(x_img, x_hand)
            loss = criterion(logits, y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
            train_labels.extend(y.cpu().numpy().flatten())
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                x_img = batch['x_img'].to(device)
                x_hand = batch['x_hand'].to(device)
                y = batch['y'].to(device)
                
                # Normalize
                x_img = (x_img - norm_stats['img_mean']) / norm_stats['img_std']
                x_hand = (x_hand - norm_stats['hand_mean']) / norm_stats['hand_std']
                
                logits = model(x_img, x_hand)
                loss = criterion(logits, y)
                
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                val_labels.extend(y.cpu().numpy().flatten())
        
        # Compute metrics
        train_auc = roc_auc_score(train_labels, train_preds)
        val_auc = roc_auc_score(val_labels, val_preds)
        
        # Update scheduler
        if scheduler is not None:
            if scheduler_name == 'plateau':
                scheduler.step(val_auc)
            else:
                scheduler.step()
        
        # Track best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Report intermediate results to Optuna
        trial.report(val_auc, epoch)
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            break
        
        # Handle pruning based on intermediate value
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_val_auc


def run_optuna_study(train_loader, val_loader, device, n_trials=50, study_name="fusion_mlp_optimization"):
    """Run Optuna hyperparameter optimization"""
    
    # Get feature dimensions
    sample = train_loader.dataset[0]
    img_dim = sample['x_img'].shape[0]
    hand_dim = sample['x_hand'].shape[0]
    
    print("\n" + "="*80)
    print(f"OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    print(f"Feature dimensions: img={img_dim}, hand={hand_dim}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Number of trials: {n_trials}")
    print("="*80 + "\n")
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',  # Maximize validation AUC
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, device, img_dim, hand_dim),
        n_trials=n_trials,
        show_progress_bar=True,
        callbacks=[lambda study, trial: print(f"\nTrial {trial.number}: Val AUC = {trial.value:.4f}")]
    )
    
    # Print results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80 + "\n")
    
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == TrialState.COMPLETE])}")
    
    print("\n" + "-"*80)
    print("BEST TRIAL")
    print("-"*80)
    best_trial = study.best_trial
    print(f"\nValidation AUC: {best_trial.value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save study
    df = study.trials_dataframe()
    df.to_csv(f'optuna_results_{timestamp}.csv', index=False)
    print(f"\n✓ Results saved to: optuna_results_{timestamp}.csv")
    
    # Save best params
    with open(f'best_params_{timestamp}.json', 'w') as f:
        json.dump(best_trial.params, f, indent=2)
    print(f"✓ Best parameters saved to: best_params_{timestamp}.json")
    
    # Visualization (requires optuna and plotly)
    try:
        
        import plotly
        # Optimization history
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(f'optuna_history_{timestamp}.html')
        
        # Parameter importance
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(f'optuna_importance_{timestamp}.html')
        
        # Parallel coordinate plot
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(f'optuna_parallel_{timestamp}.html')
        
        print(f"✓ Visualizations saved as HTML files")
    except ImportError:
        print("⚠️  Install plotly for visualizations: pip install plotly")
    
    return study, best_trial


def train_with_best_params(best_params, train_loader, val_loader, test_loader, device, max_epochs=100):
    """Train final model with best hyperparameters"""
    
    print("\n" + "="*80)
    print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
    print("="*80 + "\n")
    
    # Get dimensions
    sample = train_loader.dataset[0]
    img_dim = sample['x_img'].shape[0]
    hand_dim = sample['x_hand'].shape[0]
    
    # Create trial-like object for model creation
    class FakeTrial:
        def __init__(self, params):
            self.params = params
        def suggest_categorical(self, name, choices):
            return self.params[name]
        def suggest_int(self, name, low, high):
            return self.params[name]
        def suggest_float(self, name, low, high, **kwargs):
            return self.params[name]
    
    fake_trial = FakeTrial(best_params)
    
    # Recreate dataloaders with best batch size
    batch_size = best_params.get('batch_size', 32)
    train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_loader.dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_loader.dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    model = FusionMLPOptuna(img_dim, hand_dim, fake_trial).to(device)
    
    # Setup training components
    if best_params['use_class_weights']:
        labels = [train_loader.dataset[i]['y'].item() for i in range(len(train_loader.dataset))]
        pos_weight = torch.tensor([labels.count(0) / max(labels.count(1), 1)]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    if best_params['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    
    if best_params['scheduler'] == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    elif best_params['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    else:
        scheduler = None
    
    # Normalization
    all_x_img = torch.cat([batch['x_img'] for batch in train_loader], dim=0)
    all_x_hand = torch.cat([batch['x_hand'] for batch in train_loader], dim=0)
    norm_stats = {
        'img_mean': all_x_img.mean(dim=0).to(device),
        'img_std': (all_x_img.std(dim=0) + 1e-8).to(device),
        'hand_mean': all_x_hand.mean(dim=0).to(device),
        'hand_std': (all_x_hand.std(dim=0) + 1e-8).to(device)
    }
    
    # Training loop
    best_val_auc = 0
    history = {'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': []}
    
    for epoch in range(max_epochs):
        # Train
        model.train()
        train_loss, train_preds, train_labels = 0, [], []
        
        for batch in train_loader:
            x_img = (batch['x_img'].to(device) - norm_stats['img_mean']) / norm_stats['img_std']
            x_hand = (batch['x_hand'].to(device) - norm_stats['hand_mean']) / norm_stats['hand_std']
            y = batch['y'].to(device)
            
            optimizer.zero_grad()
            logits = model(x_img, x_hand)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), best_params['grad_clip'])
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
            train_labels.extend(y.cpu().numpy().flatten())
        
        # Validate
        model.eval()
        val_loss, val_preds, val_labels = 0, [], []
        
        with torch.no_grad():
            for batch in val_loader:
                x_img = (batch['x_img'].to(device) - norm_stats['img_mean']) / norm_stats['img_std']
                x_hand = (batch['x_hand'].to(device) - norm_stats['hand_mean']) / norm_stats['hand_std']
                y = batch['y'].to(device)
                
                logits = model(x_img, x_hand)
                val_loss += criterion(logits, y).item()
                val_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                val_labels.extend(y.cpu().numpy().flatten())
        
        # Metrics
        train_auc = roc_auc_score(train_labels, train_preds)
        val_auc = roc_auc_score(val_labels, val_preds)
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_auc'].append(val_auc)
        
        if scheduler:
            scheduler.step(val_auc) if best_params['scheduler'] == 'plateau' else scheduler.step()
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({'model': model.state_dict(), 'norm_stats': norm_stats, 'params': best_params}, 
                      'final_best_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{max_epochs} - Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
    
    # Test evaluation
    checkpoint = torch.load('final_best_model.pth')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_img = (batch['x_img'].to(device) - norm_stats['img_mean']) / norm_stats['img_std']
            x_hand = (batch['x_hand'].to(device) - norm_stats['hand_mean']) / norm_stats['hand_std']
            
            logits = model(x_img, x_hand)
            test_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            test_labels.extend(batch['y'].numpy().flatten())
    
    test_auc = roc_auc_score(test_labels, test_preds)
    test_acc = accuracy_score(test_labels, (np.array(test_preds) >= 0.5).astype(int))
    test_f1 = f1_score(test_labels, (np.array(test_preds) >= 0.5).astype(int))
    
    print(f"\n{'='*80}")
    print("FINAL TEST RESULTS")
    print(f"{'='*80}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    
    return model, history

#======================================================================================
#PROGRESSION PREDICTION RISK LOADERS
#======================================================================================
class IPFDataLoaderPredictorProgression:
    """Loads and prepares data from CSV"""

    def __init__(self, csv_path: str, csv_features_path: str, npy_dir: str):
        """
        Args:
            csv_path: Path to CSV with [Patient, Weeks, FVC, Slice_files, FVC slope, FVC intercept0]
            csv_features_path: Path to CSV with handcrafted features
            npy_dir: Directory path for npy image files
        """
        self.df = pd.read_csv(csv_path)
        self.npy_dir = npy_dir
        self.df_features = pd.read_csv(csv_features_path)

        print(f"✅ Loaded {len(self.df)} records from CSV")
        print(f"✅ Unique patients in CSV: {self.df['Patient'].nunique()}")
        print(f"✅ NPY directory: {npy_dir}")
        print(f"📋 Columns: {self.df.columns.tolist()}")

        # Verify demographic columns exist
        required_demo_cols = ['Age', 'Sex', 'SmokingStatus']
        missing_cols = [col for col in required_demo_cols if col not in self.df.columns]
        if missing_cols:
            print(f"⚠️  WARNING: Missing demographic columns: {missing_cols}")
        else:
            print(f"✅ All demographic columns present: {required_demo_cols}")

        # Verify NPY availability
        self._verify_npy_availability()

    def _verify_npy_availability(self):
        """Verify that each patient in the CSV has a folder with .npy files"""
        patients_csv = set(self.df['Patient'].unique())
        patients_npy = set([d for d in os.listdir(self.npy_dir)
                           if os.path.isdir(os.path.join(self.npy_dir, d))])

        missing = patients_csv - patients_npy
        extra = patients_npy - patients_csv

        if missing:
            print(f"⚠️  {len(missing)} patients in CSV without NPY folder: {list(missing)[:5]}...")
        if extra:
            print(f"ℹ️  {len(extra)} NPY folders not in CSV (will be ignored)")

        available = patients_csv & patients_npy
        print(f"✅ {len(available)} patients with complete data (CSV + NPY)")

    def get_patient_data(self):
        """Extract patient data and features with NaN handling"""
        patient_data = {}
        features_dict = {}
        
        # First pass: collect all features to compute means
        all_features = {
            'approx_vol': [], 'avg_num_tissue_pixel': [], 'avg_tissue': [],
            'avg_tissue_thickness': [], 'avg_tissue_by_total': [], 
            'avg_tissue_by_lung': [], 'mean': [], 'skew': [], 
            'kurtosis': [], 'age': []
        }

        for patient_id in self.df['Patient'].unique():
            patient_df = self.df[self.df['Patient'] == patient_id]
            patient_df_features = self.df_features[self.df_features['Patient'] == patient_id]
            
            patient_df_sorted = patient_df.sort_values("Weeks")
            
            weeks = patient_df_sorted["Weeks"].astype(float).to_numpy()
            fvc_values = patient_df_sorted["FVC"].astype(float).to_numpy()
            
            # Optional: remove NaN/Inf in FVC
            mask = np.isfinite(weeks) & np.isfinite(fvc_values)
            weeks = weeks[mask]
            fvc_values = fvc_values[mask]
            

            # Check if patient has NPY files
            patient_npy_folder = os.path.join(self.npy_dir, patient_id)
            if not os.path.exists(patient_npy_folder):
                continue

            npy_files = sorted(glob.glob(os.path.join(patient_npy_folder, "*.npy")))
            if not npy_files:
                continue
            
            # Collect non-NaN values for computing means
            for key in all_features.keys():
                if key == 'age':
                    val = patient_df['Age'].iloc[0]
                else:
                    col_map = {
                        'approx_vol': 'ApproxVol_30_60',
                        'avg_num_tissue_pixel': 'Avg_NumTissuePixel_30_60',
                        'avg_tissue': 'Avg_Tissue_30_60',
                        'avg_tissue_thickness': 'Avg_Tissue_thickness_30_60',
                        'avg_tissue_by_total': 'Avg_TissueByTotal_30_60',
                        'avg_tissue_by_lung': 'Avg_TissueByLung_30_60',
                        'mean': 'Mean_30_60',
                        'skew': 'Skew_30_60',
                        'kurtosis': 'Kurtosis_30_60'
                    }
                    val = patient_df_features[col_map[key]].iloc[0]
                
                if pd.notna(val) and not np.isinf(val):
                    all_features[key].append(float(val))
        
        # Compute means for imputation
        feature_means = {k: np.mean(v) if len(v) > 0 else 0.0 
                        for k, v in all_features.items()}
        
        print("\n📊 Feature means for NaN imputation:")
        for k, v in feature_means.items():
            print(f"   {k}: {v:.4f}")
        
        nan_count = 0
        inf_count = 0

        # Second pass: create feature dictionaries with NaN replacement
        for patient_id in self.df['Patient'].unique():
            patient_df = self.df[self.df['Patient'] == patient_id]
            patient_df_features = self.df_features[self.df_features['Patient'] == patient_id]

            # Check if patient has NPY files
            patient_npy_folder = os.path.join(self.npy_dir, patient_id)
            if not os.path.exists(patient_npy_folder):
                continue

            npy_files = sorted(glob.glob(os.path.join(patient_npy_folder, "*.npy")))
            if not npy_files:
                continue

            # Helper function to get feature value with NaN handling
            def get_safe_value(value, feature_name):
                nonlocal nan_count, inf_count
                if pd.isna(value):
                    nan_count += 1
                    return feature_means[feature_name]
                if np.isinf(value):
                    inf_count += 1
                    return feature_means[feature_name]
                return float(value)

            # Extract handcrafted + tabular features with NaN handling
            features_dict[patient_id] = {
                'approx_vol': get_safe_value(
                    patient_df_features['ApproxVol_30_60'].iloc[0], 'approx_vol'),
                'avg_num_tissue_pixel': get_safe_value(
                    patient_df_features['Avg_NumTissuePixel_30_60'].iloc[0], 'avg_num_tissue_pixel'),
                'avg_tissue': get_safe_value(
                    patient_df_features['Avg_Tissue_30_60'].iloc[0], 'avg_tissue'),
                'avg_tissue_thickness': get_safe_value(
                    patient_df_features['Avg_Tissue_thickness_30_60'].iloc[0], 'avg_tissue_thickness'),
                'avg_tissue_by_total': get_safe_value(
                    patient_df_features['Avg_TissueByTotal_30_60'].iloc[0], 'avg_tissue_by_total'),
                'avg_tissue_by_lung': get_safe_value(
                    patient_df_features['Avg_TissueByLung_30_60'].iloc[0], 'avg_tissue_by_lung'),
                'mean': get_safe_value(
                    patient_df_features['Mean_30_60'].iloc[0], 'mean'),
                'skew': get_safe_value(
                    patient_df_features['Skew_30_60'].iloc[0], 'skew'),
                'kurtosis': get_safe_value(
                    patient_df_features['Kurtosis_30_60'].iloc[0], 'kurtosis'),
                'age': get_safe_value(
                    patient_df['Age'].iloc[0], 'age'),
                'sex': 1.0 if patient_df['Sex'].iloc[0] == 'Male' else 0.0,
                'smoking_status': 0.0 if patient_df['SmokingStatus'].iloc[0] == 'Never smoked' else 1.0
            }

            patient_data[patient_id] = {
                'slices': npy_files,
                'n_slices': len(npy_files),
                "weeks":weeks,
                "fvc_values":fvc_values,
            }
        
        if nan_count > 0:
            print(f"\n⚠️  Replaced {nan_count} NaN values with feature means")
        if inf_count > 0:
            print(f"⚠️  Replaced {inf_count} Inf values with feature means")
        if nan_count == 0 and inf_count == 0:
            print(f"\n✅ No NaN or Inf values detected in features")

        return patient_data, features_dict

    def __len__(self):
        return len(self.df)


class SliceFeatureDataset(Dataset):
    """Dataset for extracting features from individual CT slices"""
    
    def __init__(self, patient_list, patient_data):
        """
        Args:
            patient_list: List of patient IDs
            patient_data: Dictionary with patient slice paths
        """
        self.items = []
        for pid in patient_list:
            if pid not in patient_data:
                continue
            for slice_path in patient_data[pid]['slices']:
                self.items.append((pid, slice_path))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pid, slice_path = self.items[idx]

        # Load image
        img = np.load(slice_path)
        
        # Convert to 3 channels if grayscale
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=0)
        
        # Ensure correct shape (C, H, W)
        if img.shape[-1] == 3:  # If (H, W, C)
            img = np.transpose(img, (2, 0, 1))  # Convert to (C, H, W)

        img = torch.FloatTensor(img)

        return {
            'image': img,
            'patient_id': pid
        }


class PatientMLPDataset(Dataset):
    """Dataset combining CNN embeddings with handcrafted features for patient-level prediction"""
    
    def __init__(
        self,
        label_csv,
        embeddings_dict,
        handcrafted_dict,
        patient_list,
        feature_stats = None
    ):
        """
        Args:
            label_csv: Path to CSV with patient labels
            embeddings_dict: Dictionary of patient embeddings from CNN
            handcrafted_dict: Dictionary of handcrafted features
            patient_list: List of patient IDs to include
        """
        self.df = pd.read_csv(label_csv)
        self.df = self.df[self.df['Patient'].isin(patient_list)].reset_index(drop=True)

        self.embeddings = embeddings_dict
        self.handcrafted = handcrafted_dict
        
        # Filter out patients without embeddings or features
        valid_patients = []
        for pid in self.df['Patient']:
            if pid in self.embeddings and pid in self.handcrafted:
                valid_patients.append(pid)
        
        self.df = self.df[self.df['Patient'].isin(valid_patients)].reset_index(drop=True)
        
        if len(self.df) < len(patient_list):
            print(f"⚠️  Warning: Only {len(self.df)}/{len(patient_list)} patients have complete data")

        self.feature_stats = feature_stats

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row['Patient']

        raw_feats = self.handcrafted[pid]
        norm_feats = []

        for k in HAND_FEATURE_ORDER:
            v = raw_feats[k]
            if k in self.feature_stats:
                mean = self.feature_stats[k]['mean']
                std = self.feature_stats[k]['std']
                norm_feats.append((v - mean) / std)
            else:
                # sex, smoking_status
                norm_feats.append(float(v))
                
        x_img = torch.FloatTensor(self.embeddings[pid])
        x_hand = torch.FloatTensor(norm_feats)
        y = torch.FloatTensor([row['fvc_52']])

        return {
            'x_img': x_img,
            'x_hand': x_hand,
            'y': y,
            'patient_id': pid
        }

class SimpleFusionMLP(nn.Module):
    def __init__(self, img_dim=320, hand_dim=12, hidden=32, dropout=0.5):
        super().__init__()
        
        # Un solo layer per img
        self.img_fc = nn.Sequential(
            nn.Linear(img_dim, hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # Un solo layer per hand (opzionale)
        self.hand_fc = nn.Sequential(
            nn.Linear(hand_dim, hidden//4),  # Molto più piccolo
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # Output diretto (NO fusion layer intermedia!)
        self.output = nn.Linear(hidden + hidden//4, 1)
    
    def forward(self, img_emb, hand_feat):
        img_out = self.img_fc(img_emb)
        hand_out = self.hand_fc(hand_feat)
        fused = torch.cat([img_out, hand_out], dim=1)
        return self.output(fused)
    
class FusionMLP(nn.Module):
    """MLP that fuses CNN embeddings with handcrafted features"""
    
    def __init__(self, img_dim=320, hand_dim=12,img_hidden=256, hand_hidden=64, fusion_layers=[128], dropout=0.4, activation='leaky_relu', use_layer_norm=False):
        """
        Args:
            img_dim: Dimension of CNN embedding
            hand_dim: Dimension of handcrafted features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        self.img_dim = img_dim
        self.hand_dim = hand_dim

        if activation == 'leaky_relu':
            act_fn = nn.LeakyReLU()
        else:
            act_fn = nn.ReLU()

        norm_layer= nn.LayerNorm if use_layer_norm else nn.BatchNorm1d

        img_layers = [
            nn.Linear(img_dim, img_hidden),
            norm_layer(img_hidden) if not use_layer_norm else norm_layer(img_hidden),
            act_fn,
            nn.Dropout(dropout)
        ]
        self.img_fc = nn.Sequential(*img_layers)
        
        hand_layers = [
            nn.Linear(hand_dim, hand_hidden),
            norm_layer(hand_hidden) if not use_layer_norm else norm_layer(hand_hidden),
            act_fn,
            nn.Dropout(dropout)
        ]
        self.hand_fc = nn.Sequential(*hand_layers)


        
        # Fusion layers
        fusion_input_dim = img_hidden + hand_hidden
        layers = []
        prev_dim = fusion_input_dim
        
        for hidden_dim in fusion_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                norm_layer(hidden_dim) if not use_layer_norm else norm_layer(hidden_dim),
                act_fn,
                nn.Dropout(0.5)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.fusion = nn.Sequential(*layers)
    
    def forward(self, x_img, x_hand):
        # Process each modality
        img_feat = self.img_fc(x_img)
        hand_feat = self.hand_fc(x_hand)
        
        # Concatenate
        fused = torch.cat([img_feat, hand_feat], dim=1)
        
        # Final prediction
        out = self.fusion(fused)
        return out



def train_epoch(model, loader, criterion, optimizer, device, grad_clip=0.5):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        x_img = batch['x_img'].to(device)
        x_hand = batch['x_hand'].to(device)
        y = batch['y'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(x_img, x_hand)
        loss = criterion(logits, y)
        
        # Backward pass
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.sigmoid(logits).detach().cpu().numpy()
        labels = y.detach().cpu().numpy()
        
        all_preds.extend(preds.flatten())
        all_labels.extend(labels.flatten())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(loader)
    auc = roc_auc_score(all_labels, all_preds)
    
    return avg_loss, auc


def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_pids = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            x_img = batch['x_img'].to(device)
            x_hand = batch['x_hand'].to(device)
            y = batch['y'].to(device)
            
            # Forward pass
            logits = model(x_img, x_hand)
            loss = criterion(logits, y)
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.sigmoid(logits).cpu().numpy()
            labels = y.cpu().numpy()
            
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.flatten())
            all_pids.extend(batch['patient_id'])
    
    avg_loss = total_loss / len(loader)
    auc = roc_auc_score(all_labels, all_preds)
    
    # Find optimal threshold using Youden's J statistic
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Binary predictions at optimal threshold
    binary_preds = (np.array(all_preds) >= optimal_threshold).astype(int)
    acc = accuracy_score(all_labels, binary_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, binary_preds, average='binary', zero_division=0
    )
    
    metrics = {
        'loss': avg_loss,
        'auc': auc,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'optimal_threshold': optimal_threshold
    }
    
    return metrics, all_preds, all_labels, all_pids


def train_model(model, train_loader, val_loader, epochs=50, lr=1.7345566642360933e-05, 
                weight_decay=0.0002669866674274458, grad_clip=0.5, 
                use_class_weights=True, device='cuda'):
    """Complete training loop"""

    if use_class_weights:
        all_labels =[]
        for batch in train_loader:
            all_labels.extend(batch['y'].numpy())
        all_labels = np.array(all_labels)

        #Calculate weights
        n_samples = len(all_labels)
        n_pos = all_labels.sum()
        n_neg = n_samples - n_pos

        pos_weight = torch.tensor([n_neg / n_pos], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using class weights: pos_weight={pos_weight.item():.4f}")
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # More conservative scheduler to prevent LR collapse
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-7
    )
    patience = 10
    no_improve_count = 0
    best_auc = 0
    best_epoch = 0
    history = {
        'train_loss': [], 'train_auc': [],
        'val_loss': [], 'val_auc': []
    }
    
    print("\n" + "="*80)
    print("TRAINING STARTED")
    print("="*80)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, device, grad_clip)
        
        # Validate
        val_metrics, _, _, _ = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_metrics['auc'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_auc'].append(val_metrics['auc'])
        
        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val AUC: {val_metrics['auc']:.4f}")
        print(f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        print(f"Learning Rate: {current_lr:.6e}")
        print(f"Optimal Threshold: {val_metrics['optimal_threshold']:.4f}")
        
        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'best_fusion_mlp.pth')
            print(f"New best model saved! (AUC: {best_auc:.4f})")
        else:
            no_improve_count += 1
        
        if no_improve_count >= patience:
            print("Early stopping triggered.")
            break
    
    print("\n" + "="*80)
    print(f"TRAINING COMPLETED - Best AUC: {best_auc:.4f} at epoch {best_epoch}")
    print("="*80)
    
    return history

def plot_training_history(history, save_path='training_history.png'):
    """Plot training history with loss and metrics"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: AUC
    axes[0, 1].plot(epochs, history['train_auc'], 'b-', label='Train AUC', linewidth=2)
    axes[0, 1].plot(epochs, history['val_auc'], 'r-', label='Val AUC', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('AUC', fontsize=12)
    axes[0, 1].set_title('Training and Validation AUC', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Plot 3: Overfitting Gap (Train - Val)
    auc_gap = [t - v for t, v in zip(history['train_auc'], history['val_auc'])]
    axes[1, 0].plot(epochs, auc_gap, 'g-', linewidth=2)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('AUC Gap (Train - Val)', fontsize=12)
    axes[1, 0].set_title('Overfitting Monitor', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].fill_between(epochs, 0, auc_gap, alpha=0.3, color='green')
    
    # Plot 4: Loss comparison
    axes[1, 1].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2, alpha=0.7)
    axes[1, 1].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2, alpha=0.7)
    
    # Add best epoch marker
    best_epoch_idx = np.argmax(history['val_auc'])
    axes[1, 1].axvline(x=best_epoch_idx + 1, color='gold', linestyle='--', 
                       linewidth=2, label=f'Best Epoch ({best_epoch_idx + 1})')
    axes[1, 1].scatter([best_epoch_idx + 1], [history['val_loss'][best_epoch_idx]], 
                       color='red', s=100, zorder=5, marker='*')
    
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Loss', fontsize=12)
    axes[1, 1].set_title('Loss with Best Epoch Marker', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Training history plot saved to: {save_path}")
    plt.show()
    
    return fig


def plot_roc_curve(labels, predictions, save_path='roc_curve.png'):
    """Plot ROC curve with optimal threshold"""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    # Mark optimal point
    plt.scatter(optimal_fpr, optimal_tpr, color='red', s=200, zorder=5, 
                marker='*', edgecolors='black', linewidths=2,
                label=f'Optimal Threshold = {optimal_threshold:.3f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve with Optimal Threshold', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 ROC curve saved to: {save_path}")
    plt.show()
    
    return plt.gcf()

def compute_feature_stats(handcrafted_dict,patient_ids,feature_names):
    values = {k:[] for k in feature_names}

    for pid in patient_ids:
        for k in feature_names:
            v = handcrafted_dict[pid][k]
            if not np.isnan(v) and not np.isinf(v):
                values[k].append(v)

    stats = {}
    for k, v in values.items():
        v = np.array(v)
        stats[k] = {
            'mean': v.mean(),
            'std': v.std() + 1e-6
        }

    return stats




def train_with_config(config, train_loader, val_loader, device, max_epochs=50):
    """Train model with specific hyperparameter configuration"""
    
    # Get dimensions
    sample = train_loader.dataset[0]
    img_dim = sample['x_img'].shape[0]
    hand_dim = sample['x_hand'].shape[0]
    
    # Create model
    model = FusionMLPV2(img_dim, hand_dim, config['model']).to(device)
    
    # Loss function with optional class weights
    if config['use_class_weights']:
        # Compute class weights
        labels = [train_loader.dataset[i]['y'].item() for i in range(len(train_loader.dataset))]
        pos_weight = torch.tensor([labels.count(0) / max(labels.count(1), 1)]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config['lr'], 
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['lr'], 
            weight_decay=config['weight_decay']
        )
    else:  # sgd
        optimizer = optim.SGD(
            model.parameters(), 
            lr=config['lr'], 
            momentum=0.9,
            weight_decay=config['weight_decay']
        )
    
    # Scheduler
    if config['scheduler'] == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
    elif config['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs
        )
    else:
        scheduler = None
    
    # Compute normalization stats
    from collections import defaultdict
    all_x_img, all_x_hand = [], []
    for batch in train_loader:
        all_x_img.append(batch['x_img'])
        all_x_hand.append(batch['x_hand'])
    all_x_img = torch.cat(all_x_img, dim=0)
    all_x_hand = torch.cat(all_x_hand, dim=0)
    
    norm_stats = {
        'img_mean': all_x_img.mean(dim=0).to(device),
        'img_std': (all_x_img.std(dim=0) + 1e-8).to(device),
        'hand_mean': all_x_hand.mean(dim=0).to(device),
        'hand_std': (all_x_hand.std(dim=0) + 1e-8).to(device)
    }
    
    best_val_auc = 0
    patience_counter = 0
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        for batch in train_loader:
            x_img = batch['x_img'].to(device)
            x_hand = batch['x_hand'].to(device)
            y = batch['y'].to(device)
            
            x_img = (x_img - norm_stats['img_mean']) / norm_stats['img_std']
            x_hand = (x_hand - norm_stats['hand_mean']) / norm_stats['hand_std']
            
            optimizer.zero_grad()
            logits = model(x_img, x_hand)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
        
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                x_img = batch['x_img'].to(device)
                x_hand = batch['x_hand'].to(device)
                y = batch['y'].to(device)
                
                x_img = (x_img - norm_stats['img_mean']) / norm_stats['img_std']
                x_hand = (x_hand - norm_stats['hand_mean']) / norm_stats['hand_std']
                
                logits = model(x_img, x_hand)
                preds = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(preds.flatten())
                val_labels.extend(y.cpu().numpy().flatten())
        
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(val_labels, val_preds)
        
        if scheduler is not None:
            if config['scheduler'] == 'plateau':
                scheduler.step(val_auc)
            else:
                scheduler.step()
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['early_stop_patience']:
                break
    
    return best_val_auc


def grid_search(train_loader, val_loader, device):
    """Perform grid search over hyperparameters"""
    
    # Define search space
    search_space = {
        'lr': [1e-5, 5e-5, 1e-4, 5e-4],
        'dropout': [0.3, 0.5, 0.7],
        'weight_decay': [1e-5, 1e-4, 1e-3],
        'optimizer': ['adam', 'adamw'],
        'scheduler': ['plateau', 'cosine', None],
        'use_class_weights': [True, False],
        'img_hidden': [128, 256, 512],
        'fusion_hidden': [
            [128, 64],
            [256, 128, 64],
            [512, 256, 128]
        ],
        'activation': ['relu', 'leaky_relu'],
        'use_layer_norm': [True, False]
    }
    
    results = []
    
    # For quick testing, try random search with N configs
    import random
    n_trials = 20
    
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER SEARCH - Testing {n_trials} configurations")
    print(f"{'='*80}\n")
    
    for trial in range(n_trials):
        config = {
            'lr': random.choice(search_space['lr']),
            'dropout': random.choice(search_space['dropout']),
            'weight_decay': random.choice(search_space['weight_decay']),
            'optimizer': random.choice(search_space['optimizer']),
            'scheduler': random.choice(search_space['scheduler']),
            'use_class_weights': random.choice(search_space['use_class_weights']),
            'grad_clip': 1.0,
            'early_stop_patience': 10,
            'model': {
                'img_hidden': random.choice(search_space['img_hidden']),
                'hand_hidden': 64,
                'fusion_hidden': random.choice(search_space['fusion_hidden']),
                'dropout': random.choice(search_space['dropout']),
                'activation': random.choice(search_space['activation']),
                'use_layer_norm': random.choice(search_space['use_layer_norm'])
            }
        }
        
        print(f"\nTrial {trial+1}/{n_trials}")
        print(f"Config: {json.dumps(config, indent=2)}")
        
        try:
            val_auc = train_with_config(config, train_loader, val_loader, device, max_epochs=50)
            print(f"✓ Validation AUC: {val_auc:.4f}")
            
            results.append({
                'trial': trial + 1,
                'config': config,
                'val_auc': val_auc
            })
        except Exception as e:
            print(f"✗ Failed: {e}")
            continue
    
    # Sort by validation AUC
    results.sort(key=lambda x: x['val_auc'], reverse=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df = pd.DataFrame([
        {
            'trial': r['trial'],
            'val_auc': r['val_auc'],
            **{k: str(v) for k, v in r['config'].items()}
        }
        for r in results
    ])
    results_df.to_csv(f'hyperparameter_search_{timestamp}.csv', index=False)
    
    print(f"\n{'='*80}")
    print("TOP 5 CONFIGURATIONS")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. Validation AUC: {result['val_auc']:.4f}")
        print(f"   Config: {json.dumps(result['config'], indent=6)}\n")
    
    return results

# ===============================================================
# XGBOOST UTILITY
# ===============================================================



class IPFDataLoader:
    """Carica e prepara i dati dal CSV"""

    def __init__(self, csv_path: str, csv_features_path : str,npy_dir:str):
        """
        Args:
            csv_path: Path al CSV con [Patient, Weeks, FVC, Slice_files, FVC slope, FVC intercept0]
            npy_dir: Directory path di file npy images

        """
        self.df = pd.read_csv(csv_path)
        self.npy_dir = npy_dir
        self.df_features = pd.read_csv(csv_features_path)

        print(f"✅ Loaded {len(self.df)} records from CSV")
        print(f"✅ Unique patients in CSV: {self.df['Patient'].nunique()}")
        print(f"✅ NPY directory: {npy_dir}")
        print(f"📋 Columns: {self.df.columns.tolist()}")

        # Verify demographic columns exist
        required_demo_cols = ['Age', 'Sex', 'SmokingStatus']
        missing_cols = [col for col in required_demo_cols if col not in self.df.columns]
        if missing_cols:
            print(f"⚠️  WARNING: Missing demographic columns: {missing_cols}")
        else:
            print(f"✅ All demographic columns present: {required_demo_cols}")

        # Verifica che i pazienti nel CSV abbiano cartelle NPY
        self._verify_npy_availability()

    def _verify_npy_availability(self):
        """Verifica che ogni paziente nel CSV abbia una cartella con file .npy"""
        patients_csv = set(self.df['Patient'].unique())
        patients_npy = set([d for d in os.listdir(self.npy_dir)
                           if os.path.isdir(os.path.join(self.npy_dir, d))])

        missing = patients_csv - patients_npy
        extra = patients_npy - patients_csv

        if missing:
            print(f"⚠️  {len(missing)} pazienti nel CSV senza cartella NPY: {list(missing)[:5]}...")
        if extra:
            print(f"ℹ️  {len(extra)} cartelle NPY non nel CSV (verranno ignorate)")

        available = patients_csv & patients_npy
        print(f"✅ {len(available)} pazienti con dati completi (CSV + NPY)")



    def get_patient_data(self) -> Dict[str, Dict]:
        """
        Organizza i dati per paziente

        Returns:
            {patient_id: {
                'slope': float,
                'intercept': float,
                'slices': [list_of_slice_paths],
                'weeks': [list_of_weeks],
                'fvc_values': [list_of_fvc]
            }}
        """
        patient_data = {}
        features_dict = {}

        for patient_id in self.df['Patient'].unique():
            patient_df = self.df[self.df['Patient'] == patient_id].sort_values('Weeks')
            patient_df_features = self.df_features[self.df_features['Patient'] == patient_id]

            # Slope e Intercept FVC (costanti per paziente)
            slope = patient_df['fvc_slope'].iloc[0]
            intercept = patient_df['fvc_intercept0'].iloc[0]

            #Get tabular data
            age = patient_df['Age'].iloc[0]
            sex = patient_df['Sex'].iloc[0]
            smoking_status = patient_df['SmokingStatus'].iloc[0]

            # Baseline (prima misurazione) -> Firstly use intercept a t=0
            #baseline_week = patient_df['Week'].iloc[0]
            #baseline_fvc = patient_df['FVC'].iloc[0]

            patient_npy_folder = os.path.join(self.npy_dir, patient_id)

            if not os.path.exists(patient_npy_folder):
                print(f"⚠️  Cartella NPY non trovata per {patient_id}, paziente saltato")
                continue

            # Ottieni tutti i file .npy ordinati
            npy_files = sorted(glob.glob(os.path.join(patient_npy_folder, "*.npy")))

            if not npy_files:
                print(f"⚠️  Nessun file NPY trovato per {patient_id}, paziente saltato")
                continue


            # Timeline completa
            weeks = patient_df['Weeks'].tolist()
            fvc_values = patient_df['FVC'].tolist()

            #Features
            ApproxVol_30_60 = float(patient_df_features['ApproxVol_30_60'].iloc[0])
            Avg_NumTissuePixel_30_60 = float(patient_df_features['Avg_NumTissuePixel_30_60'].iloc[0])
            Avg_Tissue_30_60 = float(patient_df_features['Avg_Tissue_30_60'].iloc[0])
            Avg_Tissue_thickness_30_60 = float(patient_df_features['Avg_Tissue_thickness_30_60'].iloc[0])
            Avg_TissueByTotal_30_60 = float(patient_df_features['Avg_TissueByTotal_30_60'].iloc[0])
            Avg_TissueByLung_30_60 = float(patient_df_features['Avg_TissueByLung_30_60'].iloc[0])
            Mean_30_60 = float(patient_df_features['Mean_30_60'].iloc[0])
            Skew_30_60 = float(patient_df_features['Skew_30_60'].iloc[0])
            Kurtosis_30_60 = float(patient_df_features['Kurtosis_30_60'].iloc[0])

            # ========================================
            # ENCODE CATEGORICAL FEATURES
            # ========================================

            # Sex: Male=1, Female=0
            sex_encoded = 1.0 if sex == 'Male' else 0.0

            # SmokingStatus: Ex-smoker=1, Never smoked=0, Currently smokes=2
            # (Adjust based on your actual categories)
            smoking_map = {
                'Ex-smoker': 1.0,
                'Never smoked': 0.0,
                'Currently smokes': 2.0,
            }
            smoking_encoded = smoking_map.get(smoking_status, 0.0)  # Default to 0 if unknown

            # Age: already numeric, just cast to float
            age_float = float(age)


            features_dict[patient_id] = {
                'approx_vol': ApproxVol_30_60,
                'avg_num_tissue_pixel' : Avg_NumTissuePixel_30_60,
                'avg_tissue': Avg_Tissue_30_60,
                'avg_tissue_thickness':Avg_Tissue_thickness_30_60,
                'avg_tissue_by_total': Avg_TissueByTotal_30_60,
                'avg_tissue_by_lung': Avg_TissueByLung_30_60,
                'mean' : Mean_30_60,
                'skew': Skew_30_60,
                'kurtosis': Kurtosis_30_60,
                # Demographic features (3) - NEWLY ADDED
                'age': age_float,
                'sex': sex_encoded,
                'smoking_status': smoking_encoded,
            }


            patient_data[patient_id] = {
                'slope': slope,
                'intercept': intercept,
                'slices': npy_files,
                'n_slices':len(npy_files),
                'weeks': weeks,
                'fvc_values': fvc_values
            }

        print(f"\n📊 Patient data prepared for {len(patient_data)} patients")

        # Statistiche sulle slice
        slice_counts = [data['n_slices'] for data in patient_data.values()]
        print(f"   Min slices: {min(slice_counts)}")
        print(f"   Max slices: {max(slice_counts)}")
        print(f"   Avg slices: {np.mean(slice_counts):.2f}")



        return patient_data,features_dict

    def split_patients(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split a livello di PAZIENTE per evitare data leakage

        Returns:
            train_patients, val_patients, test_patients (liste di patient IDs)
        """
        # Solo pazienti che hanno sia dati CSV che cartella NPY
        patient_npy_folders = set([d for d in os.listdir(self.npy_dir)
                                   if os.path.isdir(os.path.join(self.npy_dir, d))])
        csv_patients = set(self.df['Patient'].unique())

        valid_patients = np.array(list(csv_patients & patient_npy_folders))

        print(f"\n🔄 Splitting {len(valid_patients)} valid patients...")

        # Prima split: Train+Val vs Test
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_val_idx, test_idx = next(splitter.split(valid_patients, groups=valid_patients))

        train_val_patients = valid_patients[train_val_idx]
        test_patients = valid_patients[test_idx]

        # Seconda split: Train vs Val
        val_size_adjusted = val_size / (1 - test_size)
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=random_state)
        train_idx, val_idx = next(splitter.split(train_val_patients, groups=train_val_patients))

        train_patients = train_val_patients[train_idx]
        val_patients = train_val_patients[val_idx]

        print(f"✅ Data Split:")
        print(f"   Train: {len(train_patients)} patients")
        print(f"   Val: {len(val_patients)} patients")
        print(f"   Test: {len(test_patients)} patients")

        return train_patients.tolist(), val_patients.tolist(), test_patients.tolist()

    def __len__(self):
        return len(self.df)

class IPFSliceDataset(Dataset):
    """Dataset che carica singole slice DICOM con slope come label"""

    def __init__(self, patient_list: List[str], patient_data: Dict,features_data:Dict,
                 transform=None, normalize_slope=True, image_size=(224, 224)):
        """
        Args:
            patient_list: Lista di patient IDs da includere (based on split)
            patient_data: Dict da IPFDataLoader.get_patient_data()
            transform: Trasformazioni immagini
            normalize_slope: Se normalizzare gli slope
            image_size: Dimensione target (H, W)
        """
        self.data = []
        self.slopes = []
        self.image_size = image_size
        self.transform = transform
        self._cache = {}

        # Crea lista (patient_id, slice_path, slope)
        for patient_id in patient_list:

            if patient_id not in patient_data:
                continue

            if patient_id not in features_data:
                continue

            #Extracts data from dictionary
            pdata = patient_data[patient_id]
            fdata = features_data[patient_id]
            #Slope
            slope = pdata['slope']

            slices = pdata['slices']
            weeks = pdata['weeks']

            for slice_path in slices:
              self.data.append({
                  'patient_id':patient_id,
                  'slice_path':slice_path,
                  'slope':float(slope),
                  'intercept':float(pdata.get('intercept',0.0)),
                  'features_patient':fdata
              })
              self.slopes.append(float(slope))

        #Mappa : paziente -> indice delle sue slices nel dataset

        self.patient_to_indices = defaultdict(list)
        for idx, item in enumerate(self.data):
          self.patient_to_indices[item['patient_id']].append(idx)
        self.patients = list(self.patient_to_indices.keys())

        print(f"Dataset created with {len(self.data)} slices from {len(patient_list)} patients")

        # Normalizzazione slope -> Try with both
        self.slope_scaler = None
        if normalize_slope and len(self.slopes) > 0:
            self.slope_scaler = StandardScaler()
            slopes_array = np.array(self.slopes).reshape(-1, 1)
            self.slope_scaler.fit(slopes_array)
            print(f"Slope normalization: mean={self.slope_scaler.mean_[0]:.2f}, std={self.slope_scaler.scale_[0]:.2f}")

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        if idx in self._cache:
          return self._cache[idx]

        item = self.data[idx]

        try:
          # Carica NPY (già preprocessata: [224, 224] float32 normalizzata [0,1])
            img = np.load(item['slice_path'])

            # Verifica dimensioni
            if img.shape != (224, 224):
                raise ValueError(f"Unexpected shape: {img.shape}, expected (224, 224)")

            # Converti a 3 canali (RGB) se necessario
            # Se le tue NPY sono già [3, 224, 224], togli questa parte
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=0)  # [3, 224, 224]

            # Applica trasformazioni aggiuntive se presenti
            if self.transform:
                img = self.transform(img)

            # Converti a tensor
            img_tensor = torch.FloatTensor(img)


        except Exception as e:
            print(f"⚠️  Error loading {item['slice_path']}: {e}")
            self._cache[idx] = None
            return None



        # Slope normalizzato
        slope = item['slope']
        if self.slope_scaler:
            slope = self.slope_scaler.transform([[slope]])[0][0]

        return {
            'image': img_tensor,
            'slope': torch.FloatTensor([slope]),
            'patient_id': item['patient_id'],
            'slice_path': item['slice_path'],
            'feature_patient': item['features_patient']
        }

# ============================================================
# UTILITY: COLLATE FUNCTION PER FILTRARE SLICE INVALIDE
# ============================================================

def collate_fn_filter_none(batch):
    """
    Collate function che filtra sample None (slice invalide).
    Utile se alcuni file NPY sono corrotti.
    """
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    return torch.utils.data.dataloader.default_collate(batch)

# ============================================================================
# FIX 1: IMPROVED PREPROCESSING WITH LUNG WINDOWING
# ============================================================================

class LungWindowTransform:
    """
    Apply proper lung windowing to CT images
    This enhances lung tissue and suppresses other structures
    """
    def __init__(self, window_center=-600, window_width=1500):
        """
        Standard lung window:
        - Center: -600 HU (air + lung tissue range)
        - Width: 1500 HU (captures full lung dynamic range)
        """
        self.wc = window_center
        self.ww = window_width
        self.wmin = self.wc - self.ww / 2
        self.wmax = self.wc + self.ww / 2

    def __call__(self, img):
        """
        Args:
            img: numpy array, assumed to be in HU units or normalized
        """
        # If already normalized [0,1], convert back to approximate HU
        if img.max() <= 1.0:
            # Assume original range was [-1000, 500] (typical CT)
            img = img * 1500 - 1000

        # Apply window
        img_windowed = np.clip(img, self.wmin, self.wmax)

        # Normalize to [0, 1]
        img_normalized = (img_windowed - self.wmin) / (self.wmax - self.wmin)

        return img_normalized


def create_lung_mask_attention(img, threshold=-500):
    """
    Create a soft mask emphasizing lung regions
    This can be used as attention guidance during training

    Args:
        img: (H, W) numpy array in HU units
        threshold: HU value to separate lung from other tissue

    Returns:
        mask: (H, W) soft attention mask [0, 1]
    """
    # Lung tissue is typically -900 to -500 HU
    # Soft tissue is > -100 HU

    # Create binary lung mask
    lung_mask = (img > -900) & (img < -500)

    # Dilate slightly to include peri-bronchial regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lung_mask = cv2.dilate(lung_mask.astype(np.uint8), kernel, iterations=2)

    # Create soft mask with Gaussian blur
    soft_mask = cv2.GaussianBlur(lung_mask.astype(float), (15, 15), 0)

    # Normalize
    soft_mask = soft_mask / (soft_mask.max() + 1e-8)

    return soft_mask


# ============================================================================
# FIX 2: ATTENTION-GUIDED LOSS
# ============================================================================

class AttentionGuidedLoss(nn.Module):
    """
    Loss that penalizes attention on non-lung regions
    Encourages CNN to focus on relevant anatomy
    """
    def __init__(self, lung_mask_weight=0.1):
        super().__init__()
        self.lung_mask_weight = lung_mask_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets, attention_maps, lung_masks):
        """
        Args:
            predictions: (batch,) predicted slopes
            targets: (batch,) true slopes
            attention_maps: (batch, H, W) attention from Grad-CAM or attention layer
            lung_masks: (batch, H, W) binary lung masks

        Returns:
            total_loss
        """
        # Standard prediction loss
        prediction_loss = self.mse_loss(predictions, targets)

        # Attention regularization: penalize attention outside lungs
        # attention_maps should sum to 1 per image
        attention_maps = attention_maps / (attention_maps.sum(dim=(1,2), keepdim=True) + 1e-8)

        # Compute attention inside vs outside lungs
        attention_inside = (attention_maps * lung_masks).sum(dim=(1,2))
        attention_outside = (attention_maps * (1 - lung_masks)).sum(dim=(1,2))

        # Penalize attention outside lungs
        attention_loss = attention_outside.mean()

        # Total loss
        total_loss = prediction_loss + self.lung_mask_weight * attention_loss

        return total_loss

# ============================================================================
# FIX 3: SPATIAL ATTENTION MODULE
# ============================================================================

class SpatialAttentionModule(nn.Module):
    """
    Explicit spatial attention that learns to focus on informative regions
    This is better than just relying on the CNN to learn implicitly
    """
    def __init__(self, in_channels):
        super().__init__()

        # Spatial attention pathway
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()  # Output attention map [0, 1]
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature map
        Returns:
            attended_x: (B, C, H, W) spatially weighted features
            attention_map: (B, 1, H, W) for visualization
        """
        attention_map = self.spatial_attention(x)  # (B, 1, H, W)
        attended_x = x * attention_map  # Element-wise multiplication

        return attended_x, attention_map

class FeatureExtractorCNN(nn.Module):
    """
    CNN with explicit spatial attention
    Forces model to learn WHERE to look
    """
    def __init__(self, backbone_name='efficientnet_b0', pretrained=True):
        super().__init__()

        # Backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            features_only=True  # Return intermediate features
        )

        # Get feature dimensions
        # For EfficientNet-B0: [16, 24, 40, 112, 1280]
        feat_dims = self.backbone.feature_info.channels()

        # Add spatial attention to last feature map
        self.spatial_attention = SpatialAttentionModule(feat_dims[-1])

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        


    def forward(self, x, return_attention=False):
        """
        Args:
            x: (B, 3, H, W) input images
            return_attention: if True, return attention maps for visualization
        """
        # Extract features
        features = self.backbone(x)
        last_feature_map = features[-1]  # (B, C, h, w)

        # Apply spatial attention
        attended_features, attention_map = self.spatial_attention(last_feature_map)

        # Global pooling
        pooled = self.global_pool(attended_features).flatten(1)  # (B, C)

        # Prediction
        output = self.head(pooled).squeeze(-1)  # (B,)

        if return_attention:
            return output, attention_map
        return output

class ImprovedSliceLevelCNN(nn.Module):
    """
    CNN with explicit spatial attention
    Forces model to learn WHERE to look
    """
    def __init__(self, backbone_name='efficientnet_b0', pretrained=True):
        super().__init__()

        # Backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            features_only=True  # Return intermediate features
        )

        # Get feature dimensions
        # For EfficientNet-B0: [16, 24, 40, 112, 1280]
        feat_dims = self.backbone.feature_info.channels()

        # Add spatial attention to last feature map
        self.spatial_attention = SpatialAttentionModule(feat_dims[-1])

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Prediction head
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dims[-1], 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
    def extract_features(self,x,return_attention=False):
        
        features= self.backbone(x)
        last_feature_map = features[-1]
        
        attented_features, attention_map = self.spatial_attention(last_feature_map)
        
        pooled = self.global_pool(attented_features).flatten(1) #(B,1280)
        
        if return_attention:
            return pooled,attention_map
        
        return pooled

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (B, 3, H, W) input images
            return_attention: if True, return attention maps for visualization
        """
        # Extract features
        features = self.backbone(x)
        last_feature_map = features[-1]  # (B, C, h, w)

        # Apply spatial attention
        attended_features, attention_map = self.spatial_attention(last_feature_map)

        # Global pooling
        pooled = self.global_pool(attended_features).flatten(1)  # (B, C)

        # Prediction
        output = self.head(pooled).squeeze(-1)  # (B,)

        if return_attention:
            return output, attention_map
        return output

class ImprovedSlopeTrainer:
    """
    Trainer with attention-guided loss and better monitoring
    """
    def __init__(self, model, device='cuda', use_attention_loss=False):
        self.model = model.to(device)
        self.device = device
        self.use_attention_loss = use_attention_loss

        # Losses
        self.criterion = nn.MSELoss()
        if use_attention_loss:
            self.attention_criterion = AttentionGuidedLoss(lung_mask_weight=0.1)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-5
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=5,
            factor=0.5
        )

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    def train_epoch(self, train_loader, lung_masks=None):

        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in train_loader:

            images = batch['images'].to(self.device)
            lengths = batch['lengths'].tolist()
            slopes = batch['slopes'].to(self.device)

            # Forward
            if self.use_attention_loss:
                preds_per_slice, attention_maps = self.model(images, return_attention=True)
            else:
                preds_per_slice = self.model(images)

            preds_per_slice = preds_per_slice.view(-1)

            # Aggregate per-patient
            pred_blocks = torch.split(preds_per_slice, lengths)
            slope_blocks = torch.split(slopes, lengths)

            patient_preds = torch.stack([blk.mean() for blk in pred_blocks])
            patient_slopes = torch.stack([blk[0] for blk in slope_blocks])

            # Loss
            loss = self.criterion(patient_preds, patient_slopes)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    # =========================================================================
    # VALIDATION LOOP — MATCHES TRAIN EPOCH
    # =========================================================================
    def validate(self, val_loader):

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:

                images = batch['images'].to(self.device)
                lengths = batch['lengths'].tolist()
                slopes = batch['slopes'].to(self.device)

                preds_per_slice = self.model(images).view(-1)

                # SAME AGGREGATION AS TRAINING
                pred_blocks = torch.split(preds_per_slice, lengths)
                slope_blocks = torch.split(slopes, lengths)

                patient_preds = torch.stack([blk.mean() for blk in pred_blocks])
                patient_slopes = torch.stack([blk[0] for blk in slope_blocks])

                loss = self.criterion(patient_preds, patient_slopes)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    # =========================================================================
    # PREDICTION PER PATIENT — FIXED
    # =========================================================================
    def predict_per_patient(self, loader):

        self.model.eval()
        results = []

        with torch.no_grad():
            for batch in loader:

                images = batch['images'].to(self.device)
                lengths = batch['lengths'].tolist()
                slopes = batch['slopes']
                patient_ids = batch['patient_ids']
                features = batch['features']

                preds_per_slice = self.model(images).view(-1)
                pred_blocks = torch.split(preds_per_slice, lengths)

                # IMPORTANT — reset slice pointer before each batch
                start_slice = 0

                # Iterate patients in batch
                for i, pid in enumerate(patient_ids):

                    slope_pred = pred_blocks[i].mean().item()
                    true_slope = slopes[start_slice].item()

                    # Build row
                    row = {
                        "patient_id": pid,
                        "slope_cnn_mean": slope_pred,
                        "true_slope": true_slope
                    }

                    # add features (from first slice of this patient)
                    for fname, fvals in features.items():

                        row[fname] = float(fvals[start_slice])

                    results.append(row)

                    # move slice pointer
                    start_slice += lengths[i]

        return results

# =============================================================================
# FLEXIBLE SLOPE CORRECTOR (HANDLES VARIABLE FEATURES)
# =============================================================================

class SlopeCorrector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.10),   # ↓ molto più basso

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.05),   # ↓ molto più basso

            nn.Linear(16, 1)
        )

    def forward(self, x):
        base = x[:, 0:1]      # slope_cnn_mean
        corr = self.mlp(x)
        return base + corr

# =============================================================================
# FEATURE BUILDER (CONSTRUCTS FEATURE VECTORS FOR EACH VARIANT)
# =============================================================================

class FeatureBuilder:
    """
    Builds feature vectors for different model variants
    """

    @staticmethod
    def get_cnn_only(slope_cnn):
        """
        Variant 1: CNN-only (no corrector needed)
        Returns: slope_cnn directly
        """
        return None  # Signal to use CNN prediction directly

    @staticmethod
    def get_demographics_only(slope_cnn, features_dict):
        """
        Variant 2: CNN + Demographics (3 features)
        Returns: [slope_cnn, age, sex, smoking_status]
        """
        return np.array([
            slope_cnn,
            features_dict['age'],
            features_dict['sex'],
            features_dict['smoking_status'],
        ], dtype=float)

    @staticmethod
    def get_handcrafted_only(slope_cnn, features_dict):
        """
        Variant 3: CNN + Handcrafted (9 features)
        Returns: [slope_cnn, approx_vol, ..., kurtosis]
        """
        return np.array([
            slope_cnn,
            features_dict['approx_vol'],
            features_dict['avg_num_tissue_pixel'],
            features_dict['avg_tissue'],
            features_dict['avg_tissue_thickness'],
            features_dict['avg_tissue_by_total'],
            features_dict['avg_tissue_by_lung'],
            features_dict['mean'],
            features_dict['skew'],
            features_dict['kurtosis'],
        ], dtype=float)

    @staticmethod
    def get_full_features(slope_cnn, features_dict):
        """
        Variant 4: CNN + Demographics + Handcrafted (12 features)
        Returns: [slope_cnn, demographics..., handcrafted...]
        """
        return np.array([
            slope_cnn,
            # Demographics
            features_dict['age'],
            features_dict['sex'],
            features_dict['smoking_status'],
            # Handcrafted
            features_dict['approx_vol'],
            features_dict['avg_num_tissue_pixel'],
            features_dict['avg_tissue'],
            features_dict['avg_tissue_thickness'],
            features_dict['avg_tissue_by_total'],
            features_dict['avg_tissue_by_lung'],
            features_dict['mean'],
            features_dict['skew'],
            features_dict['kurtosis'],
        ], dtype=float)


# =============================================================================
# MULTI-MODEL EVALUATOR
# =============================================================================

class MultiModelEvaluator:
    """
    Evaluates all 4 model variants on test set
    """

    def __init__(self, cnn_model, correctors_dict, scalers_dict,
                 patient_data, features_data, device='cuda'):
        """
        Args:
            cnn_model: Base CNN model
            correctors_dict: {
                'demographics': (corrector_model, scaler),
                'handcrafted': (corrector_model, scaler),
                'full': (corrector_model, scaler)
            }
            patient_data: Dict with slope, intercept, weeks, fvc_values
            features_data: Dict with all features
            device: cuda/cpu
        """
        self.cnn = cnn_model.to(device).eval()
        self.correctors = correctors_dict
        self.patient_data = patient_data
        self.features_data = features_data
        self.device = device

    def predict_slope(self, patient_id, test_ds, variant='cnn_only'):
      # -----------------------------
      # 1. CNN prediction (unchanged)
      # -----------------------------
      patient_indices = test_ds.patient_to_indices[patient_id]
      slopes_predicted = []

      with torch.no_grad():
          for idx in patient_indices:
              sample = test_ds[idx]
              if sample is None:
                  continue
              img = sample['image'].unsqueeze(0).to(self.device)
              slope_raw = self.cnn(img).cpu().item()
              slopes_predicted.append(slope_raw)

      if not slopes_predicted:
          return 0.0

      slope_cnn_mean = np.mean(slopes_predicted)

      # CNN-only → just denormalize and return
      if variant == 'cnn_only':
          if test_ds.slope_scaler:
              return test_ds.slope_scaler.inverse_transform([[slope_cnn_mean]])[0][0]
          return slope_cnn_mean

      # ---------------------------------------------
      # 2. Corrector-based variants (2, 3, 4)
      # ---------------------------------------------
      corrector, scaler, cols = self.correctors[variant]
      features_dict = self.features_data[patient_id]

      # === Variant 4: full features — MUST use training column order ===
      if variant == 'full':
          raw_dict = {
              'slope_cnn_mean': slope_cnn_mean,
              **features_dict
          }
          # Build vector in EXACT training order
          feature_vec = np.array([raw_dict[c] for c in cols], dtype=float)

      # === Variant 2: demographics only ===
      elif variant == 'demographics':
          feature_vec = np.array([
              slope_cnn_mean,
              features_dict['age'],
              features_dict['sex'],
              features_dict['smoking_status']
          ], dtype=float)

      # === Variant 3: handcrafted only ===
      elif variant == 'handcrafted':
          feature_vec = np.array([
              slope_cnn_mean,
              features_dict['approx_vol'],
              features_dict['avg_num_tissue_pixel'],
              features_dict['avg_tissue'],
              features_dict['avg_tissue_thickness'],
              features_dict['avg_tissue_by_total'],
              features_dict['avg_tissue_by_lung'],
              features_dict['mean'],
              features_dict['skew'],
              features_dict['kurtosis'],
          ], dtype=float)

      # Clean NaN
      feature_vec = np.nan_to_num(feature_vec, nan=0.0)

      # Scale
      feature_scaled = scaler.transform([feature_vec])

      feature_tensor = torch.tensor(feature_scaled, dtype=torch.float32).to(self.device)

      # Predict
      with torch.no_grad():
          slope_pred_norm = corrector(feature_tensor).cpu().item()

      # Denormalize
      if test_ds.slope_scaler:
          return test_ds.slope_scaler.inverse_transform([[slope_pred_norm]])[0][0]
      return slope_pred_norm


    def predict_fvc(self, patient_id, predicted_slope):
        """
        Predict FVC trajectory given slope

        Returns:
            weeks, predicted_fvc
        """
        pdata = self.patient_data[patient_id]
        weeks = np.array(pdata['weeks'])
        intercept = pdata['intercept']

        predicted_fvc = intercept + predicted_slope * weeks

        return weeks, predicted_fvc

    def evaluate_all_variants(self, test_ds, test_patients):
        """
        Evaluate all 4 variants on test set

        Returns:
            results_dict: {variant_name: {metrics, predictions_df}}
        """
        results = {}

        variants = ['cnn_only', 'demographics', 'handcrafted', 'full']
        variant_names = {
            'cnn_only': 'CNN-only',
            'demographics': 'CNN + Demographics',
            'handcrafted': 'CNN + Handcrafted',
            'full': 'CNN + Demographics + Handcrafted'
        }

        for variant in variants:
            print(f"\n{'='*80}")
            print(f"Evaluating: {variant_names[variant]}")
            print(f"{'='*80}")

            predictions_list = []

            for patient_id in tqdm(test_patients, desc=f"Testing {variant}"):
                if patient_id not in self.patient_data:
                    continue

                # Predict slope
                pred_slope = self.predict_slope(patient_id, test_ds, variant=variant)
                true_slope = self.patient_data[patient_id]['slope']

                # Predict FVC trajectory
                weeks, pred_fvc = self.predict_fvc(patient_id, pred_slope)
                true_fvc = np.array(self.patient_data[patient_id]['fvc_values'])

                # Compute errors
                slope_error = abs(pred_slope - true_slope)
                fvc_errors = np.abs(pred_fvc - true_fvc)
                fvc_mae = fvc_errors.mean()

                predictions_list.append({
                    'patient_id': patient_id,
                    'pred_slope': pred_slope,
                    'true_slope': true_slope,
                    'slope_error': slope_error,
                    'fvc_mae': fvc_mae,
                    'n_timepoints': len(weeks)
                })

            # Aggregate metrics
            df = pd.DataFrame(predictions_list)

            metrics = {
                'variant': variant_names[variant],
                'slope_mae': df['slope_error'].mean(),
                'slope_rmse': np.sqrt((df['slope_error']**2).mean()),
                'fvc_mae': df['fvc_mae'].mean(),
                'fvc_rmse': np.sqrt((df['fvc_mae']**2).mean()),
                'n_patients': len(df)
            }

            print(f"\n📊 Results:")
            print(f"   Slope MAE: {metrics['slope_mae']:.4f} ml/week")
            print(f"   FVC MAE:   {metrics['fvc_mae']:.2f} ml")

            results[variant] = {
                'metrics': metrics,
                'predictions_df': df
            }

        return results


# =============================================================================
# VISUALIZATION SUITE
# =============================================================================

class ComparisonVisualizer:
    """
    Creates comprehensive comparison visualizations
    """

    @staticmethod
    def plot_performance_comparison(results_dict, save_path=None):
        """
        Bar charts comparing all 4 variants
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        variants = ['cnn_only', 'demographics', 'handcrafted', 'full']
        variant_labels = ['CNN\nonly', 'CNN +\nDemo', 'CNN +\nHandcraft', 'CNN +\nBoth']
        colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']

        # Slope MAE
        slope_maes = [results_dict[v]['metrics']['slope_mae'] for v in variants]
        axes[0].bar(variant_labels, slope_maes, color=colors, alpha=0.8, edgecolor='black')
        axes[0].set_ylabel('Slope MAE (ml/week)', fontsize=12)
        axes[0].set_title('Slope Prediction Error', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        # Add values on bars
        for i, v in enumerate(slope_maes):
            axes[0].text(i, v + 0.1, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

        # FVC MAE
        fvc_maes = [results_dict[v]['metrics']['fvc_mae'] for v in variants]
        axes[1].bar(variant_labels, fvc_maes, color=colors, alpha=0.8, edgecolor='black')
        axes[1].set_ylabel('FVC MAE (ml)', fontsize=12)
        axes[1].set_title('FVC Prediction Error', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        # Add values on bars
        for i, v in enumerate(fvc_maes):
            axes[1].text(i, v + 3, f'{v:.1f}', ha='center', fontsize=10, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Print comparison table
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON TABLE")
        print("="*80)

        comparison_df = pd.DataFrame({
            'Model': [results_dict[v]['metrics']['variant'] for v in variants],
            'Slope MAE': slope_maes,
            'FVC MAE': fvc_maes,
        })

        print(comparison_df.to_string(index=False))

        # Compute improvements
        baseline = fvc_maes[0]  # CNN-only
        print(f"\n📈 Improvements vs CNN-only baseline ({baseline:.2f} ml):")
        for i, variant in enumerate(variants[1:], 1):
            diff = baseline - fvc_maes[i]
            pct = 100 * diff / baseline
            symbol = "✓" if diff > 0 else "✗"
            print(f"   {symbol} {results_dict[variants[i]]['metrics']['variant']}: "
                  f"{diff:+.2f} ml ({pct:+.1f}%)")

    @staticmethod
    def plot_patient_trajectories(evaluator, test_ds, patient_ids, save_path=None):
        """
        Plot FVC trajectories for sample patients with all 4 model predictions
        """
        n_patients = len(patient_ids)
        fig, axes = plt.subplots(n_patients, 1, figsize=(12, 5*n_patients))

        if n_patients == 1:
            axes = [axes]

        variants = ['cnn_only', 'demographics', 'handcrafted', 'full']
        colors = {'cnn_only': '#3498db', 'demographics': '#e74c3c',
                  'handcrafted': '#f39c12', 'full': '#2ecc71'}
        labels = {'cnn_only': 'CNN-only', 'demographics': 'CNN + Demo',
                  'handcrafted': 'CNN + Handcraft', 'full': 'CNN + Both'}

        for idx, patient_id in enumerate(patient_ids):
            ax = axes[idx]

            # Get true trajectory
            pdata = evaluator.patient_data[patient_id]
            true_weeks = np.array(pdata['weeks'])
            true_fvc = np.array(pdata['fvc_values'])
            true_slope = pdata['slope']

            # Plot true FVC
            ax.scatter(true_weeks, true_fvc, s=100, color='black',
                      marker='o', label='True FVC', zorder=5)
            ax.plot(true_weeks, true_fvc, 'k--', alpha=0.3, linewidth=2)

            # Plot predictions from each variant
            for variant in variants:
                pred_slope = evaluator.predict_slope(patient_id, test_ds, variant=variant)
                weeks_pred, fvc_pred = evaluator.predict_fvc(patient_id, pred_slope)

                ax.plot(weeks_pred, fvc_pred, '-', linewidth=2.5,
                       color=colors[variant], label=labels[variant], alpha=0.8)

            # Formatting
            ax.set_xlabel('Weeks from Baseline', fontsize=12)
            ax.set_ylabel('FVC (ml)', fontsize=12)
            ax.set_title(f'Patient {patient_id}\n'
                        f'True Slope: {true_slope:.2f} ml/week',
                        fontsize=13, fontweight='bold')
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_error_distributions(results_dict, save_path=None):
        """
        Violin plots showing error distributions for each variant
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        variants = ['cnn_only', 'demographics', 'handcrafted', 'full']

        # Prepare data for violin plots
        slope_data = []
        fvc_data = []
        variant_labels = []

        for variant in variants:
            df = results_dict[variant]['predictions_df']
            slope_data.append(df['slope_error'].values)
            fvc_data.append(df['fvc_mae'].values)
            variant_name = results_dict[variant]['metrics']['variant']
            variant_labels.append(variant_name.replace(' ', '\n'))

        # Slope errors
        parts1 = axes[0].violinplot(slope_data, positions=range(len(variants)),
                                     showmeans=True, showmedians=True)
        axes[0].set_xticks(range(len(variants)))
        axes[0].set_xticklabels(variant_labels, fontsize=10)
        axes[0].set_ylabel('Slope Error (ml/week)', fontsize=12)
        axes[0].set_title('Slope Error Distribution', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        # FVC errors
        parts2 = axes[1].violinplot(fvc_data, positions=range(len(variants)),
                                     showmeans=True, showmedians=True)
        axes[1].set_xticks(range(len(variants)))
        axes[1].set_xticklabels(variant_labels, fontsize=10)
        axes[1].set_ylabel('FVC MAE per patient (ml)', fontsize=12)
        axes[1].set_title('FVC Error Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def run_comprehensive_comparison(
    cnn_model,
    test_ds,
    test_patients,
    patient_data,
    features_data,
    correctors_dict,  # {'demographics': (model, scaler), 'handcrafted': ..., 'full': ...}
    device='cuda',
    save_dir='./comparison_results'
):
    """
    Run complete comparison of all 4 variants

    Args:
        cnn_model: Trained CNN
        test_ds: Test dataset
        test_patients: List of test patient IDs
        patient_data: Dict with patient info
        features_data: Dict with features
        correctors_dict: Dict of trained correctors
        device: cuda/cpu
        save_dir: Where to save results

    Returns:
        results_dict: Complete results for all variants
    """

    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    print(f"\nComparing 4 variants:")
    print("  1. CNN-only (baseline)")
    print("  2. CNN + Demographics (age, sex, smoking)")
    print("  3. CNN + Handcrafted Features (9 mask-based)")
    print("  4. CNN + Demographics + Handcrafted (all 12)")

    # Initialize evaluator
    evaluator = MultiModelEvaluator(
        cnn_model, correctors_dict, {},
        patient_data, features_data, device
    )

    # Evaluate all variants
    results_dict = evaluator.evaluate_all_variants(test_ds, test_patients)

    # Visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # 1. Performance comparison
    ComparisonVisualizer.plot_performance_comparison(
        results_dict,
        save_path=os.path.join(save_dir, 'performance_comparison.png')
    )

    # 2. Error distributions
    ComparisonVisualizer.plot_error_distributions(
        results_dict,
        save_path=os.path.join(save_dir, 'error_distributions.png')
    )

    # 3. Sample patient trajectories (select 3 diverse patients)
    # Pick patients with different progression rates
    sample_patients = select_diverse_patients(patient_data, test_patients, n=3)

    ComparisonVisualizer.plot_patient_trajectories(
        evaluator, test_ds, sample_patients,
        save_path=os.path.join(save_dir, 'patient_trajectories.png')
    )

    # Save detailed results
    for variant in results_dict:
        df = results_dict[variant]['predictions_df']
        df.to_csv(os.path.join(save_dir, f'{variant}_predictions.csv'), index=False)

    # Save summary
    summary = {
        variant: results_dict[variant]['metrics']
        for variant in results_dict
    }

    import json
    with open(os.path.join(save_dir, 'comparison_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ All results saved to: {save_dir}")

    return results_dict

def select_diverse_patients(patient_data, patient_list, n=3):
    """
    Select n patients with diverse progression patterns
    """
    slopes = [(pid, patient_data[pid]['slope']) for pid in patient_list
              if pid in patient_data]
    slopes.sort(key=lambda x: x[1])

    # Pick: most declining, stable, improving
    indices = [0, len(slopes)//2, -1]
    selected = [slopes[i][0] for i in indices[:n]]

    return selected

def build_dataframe(patient_rows):
    """Build dataframe from patient predictions - handles dynamic features"""
    df = pd.DataFrame(patient_rows)
    print(df.columns)
    df = df.drop_duplicates(subset=["patient_id"])
    return df

def normalize_df(df, scaler=None):
    """Normalize all features except patient_id and true_slope"""
    exclude = ["patient_id", "true_slope"]
    feature_cols = [c for c in df.columns if c not in exclude]

    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(df[feature_cols])
    else:
        X = scaler.transform(df[feature_cols])

    df_scaled = pd.DataFrame(X, columns=feature_cols)
    df_scaled["patient_id"] = df["patient_id"].values
    df_scaled["true_slope"] = df["true_slope"].values

    return df_scaled, scaler, feature_cols

def build_feature_vectors(features_data):
    feat_dict = {}
    for pid, f in features_data.items():
        feat_vec = np.array([
            f['approx_vol'],
            f['avg_num_tissue_pixel'],
            f['avg_tissue'],
            f['avg_tissue_thickness'],
            f['avg_tissue_by_total'],
            f['avg_tissue_by_lung'],
            f['mean'],
            f['skew'],
            f['kurtosis'],
            f['age'],
            f['sex'],
            f['smoking_status'],
            f['age'],
            f['sex'],
            f['smoking_status']
        ], dtype=float)
        feat_dict[pid] = feat_vec
    return feat_dict

class SlopeCorrector(nn.Module):
    def __init__(self, input_dim=10):  # ← Changed default from 10 to 13
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        slope_cnn = x[:, 0:1]  # slope_cnn_mean is FIRST feature
        correction = self.mlp(x)
        final_slope = slope_cnn + correction
        return final_slope

def train_slope_corrector(df_train, df_val, feature_cols, device):
    """Train corrector - automatically uses correct input_dim"""

    X_train = torch.tensor(df_train[feature_cols].values, dtype=torch.float32).to(device)
    y_train = torch.tensor(df_train["true_slope"].values, dtype=torch.float32).unsqueeze(1).to(device)

    X_val = torch.tensor(df_val[feature_cols].values, dtype=torch.float32).to(device)
    y_val = torch.tensor(df_val["true_slope"].values, dtype=torch.float32).unsqueeze(1).to(device)

    # Corrector with correct input dim (13 features)
    corrector = SlopeCorrector(input_dim=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(corrector.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None

    print(f"\n🔧 Training corrector with {len(feature_cols)} features:")
    print(f"   Features: {feature_cols}")

    for epoch in range(30):
        corrector.train()
        pred = corrector(X_train)
        loss = criterion(pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        corrector.eval()
        with torch.no_grad():
            pred_val = corrector(X_val)
            val_loss = criterion(pred_val, y_val).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(corrector.state_dict())

        if epoch % 5 == 0:
            print(f"[Corrector] Epoch {epoch+1:02d} | Train={loss.item():.4f} | Val={val_loss:.4f}")

    corrector.load_state_dict(best_state)
    print(f"✅ Best validation loss: {best_val:.4f}")

    return corrector, best_val

class PatientBatchSampler(Sampler):
  def __init__(self,dataset,patients_per_batch=4,shuffle=True):
    self.ds = dataset
    self.shuffle = shuffle
    self.ppb =patients_per_batch

  def __iter__(self):
    patients = list(self.ds.patients)
    if self.shuffle:
      random.shuffle(patients)

    for i in range(0,len(patients),self.ppb):
      batch_pids = patients[i:i+self.ppb]
      idxs = []
      for pid in batch_pids:
        pidxs = list(self.ds.patient_to_indices[pid])
        idxs.extend(pidxs)
      yield idxs

  def __len__(self):
    from math import ceil
    return (len(self.ds.patients)+self.ppb -1) // self.ppb

def patient_group_collate(batch):

    # Filtra elementi None
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return {
            'images': torch.empty(0, 3, 224, 224),
            'slopes': torch.empty(0),
            'patient_ids': [],
            'lengths': torch.empty(0, dtype=torch.long),
            'slice_paths': [],
            'features': { }  # dizionario vuoto
        }

    # Estrai tensori
    images = torch.stack([b['image'] for b in batch])
    slopes = torch.stack([b['slope'] for b in batch]).view(-1)
    slice_paths = [b['slice_path'] for b in batch]

    # Feature per slice (dict of lists)
    features_dict = {k: [] for k in batch[0]['feature_patient'].keys()}
    for b in batch:
        for k, v in b['feature_patient'].items():
            features_dict[k].append(v)

    # Group slices by patient
    lengths, pid_order = [], []
    i = 0
    while i < len(batch):
        pid = batch[i]['patient_id']
        j = i
        while j < len(batch) and batch[j]['patient_id'] == pid:
            j += 1
        lengths.append(j - i)
        pid_order.append(pid)
        i = j

    return {
        'images': images,
        'slopes': slopes,
        'patient_ids': pid_order,
        'lengths': torch.tensor(lengths, dtype=torch.long),
        'slice_paths': slice_paths,
        'features': features_dict
    }


