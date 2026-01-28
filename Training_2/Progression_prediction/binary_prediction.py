import json
from utilities import *
import warnings
import pickle
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def collate_fn(batch):
    """Custom collate function to handle dictionary outputs"""
    images = torch.stack([item['image'] for item in batch])
    patient_ids = [item['patient_id'] for item in batch]
    return images, patient_ids

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

def collate_fn(batch):
    """Custom collate function to handle dictionary outputs"""
    images = torch.stack([item['image'] for item in batch])
    patient_ids = [item['patient_id'] for item in batch]
    return images, patient_ids

def  extract_features_with_cnn(cnn_model, patient_ids, patient_data, device, batch_size=32):
    """Extract features using CNN model"""
    loader = DataLoader(
        SliceFeatureDataset(patient_ids, patient_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    patient_feats = defaultdict(list)
    cnn_model.eval()
    with torch.no_grad():
        for images, pids in tqdm(loader, desc="Extracting features", leave=False):
            images = images.to(device)
            z = cnn_model.extract_features(images)
            for i, pid in enumerate(pids):
                patient_feats[pid].append(z[i].cpu().numpy())

    patient_embeddings = {
        pid: np.mean(v, axis=0)
        for pid, v in patient_feats.items()
    }
    return patient_embeddings
"""
# Run Optuna study
study, best_trial = run_optuna_study(
    train_loader, 
    val_loader, 
    device,
    n_trials=50  # Adjust based on your time budget
)

# Train final model with best parameters
final_model, history = train_with_best_params(
    best_trial.params,
    train_loader,
    val_loader,
    test_loader,
    device,
    max_epochs=100
)

Best hyperparameters:
lr: 1.7345566642360933e-05
weight_decay: 0.0002669866674274458
optimizer: adam
scheduler: plateau
use_class_weights: True
grad_clip: 0.5
batch_size: 32
img_hidden: 256
hand_hidden: 64
n_fusion_layers: 1
dropout: 0.4
activation: leaky_relu
use_layer_norm: False
fusion_layer_0: 256


"""



if __name__ == "__main__":
    
    # Disable warnings
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', message='X does not have valid feature names')


    
    # Paths
    KFOLD_SPLITS_PATH = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training_2\Kfold_cyclic\kfold_cyclic_splits.pkl'
    CNN_MODELS_DIR = Path(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training_2\CNN_Training\Cyclic_kfold\checkpoints_trainings\checkpoints_mse')
    N_FOLDS = 5
    BATCH_SIZE = 32

    # =============================================================================
    # STEP 1: LOAD CYCLIC K-FOLD SPLITS
    # =============================================================================
    print("\n[1/5] LOADING CYCLIC K-FOLD SPLITS")
    
    with open(KFOLD_SPLITS_PATH, 'rb') as f:
        kfold_splits = pickle.load(f)
    
    print(f"✓ Loaded {len(kfold_splits)} folds")

    # =============================================================================
    # PATHS AND CONFIGURATION
    # =============================================================================
    CSV_PATH = r'Training\CNN_Slope_Prediction\train_with_coefs.csv'
    CSV_PATH_LABEL_52 = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\Progression_prediction_risk_2\data\patient_progression_52w.csv'
    CSV_FEATURES_PATH = r'Training\CNN_Slope_Prediction\patient_features.csv'
    NPY_DIR = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Device: {device}")

    # =============================================================================
    # STEP 2: LOAD DATA
    # =============================================================================
    print("\n" + "="*80)
    print("[2/10] LOADING DATA")
    print("="*80)

    dl = IPFDataLoaderPredictorProgression(CSV_PATH, CSV_FEATURES_PATH, NPY_DIR)
    patient_data, features_data = dl.get_patient_data()

    print(f"\n✓ Loaded patient_data for {len(patient_data)} patients")
    print(f"✓ Loaded features_data for {len(features_data)} patients")

    # Verify data structure
    sample_patient = list(patient_data.keys())[0]
    print(f"\n📋 Sample patient data structure (ID: {sample_patient}):")
    for key, value in patient_data[sample_patient].items():
        if isinstance(value, list):
            print(f"   {key}: list with {len(value)} items")
        else:
            print(f"   {key}: {type(value).__name__} = {value}")

    print(f"\n📋 Sample feature data structure:")
    for key, value in features_data[sample_patient].items():
        print(f"   {key}: {value:.2f}" if isinstance(value, float) else f"   {key}: {value}")


    # =============================================================================
    # STEP 3: CYCLIC K-FOLD TRAINING
    # =============================================================================
    print(f"\n[3/5] RUNNING CYCLIC {N_FOLDS}-FOLD CROSS-VALIDATION")
    
    all_fold_results = []
    all_predictions = {}
    
    for fold_idx in range(N_FOLDS):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx}/{N_FOLDS - 1}")
        print(f"{'='*80}")
        
        # Get splits for this fold
        split = kfold_splits[fold_idx]
        train_ids = split['train']
        val_ids = split['val']
        test_ids = split['test']

        print(f"Train: {len(train_ids)} patients")
        print(f"Val:   {len(val_ids)} patients")
        print(f"Test:  {len(test_ids)} patients")
        
        # =============================================================================
        # LOAD CNN MODEL FOR THIS FOLD
        # =============================================================================
        print(f"\n📦 Loading CNN model for fold {fold_idx}...")
        
        cnn_model_path = CNN_MODELS_DIR / f'cnn_fold{fold_idx}.pt'
        
        if not cnn_model_path.exists():
            print(f"⚠️  CNN model not found at {cnn_model_path}")
            print(f"   Skipping fold {fold_idx}")
            continue
        
        # Load CNN model
        cnn_model = ImprovedSliceLevelCNNExtractor(backbone_name='efficientnet_b1', pretrained=False)
        checkpoint = torch.load(cnn_model_path, map_location=device, weights_only=False)
        cnn_model.load_state_dict(checkpoint['model_state_dict'])
        cnn_model = cnn_model.to(device)
        print(f"✓ CNN model loaded from {cnn_model_path}")
        
        # =============================================================================
        # EXTRACT FEATURES
        # =============================================================================
        print(f"\n🔍 Extracting features for fold {fold_idx}...")
        
        # Get all unique patient IDs for this fold
        all_fold_ids = list(set(train_ids + val_ids + test_ids))
        
        # Extract features using the fold-specific CNN
        patient_embeddings = extract_features_with_cnn(
            cnn_model, all_fold_ids, patient_data, device, BATCH_SIZE
        )
        print(f"✓ Extracted {len(patient_embeddings)} embeddings")
        
        # Free up memory
        del cnn_model
        torch.cuda.empty_cache()
       
    # =============================================================================
    # COMPUTE NORMALIZATION STATS (TRAIN ONLY)
    # =============================================================================

    feature_stats = compute_feature_stats(
        handcrafted_dict=features_data,
        patient_ids=train_ids,
        feature_names=NORMALIZE_FEATURES
    )

    print("\n📊 Feature normalization stats (TRAIN only):")
    for k, v in feature_stats.items():
        print(f"   {k}: mean={v['mean']:.3f}, std={v['std']:.3f}")

    # =============================================================================
    # STEP 5: CREATE MLP DATASETS
    # =============================================================================
    print("\n" + "="*80)
    print("[5/10] CREATING MLP DATASETS")
    print("="*80)

    train_ds = PatientMLPDataset(
        label_csv=CSV_PATH_LABEL_52,
        embeddings_dict=patient_embeddings,
        handcrafted_dict=features_data,
        patient_list=train_ids,
        feature_stats=feature_stats
    )
    
    val_ds = PatientMLPDataset(
        label_csv=CSV_PATH_LABEL_52,
        embeddings_dict=patient_embeddings,
        handcrafted_dict=features_data,
        patient_list=val_ids,
        feature_stats=feature_stats
    )
    
    test_ds = PatientMLPDataset(
        label_csv=CSV_PATH_LABEL_52,
        embeddings_dict=patient_embeddings,
        handcrafted_dict=features_data,
        patient_list=test_ids,
        feature_stats=feature_stats
    )
    
    print(f"✓ Train dataset: {len(train_ds)} patients")
    print(f"✓ Val dataset: {len(val_ds)} patients")
    print(f"✓ Test dataset: {len(test_ds)} patients")
    
    # Test dataset
    print("\n📋 Sample from train dataset:")
    sample = train_ds[0]
    print(f"   x_img shape: {sample['x_img'].shape}")
    print(f"   x_hand shape: {sample['x_hand'].shape}")
    print(f"   y: {sample['y'].item()}")
    print(f"   patient_id: {sample['patient_id']}")

    sample = train_ds[0]['x_hand']
    print("Mean:", sample.mean().item())
    print("Std:", sample.std().item())
    
    print("\n✅ Data preparation complete!")

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"\n✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    
    # Get embedding dimension from a sample
    sample = train_ds[0]
    img_dim = sample['x_img'].shape[0]
    hand_dim = sample['x_hand'].shape[0]
    
    print(f"\n📊 Feature dimensions:")
    print(f"   Image embedding: {img_dim}")
    print(f"   Handcrafted features: {hand_dim}")
    






    # =============================================================================
    # MLP MODEL 
    # =============================================================================
    print("\n" + "="*80)
    print("[6/10] INITIALIZING AND TRAINING MLP MODEL")
    print("="*80)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


   

    model = SimpleFusionMLP(
        img_dim=320,
        hand_dim=12,
        hidden=32,       
        dropout=0.6         
    ).to('cuda')
    
    print(f"\n✓ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,  # Your train DataLoader with batch_size=32
        val_loader=val_loader,      # Your validation DataLoader
        epochs=100,
        lr=1e-4,
        weight_decay=1e-2,
        grad_clip=1.0,
        use_class_weights=True,
        device='cuda'
    )

    # Evaluate on test set
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)
    
    # Load best model
    model.load_state_dict(torch.load('best_fusion_mlp.pth'))
    
    criterion = nn.BCEWithLogitsLoss()
    test_metrics, test_preds, test_labels, test_pids = validate(
        model, test_loader, criterion, device
    )

    
    # Valuta sul test set
    test_metrics, test_preds, test_labels, _ = validate(model, test_loader, criterion, device)

    
    print(f"\n📊 Test Set Results:")
    print(f"   AUC: {test_metrics['auc']:.4f}")
    print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   Recall: {test_metrics['recall']:.4f}")
    print(f"   F1 Score: {test_metrics['f1']:.4f}")
    
    results_df = pd.DataFrame({
        'patient_id': test_pids,
        'true_label': test_labels,
        'predicted_prob': test_preds
    })
    results_df.to_csv('test_predictions.csv', index=False)
    print("\n✅ Predictions saved to 'test_predictions.csv'")


    
