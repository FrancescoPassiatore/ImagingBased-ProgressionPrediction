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

    # =============================================================================
    # STEP 1: LOAD CNN MODEL
    # =============================================================================
    print("\n" + "="*80)
    print("[1/10] LOADING CNN MODEL")
    print("="*80)
    
    cnn_model = ImprovedSliceLevelCNN(backbone_name='efficientnet_b0', pretrained=False)
    cnn_model.load_state_dict(torch.load(
        r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\checkpoints_kfold_added_hf_tabular_attention_adj_lr\checkpoints_kfold_added_hf_tabular_attention_adj_lr\cnn_final.pth', 
        map_location=torch.device('cpu')
    ))

    # =============================================================================
    # PATHS AND HYPERPARAMETERS
    # =============================================================================
    CSV_PATH = 'Training/CNN_Slope_Prediction/train_with_coefs.csv'
    CSV_PATH_LABEL_52 = 'Training/Progression_prediction_risk_2/data/patient_progression_52w.csv'
    CSV_FEATURES_PATH = 'Training/CNN_Slope_Prediction/patient_features.csv'
    NPY_DIR = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy'

    IMAGE_SIZE = (224, 224)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Using device: {device}")
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")

    # Move model to device
    cnn_model = cnn_model.to(device)

    # =============================================================================
    # STEP 2: TRAIN/VAL/TEST SPLIT
    # =============================================================================
    print("\n" + "="*80)
    print("[3/10] LOADING TRAIN/VAL/TEST SPLITS")
    print("="*80)

    train_ids = pd.read_csv("Training/Progression_prediction_risk_2/data/train_patients_52w.csv")['Patient'].tolist()
    val_ids   = pd.read_csv("Training/Progression_prediction_risk_2/data/val_patients_52w.csv")['Patient'].tolist()
    test_ids  = pd.read_csv("Training/Progression_prediction_risk_2/data/test_patients_52w.csv")['Patient'].tolist()

    print(f"✓ Train patients: {len(train_ids)}")
    print(f"✓ Val patients: {len(val_ids)}")
    print(f"✓ Test patients: {len(test_ids)}")

    # =============================================================================
    # STEP 3: LOAD DATA
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
    # STEP 4: EXTRACT FEATURES FROM SLICES
    # =============================================================================
    print("\n" + "="*80)
    print("[4/10] EXTRACTING FEATURES FROM SLICES")
    print("="*80)
    

    # Get all patient IDs (you might want to filter to train+val+test only)
    all_patient_ids = list(patient_data.keys())
    
    loader = DataLoader(
        SliceFeatureDataset(all_patient_ids, patient_data),
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn  # ADD CUSTOM COLLATE FUNCTION
    )

    patient_feats = defaultdict(list)

    cnn_model.eval()
    with torch.no_grad():
        for images, pids in tqdm(loader, desc="Extracting features"):
            images = images.to(device)
            
            # Extract features
            z = cnn_model.extract_features(images)
            
            # Store features per patient
            for i, pid in enumerate(pids):
                patient_feats[pid].append(z[i].cpu().numpy())

    # Average features across slices for each patient
    patient_embeddings = {
        pid: np.mean(v, axis=0)
        for pid, v in patient_feats.items()
    }
    
    print(f"\n✓ Extracted embeddings for {len(patient_embeddings)} patients")
    sample_emb = list(patient_embeddings.values())[0]
    print(f"✓ Embedding shape: {sample_emb.shape}")

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

    
    
    

    # Plotta la storia del training
    plot_training_history(history, save_path='training_history.png')


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

    # Plotta la ROC curve
    plot_roc_curve(test_labels, test_preds, save_path='roc_curve.png')
    
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


    
