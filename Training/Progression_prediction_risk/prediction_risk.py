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



if __name__ == "__main__":
    

    # Disabilita tutti i warning
    warnings.filterwarnings('ignore')

    # Oppure disabilita solo i warning specifici
    warnings.filterwarnings('ignore', category=UserWarning)

    # Oppure ancora più specifico
    warnings.filterwarnings('ignore', message='X does not have valid feature names')

    # 1. Load models
    cnn_model = ImprovedSliceLevelCNN(backbone_name='efficientnet_b0', pretrained=False)
    cnn_model.load_state_dict(torch.load(r'C:\Users\frank\OneDrive\Desktop\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\checkpoints_kfold_added_hf_tabular_attention_adj_lr\checkpoints_kfold_added_hf_tabular_attention_adj_lr\cnn_final.pth', map_location=torch.device('cpu')))






#-----------------------------------------------------------------------------------------------
    # Paths
    CSV_PATH = 'Training/CNN_Slope_Prediction/train_with_coefs.csv'
    CSV_PATH_LABEL_52 = 'Training/Progression_prediction_risk/data/patient_progression_52w.csv'
    CSV_FEATURES_PATH = 'Training/CNN_Slope_Prediction/patient_features.csv'
    NPY_DIR = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy'

    # Hyperparameters
    IMAGE_SIZE = (224, 224)

    # Device
    device = "cuda" 

    print(torch.__version__)
    print(torch.cuda.is_available())  # Should return True if CUDA is available

    # STEP 2: LOAD DATA

    print("\n" + "="*80)
    print("[1/10] LOADING DATA")
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
    # STEP 3: TRAIN/TEST SPLIT
    # =============================================================================

    train_ids = pd.read_csv("Training/Progression_prediction_risk/data/train_patients_52w.csv")['Patient'].tolist()
    val_ids   = pd.read_csv("Training/Progression_prediction_risk/data/val_patients_52w.csv")['Patient'].tolist()
    test_ids  = pd.read_csv("Training/Progression_prediction_risk/data/test_patients_52w.csv")['Patient'].tolist()

    loader = DataLoader(
        SliceFeatureDataset(list(patient_data.keys()),patient_data),
        batch_size=16,
        shuffle=False,
        num_workers=4
    )
    

    #Extract Patient features from slices
    
    patient_feats = defaultdict(list)

    cnn_model.eval()
    with torch.no_grad():
        for images, pids in loader:
            images = images.to(device)
            z = cnn_model.extract_features(images)

            for i, pid in enumerate(pids):
                patient_feats[pid].append(z[i].cpu().numpy())

    patient_embeddings = {
        pid: np.mean(v, axis=0)
        for pid, v in patient_feats.items()
    }
    
    #MLP

    train_ds = PatientMLPDataset(
        label_csv=CSV_PATH_LABEL_52,
        embeddings_dict=patient_embeddings,
        handcrafted_dict=features_data,
        patient_list=train_ids
    )
    
    print(train_ds)