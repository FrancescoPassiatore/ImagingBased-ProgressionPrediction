# Improved Demographics Integration - Progression Prediction

## Overview
Updated the progression prediction model (1_progression_ablation) to match the improved feature handling from the FVC prediction model (2_fvc_prediction):

1. **Separate Hand-Crafted and Demographic Branches**: Hand-crafted radiomics and demographics now flow through separate neural network branches
2. **Improved Demographics Preprocessing**: Better normalization and encoding for demographic features

## Changes Made

### 1. ablation_study.py

#### Added Functions:
- `preprocess_demographics_improved()`: New function that properly preprocesses demographic features:
  - **Age**: Normalized with StandardScaler (mean=0, std=1)
  - **Sex**: Binary encoding centered around 0 (Female=-1, Male=1) instead of (0, 1)
  - **SmokingStatus**: One-hot encoded into 3 features (Smoking_0, Smoking_1, Smoking_2) and centered to [-0.5, 0.5]

- `get_preprocessed_demo_features()`: Extracts preprocessed demographic features from a DataFrame row

#### Modified Functions:
- `normalize_features_per_fold()`: 
  - Now returns tuple `(result_df, scalers)` instead of just `result_df`
  - Uses `preprocess_demographics_improved()` for demographics instead of old normalization
  - Returns encoding info in `scalers['demo_encoding']` dict

- `create_feature_set_for_fold()`:
  - Now returns tuple `(result_df, encoding_info)` instead of just `result_df`
  - Calculates actual demographic feature count (accounts for one-hot encoding expansion)
  - Extracts and returns encoding_info for passing to dataloaders

- `train_single_fold()`:
  - Added `encoding_info` parameter
  - Passes encoding_info to `create_dataloaders()`
  - Extracts actual hand and demo dimensions from sample batch (handles None values)
  - Uses actual dimensions when creating model

- `run_ablation_study()`:
  - Receives encoding_info from `create_feature_set_for_fold()`
  - Passes encoding_info to `train_single_fold()`

### 2. utilities.py

#### Modified Classes/Functions:

- `PatientSliceDataset.__init__()`:
  - Added `encoding_info` parameter
  - Calculates actual `demo_feature_dim` accounting for one-hot encoding:
    - Age: 1 feature (Age_normalized)
    - Sex: 1 feature (Sex_encoded)
    - SmokingStatus: 3 features (one-hot: Smoking_0, Smoking_1, Smoking_2)
  - Stores encoding_info for use in `__getitem__`

- `PatientSliceDataset.__getitem__()`:
  - Now returns separate `hand_features` and `demo_features` tensors
  - Extracts preprocessed demographic features:
    - Age_normalized (if Age in demo_feature_cols)
    - Sex_encoded (if Sex in demo_feature_cols)
    - Smoking_0, Smoking_1, Smoking_2 (if SmokingStatus in demo_feature_cols)

- `collate_patient_batch()`:
  - Returns separate `hand_features` and `demo_features` instead of combined `patient_features`
  - Returns None if features are empty (instead of empty tensor)

- `create_dataloaders()`:
  - Added `encoding_info` parameter
  - Passes encoding_info to all PatientSliceDataset instances
  - Added `drop_last=True` to training DataLoader to prevent BatchNorm errors with batch_size=1

### 3. model_train.py

#### Modified Model:

- `ProgressionPredictionModel.__init__()`:
  - Creates **separate processing branches**:
    - **Hand-crafted branch**: `hand_feature_dim` → 64 (with BatchNorm, ReLU, Dropout)
    - **Demographic branch**: `demo_feature_dim` → 32 (with BatchNorm, ReLU, Dropout)
  - Total input to classifier: `cnn_dim + 64 + 32` (when both branches active)

- `ProgressionPredictionModel.forward()`:
  - Now accepts separate `hand_features` and `demo_features` parameters
  - Processes each through dedicated branches
  - Expands processed features to match slice dimensions
  - Concatenates CNN + processed_hand + processed_demo for classification

- `ProgressionPredictionModel.predict_proba()`:
  - Updated to accept separate hand and demo features

#### Modified Trainer:

- `ModelTrainer.train_epoch()`:
  - Extracts `hand_features` and `demo_features` from batch (instead of combined `patient_features`)
  - Passes both to model forward()

- `ModelTrainer.evaluate()`:
  - Extracts `hand_features` and `demo_features` from batch
  - Passes both to model forward()

## Feature Dimension Changes

### Before:
```
Demographics: 3 features (Age, Sex, SmokingStatus)
- Age: normalized [0, 1] with MinMaxScaler
- Sex: 0 or 1
- SmokingStatus: 0, 1, or 2 (ordinal encoding)
Combined with hand-crafted: [9 hand features + 3 demo features] = 12 features
```

### After:
```
Demographics: 5 features (preprocessed)
- Age_normalized: StandardScaler (mean=0, std=1)
- Sex_encoded: -1 or 1 (centered)
- Smoking_0: -0.5 or 0.5 (one-hot, centered)
- Smoking_1: -0.5 or 0.5 (one-hot, centered)  
- Smoking_2: -0.5 or 0.5 (one-hot, centered)

Separate branches:
- Hand-crafted: 9 features → 64 (dedicated branch)
- Demographics: 5 features → 32 (dedicated branch)
- Combined: CNN (2048) + 64 + 32 = 2144 features per slice
```

## Benefits

1. **Better Neural Network Learning**: 
   - Zero-centered features (Sex: -1/1, Smoking: -0.5/0.5) help gradient flow
   - Proper normalization of continuous features (Age with StandardScaler)

2. **More Expressive Representation**:
   - One-hot encoding for SmokingStatus is more expressive than ordinal
   - Separate branches allow model to learn feature-type-specific transformations

3. **Architectural Improvements**:
   - Dedicated branches for hand-crafted vs demographic features
   - BatchNorm and Dropout tailored to each feature type
   - Hand branch: 64 dims (larger, more complex radiomics)
   - Demo branch: 32 dims (smaller, simpler demographics)

4. **Consistency**:
   - Now matches the architecture used in FVC prediction (2_fvc_prediction)
   - Easier to compare ablation results across tasks

## Validation

Run the ablation study to verify:
```bash
cd Thesis_training/1_progression_ablation
python ablation_study.py
```

Expected behavior:
- ✅ No dimension mismatch errors
- ✅ Demographics preprocessed with proper centering
- ✅ Model creates separate hand/demo branches
- ✅ Features flow through dedicated pathways
- ✅ Training/validation metrics computed correctly

## Compatibility

- ✅ Backward compatible with existing checkpoint loading (model architecture unchanged for CNN-only)
- ✅ Works with all ablation configurations (cnn_only, cnn_hand, cnn_demo, full)
- ✅ Per-fold normalization maintained (no data leakage)
