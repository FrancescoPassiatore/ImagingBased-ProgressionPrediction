from Progression_prediction_slope.utilities import *




if __name__ == "__main__":
    import warnings

    # Disabilita tutti i warning
    warnings.filterwarnings('ignore')

    # Oppure disabilita solo i warning specifici
    warnings.filterwarnings('ignore', category=UserWarning)

    # Oppure ancora più specifico
    warnings.filterwarnings('ignore', message='X does not have valid feature names')

    # 1. Load models
    cnn_model = ImprovedSliceLevelCNN(backbone_name='efficientnet_b0', pretrained=False)
    cnn_model.load_state_dict(torch.load(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\checkpoints_kfold_added_hf_tabular_attention_adj_lr\checkpoints_kfold_added_hf_tabular_attention_adj_lr\cnn_final.pth', map_location=torch.device('cpu')))
    
    corrector_model = SlopeCorrector(input_dim=13)
    corrector_model.load_state_dict(torch.load(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\Progression_prediction_slope\files\corrector_full.pth', map_location=torch.device('cpu')))
    
    # Load scaler
    with open(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\Progression_prediction_slope\files\scaler_full.pkl', 'rb') as f:
        scaler, feature_cols = pickle.load(f)

    print(f"✓ Loaded corrector with {len(feature_cols)} features:")
    print(f"  Features: {feature_cols}")

    # Verify scaler looks reasonable
    print(f"\n{'='*70}")
    print("LOADED SCALER PARAMETERS:")
    print(f"{'='*70}")
    for i, col in enumerate(feature_cols):
        print(f"{col:25s}: mean={scaler.mean_[i]:12.2f}, std={scaler.scale_[i]:12.2f}")
    print(f"{'='*70}\n")

    # Paths
    CSV_PATH = 'Training/CNN_Slope_Prediction/train_with_coefs.csv'
    CSV_FEATURES_PATH = 'Training/CNN_Slope_Prediction/patient_features.csv'
    NPY_DIR = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy'

    # Hyperparameters
    IMAGE_SIZE = (224, 224)
    PATIENTS_PER_BATCH = 4

    # Device
    device = "cuda" 

    print(torch.__version__)
    print(torch.cuda.is_available())  # Should return True if CUDA is available

    # STEP 2: LOAD DATA

    print("\n" + "="*80)
    print("[1/10] LOADING DATA")
    print("="*80)

    dl = IPFDataLoader(CSV_PATH, CSV_FEATURES_PATH, NPY_DIR)
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

    print("\n" + "="*80)
    print("[2/10] CREATING TRAIN/TEST SPLIT")
    print("="*80)

    all_patients = list(patient_data.keys())
    print(f"\n✓ Total patients available: {len(all_patients)}")

    # Recreate exact same split as training
    np.random.seed(42)
    np.random.shuffle(all_patients)

    test_size = int(len(all_patients) * 0.2)
    test_patients = all_patients[:test_size]
    train_patients = all_patients[test_size:]

    print(f"✓ Train patients: {len(train_patients)}")
    print(f"✓ Test patients:  {len(test_patients)}")
    print(f"\n📝 First 5 test patients: {test_patients[:5]}")
    print(f"📝 First 5 train patients: {train_patients[:5]}")

    # Verify no overlap
    overlap = set(train_patients) & set(test_patients)
    if overlap:
        print(f"❌ ERROR: {len(overlap)} patients in both train and test!")
    else:
        print(f"✓ No patient overlap between train and test")

    # =============================================================================
    # STEP 4: CREATE DATASETS
    # =============================================================================

    print("\n" + "="*80)
    print("[3/10] CREATING DATASETS")
    print("="*80)

    train_ds = IPFSliceDataset(
        train_patients,
        patient_data,
        features_data,
        normalize_slope=True,
        image_size=IMAGE_SIZE
    )

    print(f"\n✓ Train dataset created:")
    print(f"   Total slices: {len(train_ds)}")
    print(f"   Patients: {len(train_ds.patients)}")
    print(f"   Slope scaler fitted: {train_ds.slope_scaler is not None}")

    test_ds = IPFSliceDataset(
        test_patients,
        patient_data,
        features_data,
        normalize_slope=True,
        image_size=IMAGE_SIZE
    )
    test_ds.slope_scaler = train_ds.slope_scaler

    print(f"\n✓ Test dataset created:")
    print(f"   Total slices: {len(test_ds)}")
    print(f"   Patients: {len(test_ds.patients)}")

    # Verify slope scaler
    if train_ds.slope_scaler:
        print(f"\n📊 Slope normalization:")
        print(f"   Mean: {train_ds.slope_scaler.mean_[0]:.2f} ml/week")
        print(f"   Std:  {train_ds.slope_scaler.scale_[0]:.2f} ml/week")

        # Test denormalization
        test_norm = np.array([[0.0]])
        test_denorm = train_ds.slope_scaler.inverse_transform(test_norm)[0][0]
        print(f"   Test: normalized=0.0 → denormalized={test_denorm:.2f}")

    # Sample a few items to verify loading
    print(f"\n🔍 Testing dataset loading (3 random samples)...")
    for i in np.random.choice(len(test_ds), 3, replace=False):
        sample = test_ds[i]
        if sample is not None:
            print(f"   ✓ Sample {i}: image shape={sample['image'].shape}, "
                f"slope={sample['slope'].item():.3f}, patient={sample['patient_id'][:10]}...")
        else:
            print(f"   ❌ Sample {i}: Failed to load")

    # =============================================================================
    # STEP 5: CREATE DATALOADERS
    # =============================================================================

    print("\n" + "="*80)
    print("[4/10] CREATING DATALOADERS")
    print("="*80)

    train_loader = DataLoader(
            train_ds,
            batch_sampler=PatientBatchSampler(
                train_ds,
                patients_per_batch=PATIENTS_PER_BATCH,
                shuffle=True  # ✓ Shuffle for training
            ),
            collate_fn=patient_group_collate,
            num_workers=4,
            pin_memory=True
        )

    test_loader = DataLoader(
        test_ds,
        batch_sampler=PatientBatchSampler(
            test_ds,
            patients_per_batch=PATIENTS_PER_BATCH,
            shuffle=False
        ),
        collate_fn=patient_group_collate,
        num_workers=4,
        pin_memory=True
    )

    print(f"\n✓ Test loader created:")
    print(f"   Total batches: {len(test_loader)}")
    print(f"   Patients per batch: {PATIENTS_PER_BATCH}")

    # Test loading one batch
    print(f"\n🔍 Testing batch loading...")
    try:
        test_batch = next(iter(test_loader))
        print(f"   ✓ Batch loaded successfully")
        print(f"   Images shape: {test_batch['images'].shape}")
        print(f"   Slopes shape: {test_batch['slopes'].shape}")
        print(f"   Patients in batch: {len(test_batch['patient_ids'])}")
        print(f"   Patient IDs: {test_batch['patient_ids']}")
        print(f"   Lengths: {test_batch['lengths'].tolist()}")
    except Exception as e:
        print(f"   ❌ Error loading batch: {e}")
