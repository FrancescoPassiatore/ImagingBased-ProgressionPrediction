"""
Add test set predictions to MLP corrector prediction files using saved best models.

- Loads each best model checkpoint for each fold and feature type
- Loads the test set for each fold
- Runs inference on the test set
- Updates the corresponding predictions pickle file with a 'test' key
"""

import torch
import pickle
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utilities import IPFDataLoader, CorrectorDataset, FeatureNormalizer, ImprovedSlopeCorrector

CONFIG = {
    'npy_dir': 'D:/FrancescoP/ImagingBased-ProgressionPrediction/Dataset/extracted_npy/extracted_npy',
    'train_csv': 'D:/FrancescoP/ImagingBased-ProgressionPrediction/Training/CNN_Slope_Prediction/train_with_coefs.csv',
    'features_csv': 'D:/FrancescoP/ImagingBased-ProgressionPrediction/Training/CNN_Slope_Prediction/patient_features.csv',
    'best_params_dir': Path('D:/FrancescoP/ImagingBased-ProgressionPrediction/Training_2/MLP_Corrector/optuna/best_params'),
    'models_dir': Path('D:/FrancescoP/ImagingBased-ProgressionPrediction/Training_2/MLP_Corrector/models_effnetb1_oversampling_huber_median'),
    'predictions_dir': Path('D:/FrancescoP/ImagingBased-ProgressionPrediction/Training_2/MLP_Corrector/predictions_effnetb1_oversampling_huber_median'),
    'splits_path': Path('Training_2/kfold_splits.pkl'),
    'feature_types': ['demographics', 'handcrafted', 'full'],
    'n_folds': 5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# Load patient and feature data
print("\n📁 Loading patient data...")
data_loader = IPFDataLoader(
    csv_path=CONFIG['train_csv'],
    features_path=CONFIG['features_csv'],
    npy_dir=CONFIG['npy_dir']
)
patient_data, features_data = data_loader.get_patient_data()
print(f"✓ Loaded {len(patient_data)} patients")

# Load splits
with open(CONFIG['splits_path'], 'rb') as f:
    splits = pickle.load(f)

for feature_type in CONFIG['feature_types']:
    print(f"\n{'='*80}")
    print(f"ADDING TEST PREDICTIONS: {feature_type.upper()}")
    print(f"{'='*80}")
    # Load best params for this feature type
    import yaml
    params_path = CONFIG['best_params_dir'] / f'best_params_{feature_type}.yaml'
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    for fold_idx in range(CONFIG['n_folds']):
        
        print(f"\nFOLD {fold_idx}")
        # Load CNN test slopes for this fold
        cnn_pred_path = Path('D:/FrancescoP/ImagingBased-ProgressionPrediction/Training_2/CNN_Training/predictions_efficientnet_b1_oversampling_huber_median') / f'cnn_predictions_fold{fold_idx}.pkl'
        with open(cnn_pred_path, 'rb') as f:
            cnn_predictions_all = pickle.load(f)
        cnn_test_slopes = cnn_predictions_all['test']
        # Get test IDs
        test_ids = splits[fold_idx]['test'] if isinstance(splits[fold_idx], dict) else splits[fold_idx][2]
        # Filter for NaN in features
        test_ids_clean = []
        for pid in test_ids:
            if pid not in features_data or pid not in patient_data:
                continue
            pdata = features_data[pid]
            has_nan = False
            if feature_type in ['handcrafted', 'full']:
                from utilities import HAND_FEATURE_ORDER
                hand_feats = [pdata.get(f, np.nan) for f in HAND_FEATURE_ORDER]
                if any(np.isnan(hand_feats)):
                    has_nan = True
            if feature_type in ['demographics', 'full']:
                if np.isnan(pdata.get('age', np.nan)):
                    has_nan = True
            if not has_nan:
                test_ids_clean.append(pid)
        # Load normalizer from model checkpoint
        model_path = CONFIG['models_dir'] / f'{feature_type}_fold{fold_idx}_best.pt'
        checkpoint = torch.load(model_path, map_location=CONFIG['device'], weights_only=False)
        normalizer = checkpoint['normalizer']
        # Input dim
        if feature_type == 'demographics':
            input_dim = 1 + 3
        elif feature_type == 'handcrafted':
            input_dim = 1 + 9
        else:
            input_dim = 1 + 9 + 3
        # Hidden sizes
        hidden_sizes = [2 ** params[f'hidden_{i}_log'] for i in range(params['n_layers'])]
        dropout_rates = [params[f'dropout_{i}'] for i in range(params['n_layers'])]
        # Model
        model = ImprovedSlopeCorrector(
            input_dim=input_dim,
            hidden_dims=hidden_sizes,
            dropout_rates=dropout_rates
        ).to(CONFIG['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        # Test dataset/loader
        test_dataset = CorrectorDataset(
            test_ids_clean,
            patient_data,
            features_data,
            cnn_test_slopes,  # Use CNN test slopes for test set
            feature_type=feature_type,
            normalizer=normalizer
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=0
        )
        # Predict
        test_patient_ids = []
        test_preds = []
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(CONFIG['device'])
                preds = model(features).squeeze()
                test_patient_ids.extend(batch['patient_id'])
                test_preds.extend(preds.cpu().numpy())
        test_predictions_dict = {pid: pred for pid, pred in zip(test_patient_ids, test_preds)}
        # Update predictions file
        pred_path = CONFIG['predictions_dir'] / f'{feature_type}_predictions_fold{fold_idx}.pkl'
        with open(pred_path, 'rb') as f:
            predictions = pickle.load(f)
        predictions['test'] = test_predictions_dict
        with open(pred_path, 'wb') as f:
            pickle.dump(predictions, f)
        print(f"  ✓ Added test predictions to {pred_path}")

print("\n" + "="*80)
print("✅ TEST PREDICTIONS ADDED TO ALL FOLDS")
print("="*80)
