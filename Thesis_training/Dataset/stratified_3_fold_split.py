from sklearn.model_selection import StratifiedKFold
import pickle
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utilities import (
    IPFDataLoader
    )

# Ricrea gli split con 3 fold stratificati
def create_stratified_3fold(patient_ids, labels):
    """
    Create stratified 3-fold splits
    """
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    splits = {}
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(patient_ids, labels)):
        # Get patient IDs
        test_patients = [patient_ids[i] for i in test_idx]
        train_val_patients = [patient_ids[i] for i in train_val_idx]
        train_val_labels = [labels[i] for i in train_val_idx]
        
        # Further split train_val into train/val (80/20)
        from sklearn.model_selection import train_test_split
        train_patients, val_patients = train_test_split(
            train_val_patients,
            test_size=0.2,
            stratify=train_val_labels,
            random_state=42
        )
        
        splits[f'fold_{fold_idx}'] = {
            'train': train_patients,
            'val': val_patients,
            'test': test_patients
        }
        
        print(f"\nFold {fold_idx}:")
        print(f"  Train: {len(train_patients)} patients")
        print(f"  Val: {len(val_patients)} patients")
        print(f"  Test: {len(test_patients)} patients")
    
    return splits


if __name__ == "__main__":
    
    

    # Load patient data

    data_loader = IPFDataLoader(
            csv_path=Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\ground_truth.csv"),
            features_path=Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\patient_features.csv"),
            npy_dir=Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy")
        )

    patient_data , features_data = data_loader.get_patient_data()

    # Usage
    patient_ids = list(patient_data.keys())
    labels = [patient_data[pid]['gt_has_progressed'] for pid in patient_ids]

    new_splits = create_stratified_3fold(patient_ids, labels)

    # Save
    with open('kfold_splits_3fold_stratified.pkl', 'wb') as f:
        pickle.dump(new_splits, f)