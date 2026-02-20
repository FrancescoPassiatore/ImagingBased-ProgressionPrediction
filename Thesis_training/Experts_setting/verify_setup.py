"""
Verification Script for Expert Ensemble System
Checks installation and data paths
"""

import sys
import importlib
from pathlib import Path


def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✓ {package_name}")
        return True
    except ImportError:
        print(f"✗ {package_name} - NOT INSTALLED")
        return False


def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available (GPU: {torch.cuda.get_device_name(0)})")
            return True
        else:
            print("⚠ CUDA not available (will use CPU)")
            return False
    except:
        print("✗ Cannot check CUDA")
        return False


def check_path(path_str, description):
    """Check if a path exists"""
    path = Path(path_str)
    if path.exists():
        if path.is_dir():
            n_files = len(list(path.iterdir()))
            print(f"✓ {description} ({n_files} items)")
        else:
            print(f"✓ {description}")
        return True
    else:
        print(f"✗ {description} - NOT FOUND")
        return False


def main():
    print("="*80)
    print("EXPERT ENSEMBLE SYSTEM - VERIFICATION")
    print("="*80)
    
    # Check Python version
    print(f"\nPython Version: {sys.version}")
    if sys.version_info < (3, 8):
        print("⚠ Warning: Python 3.8+ recommended")
    else:
        print("✓ Python version OK")
    
    # Check packages
    print("\n" + "="*80)
    print("CHECKING DEPENDENCIES")
    print("="*80)
    
    packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('scikit-learn', 'sklearn'),
        ('lightgbm', 'lightgbm'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('scipy', 'scipy'),
        ('tqdm', 'tqdm')
    ]
    
    all_installed = True
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            all_installed = False
    
    # Check CUDA
    print("\n" + "="*80)
    print("CHECKING CUDA")
    print("="*80)
    check_cuda()
    
    # Check data paths (examples - modify as needed)
    print("\n" + "="*80)
    print("CHECKING DATA PATHS (optional)")
    print("="*80)
    
    paths_to_check = [
        (r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\train.csv", 
         "Training CSV"),
        (r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Feature_extraction\patient_features_improved.csv", 
         "Patient Features CSV"),
        (r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy_full_dataset", 
         "NPY Directory"),
        (r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\1_progression_maxpooling\cnn_features_efficientnet_b1.csv", 
         "CNN Features CSV")
    ]
    
    print("(These paths are from config - modify if needed)")
    for path, desc in paths_to_check:
        check_path(path, desc)
    
    # Check local files
    print("\n" + "="*80)
    print("CHECKING LOCAL FILES")
    print("="*80)
    
    local_files = [
        'train_ensemble.py',
        'cnn_expert.py',
        'lgbm_expert.py',
        'meta_model.py',
        'ensemble_utils.py'
    ]
    
    current_dir = Path(__file__).parent
    for filename in local_files:
        file_path = current_dir / filename
        if file_path.exists():
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename} - MISSING")
            all_installed = False
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if all_installed:
        print("✓ All required packages and files are present")
        print("\nYou can now run: python train_ensemble.py")
    else:
        print("✗ Some packages or files are missing")
        print("\nInstall missing packages with:")
        print("  pip install -r requirements.txt")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
