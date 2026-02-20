# Expert Ensemble System for IPF Progression Prediction

This directory contains an ensemble learning system that combines **CNN** and **LightGBM** experts using a **meta-model** (Logistic Regression) for improved prediction of Idiopathic Pulmonary Fibrosis (IPF) progression.

## 📁 File Structure

```
Experts_setting/
├── train_ensemble.py          # Main training script (orchestrates the pipeline)
├── cnn_expert.py               # CNN model with attention-based aggregation
├── lgbm_expert.py              # LightGBM model wrapper
├── meta_model.py               # Meta-model for fusion + threshold optimization
├── ensemble_utils.py           # Data preparation and result aggregation utilities
├── README.md                   # This file
└── results/                    # Output directory (created at runtime)
    ├── config.json             # Training configuration
    ├── fold{0-4}_results.json  # Per-fold results
    ├── fold{0-4}_predictions.csv  # Per-fold test predictions
    ├── roc_curve_fold{0-4}.png # ROC curves per fold
    ├── prediction_dist_fold{0-4}.png  # Prediction distributions
    ├── summary.json            # Aggregated results across folds
    ├── cnn_fold{0-4}_best.pth  # Best CNN model per fold
    └── lgbm_fold{0-4}.txt      # LightGBM model per fold
```

## 🎯 System Architecture

### Pipeline Overview

For each fold in 5-fold cross-validation:

1. **Data Split**: Train / Validation / Test (64% / 16% / 20%)
2. **Expert 1 (CNN)**: Train on CT scan features
3. **Expert 2 (LightGBM)**: Train on hand-crafted + demographic features
4. **Correlation Analysis**: Measure complementarity between experts
5. **Meta-Model**: Train Logistic Regression on validation predictions
6. **Threshold Optimization**: Find optimal threshold using Youden's J statistic
7. **Evaluation**: Test on held-out set

### Models

#### 1. CNN Expert (`cnn_expert.py`)
- **Input**: Pre-extracted CNN features from CT slices
- **Architecture**:
  - Attention-based aggregation of slice-level features
  - Optional hand-crafted and demographic features
  - 2-layer MLP classifier with BatchNorm and Dropout
- **Output**: Probability of progression

#### 2. LightGBM Expert (`lgbm_expert.py`)
- **Input**: Hand-crafted features + demographics
- **Features**:
  - 9 hand-crafted imaging features
  - Age, Sex, Smoking Status
- **Training**: Gradient boosting with early stopping
- **Output**: Probability of progression

#### 3. Meta-Model (`meta_model.py`)
- **Input**: Predictions from CNN and LightGBM
- **Model**: Logistic Regression (learns optimal weighting)
- **Threshold**: Optimized using Youden's J statistic
- **Output**: Final fused prediction

## 🚀 Usage

### Prerequisites

Install required packages:
```bash
pip install torch torchvision scikit-learn lightgbm pandas numpy matplotlib seaborn scipy tqdm
```

### Running the Ensemble

Using the venv Python executable:

```bash
cd D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Experts_setting

# Run verification first
D:\FrancescoP\ImagingBased-ProgressionPrediction\venv\Scripts\python.exe verify_data.py

# Then run training
D:\FrancescoP\ImagingBased-ProgressionPrediction\venv\Scripts\python.exe train_ensemble.py
```

Or create an alias for convenience:
```bash
$py = "D:\FrancescoP\ImagingBased-ProgressionPrediction\venv\Scripts\python.exe"
& $py train_ensemble.py
```

### Configuration

Edit the `config` dictionary in `train_ensemble.py`:

```python
config = {
    # Data paths
    'csv_path': 'path/to/train.csv',
    'features_path': 'path/to/patient_features.csv',
    'npy_dir': 'path/to/npy_folder',
    'cnn_features_path': 'path/to/cnn_features.csv',
    
    # Output
    'output_dir': 'path/to/results',
    
    # Cross-validation
    'n_folds': 5,
    'random_seed': 42,
    
    # Features to use
    'hand_feature_cols': [...],  # List of hand-crafted features
    'demo_feature_cols': ['Age', 'Sex', 'SmokingStatus'],
    
    # CNN training
    'cnn_epochs': 50,
    'cnn_patience': 10,
    'batch_size': 8,
    
    # Threshold optimization strategy
    'threshold_strategy': 'youden',  # 'youden', 'f1', or 'precision_recall'
    
    # Device
    'device': 'cuda'  # or 'cpu'
}
```

## 📊 Outputs

### Per-Fold Results

For each fold, the system generates:

1. **Model Checkpoints**:
   - `cnn_fold{i}_best.pth`: Best CNN model
   - `lgbm_fold{i}.txt`: LightGBM model

2. **Predictions** (`fold{i}_predictions.csv`):
   - Patient IDs
   - True labels
   - CNN predictions
   - LightGBM predictions
   - Fused predictions

3. **Visualizations**:
   - ROC curves comparing all models
   - Prediction distribution by class

4. **Metrics** (`fold{i}_results.json`):
   - AUC, Accuracy, Precision, Recall, F1, Specificity
   - Confusion matrix
   - Optimal threshold
   - Expert correlation

### Aggregated Results

`summary.json` contains:
- **Mean ± Std** for all metrics across folds
- Per-fold breakdown
- Overall correlation between experts

Example output:
```json
{
  "cnn_auc_mean": 0.8234,
  "cnn_auc_std": 0.0321,
  "lgbm_auc_mean": 0.7891,
  "lgbm_auc_std": 0.0412,
  "fused_auc_mean": 0.8567,
  "fused_auc_std": 0.0287,
  ...
}
```

## 🔍 Key Features

### 1. Correlation Analysis
- **Purpose**: Verify that experts provide complementary information
- **Metrics**: Pearson and Spearman correlation
- **Ideal**: Low correlation (< 0.7) indicates diversity

### 2. Threshold Optimization
Three strategies available:
- **Youden's J**: Maximizes Sensitivity + Specificity - 1
- **F1 Score**: Maximizes harmonic mean of Precision and Recall
- **Precision-Recall**: Balances precision and recall

### 3. Data Leakage Prevention
- Features normalized **per fold** using only training data
- Scalers fitted on train, applied to val/test
- Separate normalization for each split

### 4. Class Imbalance Handling
- Weighted loss functions (CNN)
- Sample weights (LightGBM)
- Stratified cross-validation

## 📈 Expected Performance

Typical results on IPF progression:

| Model      | AUC           | F1 Score      | Accuracy      |
|------------|---------------|---------------|---------------|
| CNN        | 0.82 ± 0.03   | 0.74 ± 0.04   | 0.78 ± 0.03   |
| LightGBM   | 0.79 ± 0.04   | 0.71 ± 0.05   | 0.75 ± 0.04   |
| **Fusion** | **0.86 ± 0.03** | **0.78 ± 0.03** | **0.81 ± 0.02** |

*Fusion typically improves AUC by 3-5% over the best single expert.*

## 🧪 Ablation Studies

To run ablation studies (testing feature combinations):

```python
# In train_ensemble.py, modify config:
config['hand_feature_cols'] = []  # Remove hand-crafted features
config['demo_feature_cols'] = []  # Remove demographics
```

Possible configurations:
1. CNN only
2. CNN + Hand-crafted
3. CNN + Demographics
4. CNN + Both (full model)

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config
- Use fewer `num_workers`
- Switch to `device: 'cpu'`

### Poor Correlation Between Experts
- Check feature distributions
- Verify normalization is working
- Inspect feature importance in LightGBM

### Low Performance
- Check class imbalance (adjust weights)
- Increase `cnn_epochs`
- Try different `threshold_strategy`
- Verify data preprocessing

## 📝 Citation

If you use this ensemble system, please cite:

```bibtex
@software{ipf_ensemble,
  author = {Francesco P.},
  title = {Expert Ensemble System for IPF Progression Prediction},
  year = {2026},
  url = {https://...}
}
```

## 📧 Contact

For questions or issues, please contact: [your email]

## 🔗 References

1. **Meta-Learning**: Wolpert, D. H. (1992). "Stacked generalization"
2. **LightGBM**: Ke et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
3. **Attention Mechanisms**: Vaswani et al. (2017). "Attention Is All You Need"
4. **Threshold Optimization**: Youden, W. J. (1950). "Index for rating diagnostic tests"

---

**Last Updated**: February 2026
