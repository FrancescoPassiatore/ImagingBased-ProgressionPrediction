# 5-Fold Cross-Validation Retraining Framework

This folder contains a complete framework for retraining and comparing 4 different approaches to IPF progression prediction using 5-fold cross-validation.

## 📁 Folder Structure

```
Retraining_cross_validation/
├── utilities.py                    # Core utilities, data loaders, models
├── train_cnn_kfold.py             # Step 1: Train CNN slope predictor
├── train_corrector_kfold.py       # Step 2: Train slope correctors (4 approaches)
├── predict_fvc52.py               # Step 3: Predict FVC@52 and evaluate
├── compare_approaches.py          # Step 4: Compare all approaches
├── README.md                       # This file
├── checkpoints/                    # Model weights (created during training)
├── results/                        # Predictions and metrics (created during training)
└── plots/                          # Visualizations (created during training)
```

## 🎯 Four Approaches

1. **CNN Only**: Predict slope using CNN alone, no correction
2. **CNN + Handcrafted**: CNN + 9 handcrafted imaging features
3. **CNN + Demographics**: CNN + 3 demographic features (age, sex, smoking)
4. **CNN + Handcrafted + Demographics**: CNN + all 12 features

## 🚀 Usage

### Step 1: Train CNN Slope Predictor (5-Fold CV)

```bash
python train_cnn_kfold.py
```

**What it does:**
- Creates 5-fold splits of patients
- Trains EfficientNet-B0 CNN to predict per-slice slopes
- Saves 5 CNN models (one per fold)
- Saves slope normalization scalers
- Generates training curves

**Output:**
- `checkpoints/cnn_fold{0-4}.pth` - CNN weights
- `checkpoints/slope_scaler_fold{0-4}.pkl` - Slope scalers
- `results/kfold_splits.pkl` - Train/val splits
- `results/fold{0-4}_cnn_results.pkl` - Training history
- `plots/cnn_training_curves.png` - Visualization

**Expected time:** ~2-4 hours (depending on GPU)

---

### Step 2: Train Slope Correctors (4 Approaches × 5 Folds)

```bash
python train_corrector_kfold.py
```

**What it does:**
- Loads trained CNN models from Step 1
- Extracts CNN slopes for all patients
- Trains 4 different corrector models for each fold
- Compares corrector performance

**Output:**
- `checkpoints/{approach}_fold{0-4}.pth` - Corrector weights (20 files total)
- `checkpoints/{approach}_scaler_fold{0-4}.pkl` - Feature scalers (20 files)
- `results/fold{0-4}_{approach}_results.pkl` - Training history (20 files)
- `results/corrector_summary.csv` - Performance summary
- `plots/corrector_comparison.png` - Comparison plot

**Expected time:** ~1-2 hours

---

### Step 3: Predict FVC@52 and Evaluate

```bash
python predict_fvc52.py
```

**What it does:**
- Loads CNN and corrector models for all approaches
- Predicts FVC at week 52 for validation patients
- Computes metrics (MAE, RMSE, R², % error)
- Aggregates results across folds

**Output:**
- `results/{approach}_fold{0-4}_fvc52_predictions.csv` - Per-fold predictions (20 files)
- `results/{approach}_all_folds_fvc52_predictions.csv` - Combined predictions (4 files)
- `results/fvc52_summary.csv` - Performance summary

**Expected time:** ~30 minutes

---

### Step 4: Compare All Approaches

```bash
python compare_approaches.py
```

**What it does:**
- Loads predictions from all approaches
- Performs statistical comparisons (paired t-tests)
- Creates comprehensive visualizations
- Identifies best approach

**Output:**
- `results/statistical_comparisons.csv` - Pairwise statistical tests
- `plots/comprehensive_comparison.png` - Main comparison dashboard
- `plots/fold_consistency.png` - Performance across folds

**Expected time:** ~5 minutes

---

## 📊 Outputs Explained

### Metrics

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and true FVC@52 (ml)
- **RMSE (Root Mean Squared Error)**: Root of average squared errors (ml)
- **R² (R-squared)**: Proportion of variance explained (0-1, higher is better)
- **% Error**: Mean percentage error relative to true FVC

### Visualizations

**comprehensive_comparison.png** includes:
1. Mean errors (MAE/RMSE) across folds
2. R² performance comparison
3. Percentage error comparison
4. Predicted vs true scatter plot
5. Error distribution (violin plot)
6. Absolute error distribution (box plot)
7. Cumulative error distribution

**fold_consistency.png** shows:
- How each metric varies across the 5 folds
- Consistency of each approach

---

## 🔧 Configuration

Edit the `CONFIG` dictionary in each script to customize:

### train_cnn_kfold.py
```python
CONFIG = {
    'n_folds': 5,           # Number of folds
    'n_epochs': 50,         # Max epochs per fold
    'patience': 10,         # Early stopping patience
    'batch_size': 4,        # Patients per batch
    'lr': 1e-4,            # Learning rate
    'backbone': 'efficientnet_b0'
}
```

### train_corrector_kfold.py
```python
CONFIG = {
    'n_epochs': 100,
    'patience': 15,
    'batch_size': 32,
    'lr': 1e-3
}
```

### predict_fvc52.py
```python
CONFIG = {
    'target_week': 52      # Week to predict FVC at
}
```

---

## 📈 Expected Results

Based on similar studies, you should expect:

| Approach | MAE (ml) | R² |
|----------|----------|-----|
| CNN Only | 150-200 | 0.60-0.70 |
| CNN + Handcrafted | 130-170 | 0.65-0.75 |
| CNN + Demographics | 140-180 | 0.62-0.72 |
| CNN + HF + Demo | 120-160 | 0.70-0.80 |

*Note: Actual results depend on your data quality and patient characteristics*

---

## 🐛 Troubleshooting

### Out of Memory Error
- Reduce `batch_size` in config
- Use smaller backbone (e.g., `efficientnet_b0` → `resnet18`)

### CUDA Not Available
- Models automatically fall back to CPU
- Training will be slower (~10x)

### Missing Data
- Check that CSV paths are correct
- Verify NPY directory structure matches expected format

### Poor Performance
- Check data preprocessing
- Verify slope normalization is working
- Inspect training curves for overfitting

---

## 📝 Notes

- All scripts use the same K-fold splits for fair comparison
- Models are saved at best validation loss (early stopping)
- Feature normalization uses training data statistics only
- Slope predictions are normalized during CNN training

---

## 🔬 Citation

If you use this framework in your research, please cite:
```
[Your paper citation here]
```

---

## 📧 Contact

For questions or issues, please contact:
- Francesco P.
- Email: [your-email]

---

## ✅ Checklist

Before running:
- [ ] CSV files exist: `train_with_coefs.csv`, `patient_features.csv`
- [ ] NPY directory contains patient scans
- [ ] CUDA is available (optional but recommended)
- [ ] Required packages installed: `torch`, `timm`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`

After completion:
- [ ] `checkpoints/` contains 25 model files (5 CNN + 20 correctors)
- [ ] `results/` contains predictions and metrics
- [ ] `plots/` contains visualizations
- [ ] Review `fvc52_summary.csv` for final results

---

**Total expected runtime:** ~4-6 hours (with GPU)

Good luck with your retraining! 🚀
