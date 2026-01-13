# Quick Start Guide

## 🚀 One-Command Execution

To run everything at once:

```bash
cd Training/Retraining_cross_validation
python run_all.py
```

This will:
1. ✅ Check prerequisites
2. ✅ Ask for confirmation
3. ✅ Run all 4 steps sequentially
4. ✅ Show progress and timing
5. ✅ Generate final summary

---

## 📋 Step-by-Step Execution

If you prefer to run steps individually:

### Step 1: Train CNN
```bash
python train_cnn_kfold.py
```

### Step 2: Train Correctors
```bash
python train_corrector_kfold.py
```

### Step 3: Predict FVC@52
```bash
python predict_fvc52.py
```

### Step 4: Compare Results
```bash
python compare_approaches.py
```

---

## ⚡ Advanced Usage

### Skip CNN Training (use existing models)
```bash
python run_all.py --skip-cnn
```

### Run specific steps only
```bash
python run_all.py --steps 3 4
```

### Run only comparison
```bash
python run_all.py --steps 4
```

---

## 📊 Check Results

After completion, check these files:

### Key Results Files
- `results/fvc52_summary.csv` - **Main results table**
- `results/statistical_comparisons.csv` - **Statistical tests**
- `plots/comprehensive_comparison.png` - **Main visualization**

### Per-Approach Predictions
- `results/cnn_only_all_folds_fvc52_predictions.csv`
- `results/cnn_handcrafted_all_folds_fvc52_predictions.csv`
- `results/cnn_demographics_all_folds_fvc52_predictions.csv`
- `results/cnn_full_all_folds_fvc52_predictions.csv`

---

## 🐛 Common Issues

### GPU Out of Memory
**Solution:** Reduce batch size in config
```python
# In train_cnn_kfold.py
CONFIG['batch_size'] = 2  # Reduce from 4

# In train_corrector_kfold.py
CONFIG['batch_size'] = 16  # Reduce from 32
```

### Missing Files
**Solution:** Check paths in CONFIG
```python
'csv_path': 'Training/CNN_Slope_Prediction/train_with_coefs.csv',
'features_path': 'Training/CNN_Slope_Prediction/patient_features.csv',
'npy_dir': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy'
```

### Slow Training
**Solution:** Use GPU or reduce data
- Ensure CUDA is available: `torch.cuda.is_available()`
- Reduce number of epochs
- Use smaller model backbone

---

## 📈 Expected Output Structure

```
Retraining_cross_validation/
├── checkpoints/
│   ├── cnn_fold0.pth                          # CNN weights (5 files)
│   ├── slope_scaler_fold0.pkl                 # Slope scalers (5 files)
│   ├── cnn_only_fold0.pth                     # Corrector weights (20 files)
│   └── cnn_only_scaler_fold0.pkl              # Feature scalers (20 files)
├── results/
│   ├── kfold_splits.pkl                       # K-fold splits
│   ├── corrector_summary.csv                  # Corrector performance
│   ├── fvc52_summary.csv                      # ⭐ MAIN RESULTS
│   ├── statistical_comparisons.csv            # Statistical tests
│   └── {approach}_all_folds_fvc52_predictions.csv  # Predictions (4 files)
└── plots/
    ├── cnn_training_curves.png                # CNN training
    ├── corrector_comparison.png               # Corrector comparison
    ├── comprehensive_comparison.png           # ⭐ MAIN VISUALIZATION
    └── fold_consistency.png                   # Fold-wise performance
```

---

## ⏱️ Time Estimates

| Step | GPU | CPU |
|------|-----|-----|
| 1. CNN Training | 2-4h | 20-30h |
| 2. Corrector Training | 1-2h | 8-12h |
| 3. FVC@52 Prediction | 30m | 2-3h |
| 4. Comparison | 5m | 5m |
| **Total** | **~4-7h** | **~30-45h** |

---

## ✅ Validation Checklist

After completion, verify:

- [ ] 25 checkpoint files in `checkpoints/`
- [ ] `fvc52_summary.csv` exists with 4 rows
- [ ] All 3 plots generated in `plots/`
- [ ] Statistical comparisons show p-values
- [ ] Best approach is identified in summary

---

## 🎯 What to Report

In your paper/report, include:

1. **Table:** `fvc52_summary.csv` showing MAE, RMSE, R² for all approaches
2. **Figure:** `comprehensive_comparison.png` with all visualizations
3. **Statistics:** `statistical_comparisons.csv` showing pairwise tests
4. **Best Model:** Approach with lowest MAE or highest R²

Example:
> "We evaluated 4 approaches using 5-fold cross-validation. The CNN + Handcrafted + Demographics approach achieved the best performance with MAE=145.3±12.1 ml, RMSE=189.7±15.4 ml, and R²=0.752±0.031."

---

## 💡 Tips

- **Save intermediate results:** Models are saved automatically after each fold
- **Monitor training:** Watch loss curves for overfitting
- **Check data quality:** Verify FVC measurements are reasonable
- **Use tensorboard:** Add logging for better monitoring (optional)
- **GPU memory:** Close other applications before training

---

Need help? Check the main README.md for detailed documentation!
