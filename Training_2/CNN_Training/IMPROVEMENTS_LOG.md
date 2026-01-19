# CNN Training Improvements - Focal Loss + Stratified Sampling

## Date: January 18, 2026

## Problem Analysis

Previous training (DenseNet121 + Attention) achieved R² ≈ 0.007 but suffered from:
1. **Mode collapse**: All predictions clustered around -4.0 (range: 0.5)
2. **Data imbalance**: 80% mild progression (-5 to -3), 2% extreme cases
3. **MSE loss bias**: Optimizes for majority, ignores minority classes

## Implemented Solutions

### 1. Focal Loss for Regression
- **FocalMSELoss**: Focuses on hard examples (large errors)
- **FocalHuberLoss**: Combines focal weighting with Huber robustness
- Formula: `Loss = |error|^gamma * base_loss`
- `gamma=2.0`: Higher weight for difficult predictions

### 2. Stratified Batch Sampling
- **StratifiedPatientSampler**: Bins patients by slope quantiles
- Ensures each batch has diverse progression patterns
- Default: 4 bins (quartiles) for balanced representation
- Forces model to learn discrimination across all severity levels

### 3. Enhanced Diagnostics
- **PredictionTracker**: Monitors prediction distribution evolution
  - Mean, std, min, max, quantiles
  - Mode collapse score: `pred_std / true_std`
  - Correlation tracking
  - Residual analysis
- **Batch diversity analysis**: Validates stratified sampling effectiveness
- Automatic plotting of diagnostic metrics

## Configuration

```yaml
loss_type: focal_huber
focal_gamma: 2.0
huber_delta: 1.0
use_stratified_sampling: true
n_strata_bins: 4
```

## Expected Outcomes

### Without Improvements (Baseline)
- R²: +0.007
- Prediction range: 0.5
- Mode collapse score: ~0.15

### With Improvements (Target)
- R²: +0.02 to +0.03
- Prediction range: 2.0+
- Mode collapse score: >0.4
- Better extreme case prediction

## New Output Directories

- Checkpoints: `final_checkpoints_densenet_focal_strat/`
- Predictions: `predictions_densenet_focal_strat/`
- Results: `final_results_densenet_focal_strat/`
- Diagnostics: `diagnostics_densenet_focal_strat/`

## Key Metrics to Monitor

1. **Mode collapse score**: Should improve from 0.15 → 0.4+
2. **Prediction range**: Should expand from 0.5 → 2.0+
3. **R² score**: Target improvement from 0.007 → 0.02+
4. **Batch diversity**: Standard deviation within batches should be >2.0

## Next Steps

1. Complete 5-fold training with new configuration
2. Compare aggregate metrics vs baseline
3. Analyze diagnostic plots for mode collapse reduction
4. If successful, proceed to MLP corrector
5. If insufficient, consider additional strategies:
   - Focal loss gamma tuning (try 1.5, 2.5)
   - More strata bins (try 5-6)
   - Combined with class weighting
