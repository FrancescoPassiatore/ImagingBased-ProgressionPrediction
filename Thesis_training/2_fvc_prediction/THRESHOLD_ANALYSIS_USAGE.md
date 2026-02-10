# Threshold Analysis Usage Guide

The `assess_progression_from_predictions.py` script now supports threshold optimization to find the best progression classification threshold.

## Features

- **Fixed Threshold Mode**: Use a single threshold (e.g., 10%) for progression classification
- **Threshold Analysis Mode**: Test multiple thresholds and find the optimal one
- **Multiple Optimization Criteria**: Choose between F1-score, accuracy, Youden's J, or balanced approach

## Usage Examples

### 1. Fixed Threshold (Default 10%)

```bash
python assess_progression_from_predictions.py
```

Or specify a custom threshold:

```bash
python assess_progression_from_predictions.py --threshold 15.0
```

### 2. Threshold Analysis

Run threshold optimization from 0% to 30% in 1% steps (default):

```bash
python assess_progression_from_predictions.py --analyze-thresholds
```

### 3. Custom Threshold Range

Analyze thresholds from 5% to 25% in 0.5% steps:

```bash
python assess_progression_from_predictions.py --analyze-thresholds --threshold-min 5.0 --threshold-max 25.0 --threshold-step 0.5
```

### 4. Different Optimization Criteria

**Optimize for F1-score (default):**
```bash
python assess_progression_from_predictions.py --analyze-thresholds --criterion f1
```

**Optimize for accuracy:**
```bash
python assess_progression_from_predictions.py --analyze-thresholds --criterion accuracy
```

**Optimize for Youden's J statistic (sensitivity + specificity - 1):**
```bash
python assess_progression_from_predictions.py --analyze-thresholds --criterion youdens_j
```

**Optimize for balanced metric (average of F1, accuracy, and Youden's J):**
```bash
python assess_progression_from_predictions.py --analyze-thresholds --criterion balanced
```

## Output Files

### Fixed Threshold Mode

For each fold in each ablation experiment:
- `progression_assessment_detailed.csv` - Per-patient results
- `progression_assessment_metrics.csv` - Summary metrics
- `progression_assessment_plot.png` - Visualization (confusion matrix, distributions, scatter)

Aggregate across folds:
- `progression_assessment_aggregate.csv` - Mean, std, min, max for each metric

### Threshold Analysis Mode

Additional files per fold:
- `threshold_analysis.csv` - Metrics for each threshold tested
- `optimal_threshold.csv` - Optimal threshold and corresponding metrics
- `threshold_analysis_plot.png` - 4-panel plot showing:
  - Metrics vs threshold
  - Sensitivity/specificity vs threshold
  - Number of predicted progressions vs threshold
  - Key metrics comparison
- `threshold_comparison_plot.png` - **NEW**: Side-by-side comparison of 10% threshold vs optimal threshold:
  - Confusion matrices (reference, optimal, difference)
  - Metrics comparison bars
  - Sensitivity & specificity comparison
  - Summary comparison table with deltas

Aggregate across experiments:
- `optimal_thresholds_summary.csv` - Optimal threshold for each fold with F1 and accuracy

## Threshold Analysis Plots

### 1. Threshold Analysis Plot (`threshold_analysis_plot.png`)

The threshold analysis visualization includes:

1. **Classification Metrics vs Threshold**: Shows accuracy, precision, recall, and F1-score across thresholds
2. **Sensitivity/Specificity vs Threshold**: Shows sensitivity, specificity, and Youden's J statistic
3. **Predicted Progressions vs Threshold**: Shows how many patients are classified as progressed at each threshold
4. **Key Metrics Comparison**: Compares F1, accuracy, and Youden's J for threshold selection

### 2. Threshold Comparison Plot (`threshold_comparison_plot.png`) - **NEW**

Direct comparison between 10% reference threshold and optimal threshold:

**Top Row:**
1. **Reference Confusion Matrix** (10% threshold) - Shows baseline performance
2. **Optimal Confusion Matrix** - Shows performance at optimal threshold
3. **Difference Matrix** - Highlights changes (Optimal - Reference) with color coding

**Bottom Row:**
4. **Key Metrics Comparison** - Bar chart comparing accuracy, precision, recall, F1
5. **Sensitivity & Specificity** - Bar chart showing diagnostic performance
6. **Summary Table** - Complete metrics with delta (Δ) values showing improvement/decline

This plot makes it easy to see:
- How many patients switch classification categories
- Which metrics improve and by how much
- Whether the optimal threshold is worth using over the standard 10%

## Optimization Criteria Explained

- **F1-score**: Harmonic mean of precision and recall; good balance for imbalanced classes
- **Accuracy**: Overall correctness; works well when classes are balanced
- **Youden's J**: Sensitivity + Specificity - 1; maximizes both true positive and true negative rates
- **Balanced**: Average of F1, accuracy, and Youden's J; considers multiple aspects

## Example Workflow

1. **Initial assessment with default threshold:**
   ```bash
   python assess_progression_from_predictions.py
   ```

2. **Explore threshold space:**
   ```bash
   python assess_progression_from_predictions.py --analyze-thresholds
   ```

3. **Review `threshold_analysis_plot.png` for each fold to understand threshold behavior**

4. **If needed, narrow the search range:**
   ```bash
   python assess_progression_from_predictions.py --analyze-thresholds --threshold-min 8.0 --threshold-max 15.0 --threshold-step 0.25
   ```

5. **Compare different criteria:**
   ```bash
   # Try F1
   python assess_progression_from_predictions.py --analyze-thresholds --criterion f1
   
   # Try Youden's J
   python assess_progression_from_predictions.py --analyze-thresholds --criterion youdens_j
   ```

## Interpreting Results

- **Low threshold** (e.g., 5%): More patients classified as progressed → higher recall, lower precision
- **High threshold** (e.g., 20%): Fewer patients classified as progressed → lower recall, higher precision
- **Optimal threshold**: Balances sensitivity and specificity based on your chosen criterion

Look for:
- Where F1-score peaks (good overall balance)
- Where Youden's J peaks (equal emphasis on sensitivity and specificity)
- Where accuracy peaks (if classes are balanced)
- Clinical considerations (e.g., prefer higher sensitivity to catch all progressions)
