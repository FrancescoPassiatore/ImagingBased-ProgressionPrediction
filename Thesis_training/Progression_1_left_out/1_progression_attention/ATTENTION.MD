# Attention-based MIL for IPF Progression Prediction

This implementation provides attention-based Multiple Instance Learning (MIL) as an alternative to max pooling for aggregating CT slice features in IPF progression prediction.

## 📋 Overview

### Why Attention MIL?

**Max Pooling Limitations:**
- Selects only the "worst" slice (highest activation)
- Discards information from other slices
- No interpretability about which slices matter
- May be sensitive to outliers

**Attention MIL Advantages:**
- **Weighted aggregation**: All slices contribute based on learned importance
- **Interpretability**: Attention weights show which slices are diagnostically relevant
- **Flexibility**: Can capture multiple important regions across slices
- **Robustness**: Less sensitive to single outlier slices

## 🏗️ Architecture

### 1. **Gated Attention MIL** (Recommended starting point)

Based on "Attention-based Deep Multiple Instance Learning" (Ilse et al., 2018)

```
CT Slices (n × 2048) 
    ↓
Gated Attention Mechanism:
  - Attention_V: Linear(2048 → 128) → Tanh
  - Attention_U: Linear(2048 → 128) → Sigmoid  
  - Gated: V ⊙ U (element-wise multiplication)
  - Weights: Linear(128 → 1) → Softmax
    ↓
Weighted Pooling → (1 × 2048)
    ↓
Feature Branches (optional):
  - CNN Branch: FC(2048 → 512)
  - Patient Branch: FC(hand+demo → 64)
    ↓
Classification Head → Prediction
```

**Hyperparameters:**
- `attention_hidden_dim`: 128 (recommended)
- `dropout`: 0.25 for attention, 0.4 for classifier

### 2. **Multi-Head Attention MIL**

Multiple attention mechanisms capture different aspects:

```
CT Slices (n × 2048)
    ↓
4 Parallel Attention Heads (each with gated attention)
    ↓
Concatenate → (4 × 2048)
    ↓
Project → (2048)
    ↓
[Rest same as gated attention]
```

**Hyperparameters:**
- `num_attention_heads`: 4 or 8
- `attention_hidden_dim`: 128 per head

## 🚀 Quick Start

### Option 1: Quick Comparison (Single Fold)

```python
from quick_attention_comparison import quick_single_fold_comparison

# Test on one fold to see if attention helps
results, summary = quick_single_fold_comparison(
    features_df=your_features_df,
    fold_data=your_fold_data,
    config=your_config,
    pooling_types=['max', 'gated', 'multihead'],
    save_dir=Path('./results')
)
```

**Expected Runtime:** ~15-30 minutes per pooling method (50 epochs)

### Option 2: Full Ablation Study

```python
from ablation_study_with_attention import run_comprehensive_ablation_study

# Test all combinations of pooling + features
all_results = run_comprehensive_ablation_study(
    slice_features_df=slice_features_df,
    patient_features_df=patient_features_df,
    kfold_splits=kfold_splits,
    base_config=BASE_CONFIG,
    results_base_dir=results_dir,
    pooling_configs_to_test=['max_pooling', 'gated_attention'],
    feature_configs_to_test=['cnn_only', 'full']
)
```

**Expected Runtime:** ~2-4 hours for 2 pooling × 2 features × 5 folds

### Option 3: Use in Your Existing Code

Just replace your model initialization:

```python
# OLD (max pooling)
from model_train import ProgressionPredictionModel

model = ProgressionPredictionModel(
    cnn_feature_dim=2048,
    pooling_type='max',
    ...
)

# NEW (gated attention)
from attention_mil_model import AttentionMILProgressionModel

model = AttentionMILProgressionModel(
    cnn_feature_dim=2048,
    attention_type='gated',
    attention_hidden_dim=128,
    ...
)
```

Everything else (training, evaluation) stays the same!

## 📊 Expected Results

Based on MIL literature, you might expect:

### Conservative Estimate:
- **AUC improvement**: +1-3% over max pooling
- **Interpretability**: Attention weights for visualization
- **Robustness**: More stable across folds

### Optimistic Estimate (if max pooling was suboptimal):
- **AUC improvement**: +5-10%
- **Better calibration**: More reliable probabilities
- **Improved precision/recall balance**

### When Attention Helps Most:
✅ Variable-length sequences (different # slices per patient)  
✅ Heterogeneous disease patterns across slices  
✅ When multiple slices contain complementary information  

### When Attention May Not Help:
❌ If max pooling already captures the relevant information  
❌ Very small datasets (<100 patients)  
❌ If the "worst slice" truly determines outcome  

## 🔍 Interpreting Attention Weights

### Visualize Important Slices

```python
from attention_mil_model import visualize_attention_weights

# Get predictions with attention weights
model.eval()
with torch.no_grad():
    results = model.predict_proba(
        slice_features, lengths, patient_features
    )

# Visualize for a patient
visualize_attention_weights(
    attention_weights=results['attention_weights'][patient_idx].cpu().numpy(),
    patient_id=patient_ids[patient_idx],
    num_slices=results['num_slices'][patient_idx],
    save_path='attention_weights.png'
)
```

### Interpretation:
- **High attention** = Model considers this slice important for prediction
- **Low attention** = Slice provides little additional information
- **Multi-modal distribution** = Multiple important regions

### Clinical Insights:
- Do high-attention slices correspond to known disease patterns?
- Are certain anatomical regions consistently attended to?
- How does attention relate to radiologist assessments?

## ⚙️ Hyperparameter Tuning

### Start Here (Default):
```python
config = {
    'attention_hidden_dim': 128,
    'attention_type': 'gated',
    'dropout': 0.4,
    'learning_rate': 3.86e-05
}
```

### If Overfitting:
```python
config = {
    'attention_hidden_dim': 64,  # Smaller
    'dropout': 0.5,  # Higher
    'weight_decay': 0.05  # Higher
}
```

### If Underfitting:
```python
config = {
    'attention_hidden_dim': 256,  # Larger
    'num_attention_heads': 8,  # More heads
    'dropout': 0.3,  # Lower
}
```

### For Multi-Head:
```python
# More heads = more parameters
# Start with 4, try 8 if dataset is large
config = {
    'num_attention_heads': 4,
    'attention_hidden_dim': 128
}
```

## 📁 File Structure

```
attention_mil_model.py          # Core attention implementations
  ├── GatedAttentionMIL         # Gated attention mechanism
  ├── MultiHeadAttentionMIL     # Multi-head attention
  └── AttentionMILProgressionModel  # Complete model

ablation_study_with_attention.py  # Full ablation study
  └── Compares pooling × features × folds

quick_attention_comparison.py   # Quick single-fold comparison
  └── Fast testing & visualization

utilities.py                    # Data loading (unchanged)
model_train.py                  # Training utilities (unchanged)
```

## 🎯 Recommended Workflow

### Phase 1: Quick Test (1 hour)
```bash
# Test on one fold with all three pooling methods
python quick_attention_comparison.py --pooling all --fold 0 --epochs 50
```

**Look for:**
- Is gated/multihead better than max?
- How much better? (>2% = promising)
- Training curves: smoother = better

### Phase 2: Validation (3 hours)
If Phase 1 shows promise:
```python
# Test best method on all 5 folds
pooling_to_test = ['max_pooling', 'gated_attention']  # Winner from Phase 1
features_to_test = ['full']  # Your best feature config

run_comprehensive_ablation_study(...)
```

### Phase 3: Final Comparison (4 hours)
```python
# Full grid: pooling × features
pooling_to_test = ['max_pooling', 'gated_attention', 'multihead_attention_4']
features_to_test = ['cnn_only', 'cnn_hand', 'cnn_demo', 'full']
```

## 📈 Evaluation Checklist

When comparing results:

### Primary Metrics:
- [ ] **Test AUC**: Main performance metric
- [ ] **Test F1**: Balance of precision/recall
- [ ] **Std deviation across folds**: Consistency

### Secondary Metrics:
- [ ] Training curves: Smoother = better regularization
- [ ] Convergence speed: Faster = more efficient
- [ ] Attention weight distribution: Should make clinical sense

### Red Flags:
⚠️ Attention completely uniform → Not learning anything useful  
⚠️ Attention on only 1-2 slices → Might as well use max pooling  
⚠️ High variance across folds → Unstable, needs more regularization  

## 🔧 Troubleshooting

### "Attention weights are all similar"
**Solution:** Reduce `attention_hidden_dim` or increase dropout in attention

### "Attention focuses on only 1 slice"
**Possible causes:**
- Dataset has strong outliers
- Max pooling might actually be better for this task
- Try multi-head to capture more diverse patterns

### "Worse than max pooling"
**Check:**
- Are you overfitting? (val curves)
- Is attention hidden dim too large?
- Try more regularization (dropout, weight decay)

### "Training is slow"
**Solutions:**
- Reduce `num_attention_heads` (8 → 4)
- Reduce `attention_hidden_dim` (256 → 128)
- Batch size larger if GPU memory allows

## 📚 References

**Gated Attention:**
- Ilse et al. "Attention-based Deep Multiple Instance Learning" (ICML 2018)

**MIL for Medical Imaging:**
- Campanella et al. "Clinical-grade computational pathology using weakly supervised deep learning on whole slide images" (Nature Medicine 2019)

**IPF Progression:**
- Your existing references on IPF and CT imaging

## 💡 Tips for Success

1. **Start simple**: Gated attention with default hyperparameters
2. **Compare fairly**: Same features, same epochs, same folds
3. **Visualize attention**: Does it make clinical sense?
4. **Check consistency**: Low variance across folds = reliable
5. **Don't overfit**: Attention adds parameters, watch validation curves

## 🎓 Understanding the Math

### Gated Attention Formula:

```
For each slice i with features h_i:

a_i = w^T (tanh(V·h_i) ⊙ sigmoid(U·h_i))

α_i = exp(a_i) / Σ_j exp(a_j)  [softmax]

h_pooled = Σ_i α_i · h_i  [weighted sum]
```

Where:
- `V, U, w`: Learned parameters
- `⊙`: Element-wise multiplication (gating)
- `α_i`: Attention weight for slice i

**Why gating?**
- `tanh(V·h_i)`: Captures feature content
- `sigmoid(U·h_i)`: Acts as learned gate (0-1)
- Product: Amplifies important features, suppresses noise

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify your feature dimensions match
3. Ensure normalization is applied correctly
4. Compare training curves (included in outputs)

---

**Good luck! The attention mechanism should give you both better performance and better interpretability.** 🚀