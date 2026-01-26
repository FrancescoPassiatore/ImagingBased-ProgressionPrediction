# Dataset Analysis – FVC at 52 Weeks

## Overview
This document analyzes the distribution and structure of **Forced Vital Capacity at week 52 (FVC@52)** and its relationship with **baseline FVC**, in order to understand why predictive models tend to **collapse toward the mean**.

**Dataset size:** 165 patients

---

## 1. Distribution of True FVC@52

### Descriptive statistics
- **Mean:** 2633 ml  
- **Std:** 849 ml  
- **Min:** 827 ml  
- **Max:** 5768 ml  

### Observations
- Unimodal distribution
- Strong concentration between **2000–3200 ml**
- Long but sparse tails

### Implication
A regression model trained with MAE or MSE is **strongly incentivized to predict values close to the global mean (~2600 ml)**.  
This alone explains why many models converge to near-constant predictions.

---

## 2. Baseline FVC vs FVC@52

### Correlation
```
Corr(baseline_fvc, fvc_52) = 0.945
```

### Observations
- Extremely strong linear relationship
- Most samples lie close to the identity line *(y = x)*
- Deviations from baseline are relatively small

### Interpretation
**FVC@52 is largely determined by baseline FVC**, with only modest variation.  
Predicting FVC@52 directly therefore mostly means learning the baseline and smoothing the residuals.

---

## 3. Distribution of ΔFVC

Defined as:
```
ΔFVC = FVC@52 − baseline_FVC
```

### Descriptive statistics
- **Mean:** −155 ml  
- **Median:** −119 ml  
- **Std:** 280 ml  
- **Min:** −1076 ml  
- **Max:** +555 ml  

### Observations
- Strong peak around **−100 / −200 ml**
- Distribution centered near zero
- Much smaller variance than absolute FVC values

### Interpretation
**ΔFVC captures the true disease progression signal** and operates on a much more learnable scale.

---

## 4. Percentage Change (%ΔFVC)

Defined as:
```
%ΔFVC = (FVC@52 − baseline_FVC) / baseline_FVC × 100
```

### Descriptive statistics
- **Mean:** −5.6%
- **IQR:** [−10.7%, +0.8%]
- **Std:** 11.0%

### Observations
- Clinically interpretable
- Scale-invariant
- More stable than absolute ΔFVC

---

## 5. Correlation Summary

```
Corr(baseline_fvc, fvc_52)    =  0.945
Corr(baseline_fvc, delta_fvc) = -0.139
```

### Interpretation
- **FVC@52 is strongly dependent on baseline**
- **ΔFVC is largely independent of baseline**

This confirms that baseline should act as an anchor, not as the prediction target.

---

## Why Models Collapse Toward the Mean

The collapse toward ~2600 ml is statistically expected because:

1. FVC@52 has a strong global mean
2. Baseline FVC already explains most of the variance
3. Residual changes are small and noisy
4. MAE/MSE heavily penalize large absolute errors

**Optimal solution for the loss:**
```
prediction ≈ mean(FVC@52)
```

This leads to:
- Compressed predictions
- Underestimation of high-baseline patients
- Overestimation of low-baseline patients
- Misleading progression signals

---

## Key Conclusions

- ❌ Predicting **absolute FVC@52** is suboptimal  
- ❌ Collapse toward the mean is statistically expected  
- ✅ **ΔFVC or %ΔFVC** are the correct learning targets  
- ✅ Baseline FVC must be used as an anchor  

---

## Recommended Modeling Strategy

1. Train the model to predict **ΔFVC** or **%ΔFVC**
2. Reconstruct:
   ```
   FVC@52_pred = baseline_FVC + ΔFVC_pred
   ```
3. Use losses defined on relative or delta-based targets
4. Derive progression or risk classification only afterward

---

## Final Takeaway

The observed collapse toward the mean is **not a model bug**.  
It is the **statistically optimal solution** given the original target definition.

**Redefining the target is mandatory.**
