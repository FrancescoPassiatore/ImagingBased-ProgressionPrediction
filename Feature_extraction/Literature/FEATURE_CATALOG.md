# Complete Feature Catalog - CT Image Feature Extraction

## Overview
The improved extraction code generates **~150+ features per patient**, organized into:
- 5 global/metadata features
- ~37 features per region × 4 regions = ~148 regional features
- **Total: ~153 features**

---

## 1. GLOBAL METADATA FEATURES (5 features)

| Feature Name | Type | Description | Clinical Significance |
|-------------|------|-------------|----------------------|
| `patient_id` | ID | Unique patient identifier | Link to clinical data |
| `data_split` | Categorical | 'train' or 'test' | Dataset provenance |
| `total_slices` | Integer | Total number of CT slices in scan | Scan coverage/quality |
| `slice_thickness` | Float (mm) | Distance between slices | Spatial resolution |
| `pixel_spacing` | Float (mm) | Physical size of each pixel | In-plane resolution |

---

## 2. VOLUME FEATURES (1 feature)

| Feature Name | Type | Description | Clinical Significance |
|-------------|------|-------------|----------------------|
| `approx_lung_volume` | Float (mm³) | Total lung volume across all slices | Key FVC predictor; declining volume = disease progression |

**Calculation:**
```
volume = sum(lung_pixels_per_slice × pixel_spacing² × slice_thickness)
```

---

## 3. REGIONAL FEATURES (~37 features × 4 regions = ~148 features)

### Regions Analyzed:
1. **upper** - Upper lung (15-35th percentile of slices)
2. **middle** - Middle lung (35-65th percentile)
3. **lower** - Lower lung (65-85th percentile)
4. **full** - Entire lung (all slices)

### For EACH region, the following 37 features are extracted:

---

#### 3.1 Slice Count (1 feature per region)

| Feature Pattern | Example | Description |
|----------------|---------|-------------|
| `{region}_num_slices` | `lower_num_slices` | Number of slices in this region |

**Examples:**
- `upper_num_slices` = 27
- `middle_num_slices` = 40
- `lower_num_slices` = 27
- `full_num_slices` = 135

---

#### 3.2 Lung Segmentation Features (4 features per region)

| Feature Pattern | Example | Unit | Description | Clinical Significance |
|----------------|---------|------|-------------|----------------------|
| `{region}_lung_pixels_mean` | `lower_lung_pixels_mean` | pixels | Average lung area | Lung size in region |
| `{region}_lung_pixels_std` | `lower_lung_pixels_std` | pixels | Variability in lung area | Consistency of segmentation |
| `{region}_lung_pixels_min` | `lower_lung_pixels_min` | pixels | Smallest lung area | Minimum lung extent |
| `{region}_lung_pixels_max` | `lower_lung_pixels_max` | pixels | Largest lung area | Maximum lung extent |

**Clinical Note:** Lower lung typically shows larger lung area than upper due to anatomy.

---

#### 3.3 Lung Area Ratio Features (4 features per region)

| Feature Pattern | Example | Unit | Description |
|----------------|---------|------|-------------|
| `{region}_lung_area_ratio_mean` | `lower_lung_area_ratio_mean` | ratio (0-1) | Mean proportion of image that is lung |
| `{region}_lung_area_ratio_std` | `lower_lung_area_ratio_std` | ratio | Variability in lung proportion |
| `{region}_lung_area_ratio_min` | `lower_lung_area_ratio_min` | ratio | Minimum lung proportion |
| `{region}_lung_area_ratio_max` | `lower_lung_area_ratio_max` | ratio | Maximum lung proportion |

**Example Values:**
- Healthy: 0.3-0.5 (30-50% of image is lung)
- Fibrotic: May be lower due to tissue collapse

---

#### 3.4 Tissue/Fibrosis Features (4 features per region)

| Feature Pattern | Example | Unit | Description | Clinical Significance |
|----------------|---------|------|-------------|----------------------|
| `{region}_num_tissue_pixels_mean` | `lower_num_tissue_pixels_mean` | pixels | Average high-attenuation pixels | **Direct fibrosis marker** |
| `{region}_num_tissue_pixels_std` | `lower_num_tissue_pixels_std` | pixels | Variability in tissue pixels | Disease heterogeneity |
| `{region}_num_tissue_pixels_min` | `lower_num_tissue_pixels_min` | pixels | Minimum tissue pixels | Least affected slice |
| `{region}_num_tissue_pixels_max` | `lower_num_tissue_pixels_max` | pixels | Maximum tissue pixels | Most affected slice |

**Key Insight:** High-attenuation (bright) pixels in lung regions indicate fibrotic tissue.

---

#### 3.5 Tissue-by-Total Ratio (4 features per region)

| Feature Pattern | Example | Unit | Description |
|----------------|---------|------|-------------|
| `{region}_tissue_by_total_mean` | `lower_tissue_by_total_mean` | ratio (0-1) | Tissue pixels / Total image pixels |
| `{region}_tissue_by_total_std` | `lower_tissue_by_total_std` | ratio | Variability in tissue/total ratio |
| `{region}_tissue_by_total_min` | `lower_tissue_by_total_min` | ratio | Minimum tissue/total ratio |
| `{region}_tissue_by_total_max` | `lower_tissue_by_total_max` | ratio | Maximum tissue/total ratio |

**Clinical Use:** Normalized measure of fibrotic burden per image.

---

#### 3.6 Tissue-by-Lung Ratio (4 features per region) ⭐ MOST IMPORTANT

| Feature Pattern | Example | Unit | Description | Clinical Significance |
|----------------|---------|------|-------------|----------------------|
| `{region}_tissue_by_lung_mean` | `lower_tissue_by_lung_mean` | ratio (0-1) | **Tissue pixels / Lung pixels** | **PRIMARY FIBROSIS METRIC** |
| `{region}_tissue_by_lung_std` | `lower_tissue_by_lung_std` | ratio | Variability in fibrosis | Disease heterogeneity |
| `{region}_tissue_by_lung_min` | `lower_tissue_by_lung_min` | ratio | Minimum fibrosis level | Best-case scenario |
| `{region}_tissue_by_lung_max` | `lower_tissue_by_lung_max` | ratio | Maximum fibrosis level | Worst-case scenario |

**Example Values:**
- Healthy: 0.05-0.15 (5-15% of lung is dense tissue)
- Mild fibrosis: 0.15-0.30
- Severe fibrosis: 0.30-0.60+

**Why This Matters:**
- `lower_tissue_by_lung_mean > upper_tissue_by_lung_mean` → Classic fibrosis pattern
- High `tissue_by_lung_std` → Heterogeneous disease (patchy fibrosis)
- Increasing over time → Disease progression

---

#### 3.7 Histogram Mean Features (4 features per region)

| Feature Pattern | Example | Unit | Description |
|----------------|---------|------|-------------|
| `{region}_mean_mean` | `lower_mean_mean` | HU* | Average pixel intensity in lung region |
| `{region}_mean_std` | `lower_mean_std` | HU | Variability in mean intensity |
| `{region}_mean_min` | `lower_mean_min` | HU | Minimum mean intensity |
| `{region}_mean_max` | `lower_mean_max` | HU | Maximum mean intensity |

*HU = Hounsfield Units (CT attenuation values)

**Clinical Interpretation:**
- Normal lung: -700 to -900 HU (very dark, air-filled)
- Fibrotic tissue: -200 to +50 HU (brighter, denser)
- Higher mean → More fibrosis

---

#### 3.8 Histogram Standard Deviation Features (4 features per region)

| Feature Pattern | Example | Unit | Description |
|----------------|---------|------|-------------|
| `{region}_std_mean` | `lower_std_mean` | HU | Average of pixel intensity variation |
| `{region}_std_std` | `lower_std_std` | HU | Variability in intensity variation |
| `{region}_std_min` | `lower_std_min` | HU | Minimum intensity variation |
| `{region}_std_max` | `lower_std_max` | HU | Maximum intensity variation |

**Clinical Use:** Measures texture heterogeneity within lung.

---

#### 3.9 Histogram Median Features (4 features per region)

| Feature Pattern | Example | Unit | Description |
|----------------|---------|------|-------------|
| `{region}_median_mean` | `lower_median_mean` | HU | Average median intensity |
| `{region}_median_std` | `lower_median_std` | HU | Variability in median |
| `{region}_median_min` | `lower_median_min` | HU | Minimum median |
| `{region}_median_max` | `lower_median_max` | HU | Maximum median |

**Why Median?** More robust to outliers than mean.

---

#### 3.10 Histogram Skewness Features (4 features per region)

| Feature Pattern | Example | Unit | Description | Clinical Significance |
|----------------|---------|------|-------------|----------------------|
| `{region}_skewness_mean` | `lower_skewness_mean` | dimensionless | Average distribution asymmetry | Shape of intensity distribution |
| `{region}_skewness_std` | `lower_skewness_std` | dimensionless | Variability in skewness | Consistency of distribution shape |
| `{region}_skewness_min` | `lower_skewness_min` | dimensionless | Minimum skewness | Most symmetric slice |
| `{region}_skewness_max` | `lower_skewness_max` | dimensionless | Maximum skewness | Most asymmetric slice |

**Interpretation:**
- Skewness < 0: Left-skewed (tail on left) → More bright pixels
- Skewness ≈ 0: Symmetric distribution
- Skewness > 0: Right-skewed (tail on right) → More dark pixels

**Clinical:** Fibrotic lungs show different skewness patterns than healthy lungs.

---

#### 3.11 Histogram Kurtosis Features (4 features per region)

| Feature Pattern | Example | Unit | Description | Clinical Significance |
|----------------|---------|------|-------------|----------------------|
| `{region}_kurtosis_mean` | `lower_kurtosis_mean` | dimensionless | Average distribution peakedness | Outlier prevalence |
| `{region}_kurtosis_std` | `lower_kurtosis_std` | dimensionless | Variability in kurtosis | Consistency of outliers |
| `{region}_kurtosis_min` | `lower_kurtosis_min` | dimensionless | Minimum kurtosis | Flattest distribution |
| `{region}_kurtosis_max` | `lower_kurtosis_max` | dimensionless | Maximum kurtosis | Most peaked distribution |

**Interpretation:**
- Kurtosis < 3: Platykurtic (flat, few outliers)
- Kurtosis = 3: Normal distribution
- Kurtosis > 3: Leptokurtic (peaked, many outliers)

**Clinical:** High kurtosis may indicate mixed tissue types (normal + fibrotic).

---

#### 3.12 10th Percentile Features (4 features per region)

| Feature Pattern | Example | Unit | Description |
|----------------|---------|------|-------------|
| `{region}_p10_mean` | `lower_p10_mean` | HU | Average 10th percentile intensity |
| `{region}_p10_std` | `lower_p10_std` | HU | Variability in 10th percentile |
| `{region}_p10_min` | `lower_p10_min` | HU | Minimum 10th percentile |
| `{region}_p10_max` | `lower_p10_max` | HU | Maximum 10th percentile |

**Clinical Use:** Captures darkest (most air-filled) lung regions.

---

#### 3.13 90th Percentile Features (4 features per region)

| Feature Pattern | Example | Unit | Description |
|----------------|---------|------|-------------|
| `{region}_p90_mean` | `lower_p90_mean` | HU | Average 90th percentile intensity |
| `{region}_p90_std` | `lower_p90_std` | HU | Variability in 90th percentile |
| `{region}_p90_min` | `lower_p90_min` | HU | Minimum 90th percentile |
| `{region}_p90_max` | `lower_p90_max` | HU | Maximum 90th percentile |

**Clinical Use:** Captures brightest (most dense/fibrotic) lung regions.

---

#### 3.14 Texture Variance Features (4 features per region)

| Feature Pattern | Example | Unit | Description | Clinical Significance |
|----------------|---------|------|-------------|----------------------|
| `{region}_texture_variance_mean` | `lower_texture_variance_mean` | HU² | Average local intensity variation | Texture complexity |
| `{region}_texture_variance_std` | `lower_texture_variance_std` | HU² | Variability in texture variance | Heterogeneity of texture |
| `{region}_texture_variance_min` | `lower_texture_variance_min` | HU² | Minimum texture variance | Most homogeneous slice |
| `{region}_texture_variance_max` | `lower_texture_variance_max` | HU² | Maximum texture variance | Most heterogeneous slice |

**Clinical:** Fibrotic lungs show increased texture variance (irregular patterns).

---

#### 3.15 Texture Entropy Features (4 features per region)

| Feature Pattern | Example | Unit | Description | Clinical Significance |
|----------------|---------|------|-------------|----------------------|
| `{region}_texture_entropy_mean` | `lower_texture_entropy_mean` | bits | Average information complexity | Pattern randomness |
| `{region}_texture_entropy_std` | `lower_texture_entropy_std` | bits | Variability in entropy | Consistency of complexity |
| `{region}_texture_entropy_min` | `lower_texture_entropy_min` | bits | Minimum entropy | Most ordered slice |
| `{region}_texture_entropy_max` | `lower_texture_entropy_max` | bits | Maximum entropy | Most random slice |

**Interpretation:**
- Low entropy: Uniform, organized tissue
- High entropy: Complex, disordered patterns

**Clinical:** Fibrosis creates complex patterns → higher entropy.

---

## 4. FEATURE SUMMARY BY REGION

### Total Features Per Region: ~37

1. Slice count: 1
2. Lung segmentation (mean/std/min/max): 4
3. Lung area ratio (mean/std/min/max): 4
4. Tissue pixels (mean/std/min/max): 4
5. Tissue-by-total (mean/std/min/max): 4
6. Tissue-by-lung (mean/std/min/max): 4
7. Histogram mean (mean/std/min/max): 4
8. Histogram std (mean/std/min/max): 4
9. Histogram median (mean/std/min/max): 4
10. Skewness (mean/std/min/max): 4
11. Kurtosis (mean/std/min/max): 4
12. P10 (mean/std/min/max): 4
13. P90 (mean/std/min/max): 4
14. Texture variance (mean/std/min/max): 4
15. Texture entropy (mean/std/min/max): 4

**37 features × 4 regions = 148 regional features**

---

## 5. COMPLETE FEATURE COUNT

| Category | Count |
|----------|-------|
| Metadata | 5 |
| Volume | 1 |
| Upper lung features | 37 |
| Middle lung features | 37 |
| Lower lung features | 37 |
| Full lung features | 37 |
| **TOTAL** | **153** |

---

## 6. MOST IMPORTANT FEATURES (TOP 20)

Based on clinical relevance and typical predictive power:

| Rank | Feature Name | Why Important |
|------|-------------|---------------|
| 1 | `approx_lung_volume` | Direct lung capacity measure |
| 2 | `lower_tissue_by_lung_mean` | Lower lung fibrosis (disease starts here) |
| 3 | `full_tissue_by_lung_mean` | Overall fibrosis burden |
| 4 | `lower_tissue_by_lung_std` | Lower lung heterogeneity |
| 5 | `middle_tissue_by_lung_mean` | Mid-lung disease extent |
| 6 | `full_mean_mean` | Average lung attenuation |
| 7 | `lower_mean_mean` | Lower lung density |
| 8 | `full_texture_entropy_mean` | Overall pattern complexity |
| 9 | `lower_texture_entropy_mean` | Lower lung complexity |
| 10 | `upper_tissue_by_lung_mean` | Upper lung involvement (advanced disease) |
| 11 | `full_num_tissue_pixels_mean` | Total fibrotic tissue |
| 12 | `lower_num_tissue_pixels_mean` | Lower lung fibrotic tissue |
| 13 | `full_tissue_by_lung_std` | Global disease heterogeneity |
| 14 | `lower_p90_mean` | Dense tissue in lower lung |
| 15 | `middle_tissue_by_lung_std` | Mid-lung heterogeneity |
| 16 | `slice_thickness` | Scan quality/resolution |
| 17 | `total_slices` | Scan coverage |
| 18 | `full_texture_variance_mean` | Overall texture roughness |
| 19 | `lower_skewness_mean` | Lower lung distribution shape |
| 20 | `middle_mean_mean` | Mid-lung attenuation |

---

## 7. DERIVED FEATURES YOU SHOULD CREATE

After extraction, create these composite features for better predictions:

```python
# 1. Disease Progression Indicator (Lower vs Upper)
df['fibrosis_gradient'] = (
    df['lower_tissue_by_lung_mean'] / 
    (df['upper_tissue_by_lung_mean'] + 0.01)
)

# 2. Disease Heterogeneity Index
df['heterogeneity_index'] = (
    df['full_tissue_by_lung_std'] / 
    (df['full_tissue_by_lung_mean'] + 0.01)
)

# 3. Volume-Adjusted Tissue Burden
df['tissue_burden'] = (
    df['approx_lung_volume'] * df['full_tissue_by_lung_mean']
)

# 4. Regional Spread (Range)
df['regional_spread'] = (
    df['lower_tissue_by_lung_mean'] - 
    df['upper_tissue_by_lung_mean']
)

# 5. Density Range (90th - 10th percentile)
df['density_range'] = (
    df['full_p90_mean'] - df['full_p10_mean']
)

# 6. Mean-Median Difference (symmetry indicator)
df['distribution_asymmetry'] = (
    df['full_mean_mean'] - df['full_median_mean']
)

# 7. Texture Complexity Ratio
df['texture_complexity'] = (
    df['full_texture_entropy_mean'] / 
    (df['full_texture_variance_mean'] + 0.01)
)
```

---

## 8. FEATURE USAGE BY MODEL TYPE

### Traditional ML (Random Forest, XGBoost)
**Use:** All 153 features + derived features
**Preprocessing:** StandardScaler or MinMaxScaler
**Feature selection:** Use feature importance or LASSO

### Deep Learning
**Use:** Consider sequence models (LSTM) with slice-level features
**Or:** Use raw images + clinical metadata

### Ensemble
**Use:** Combine image features + clinical data (age, sex, smoking, baseline FVC)

---

## 9. TYPICAL FEATURE VALUES

### Example: Healthy vs Fibrotic Patient

| Feature | Healthy | Mild Fibrosis | Severe Fibrosis |
|---------|---------|---------------|-----------------|
| `approx_lung_volume` | 5,000,000 mm³ | 4,000,000 mm³ | 2,500,000 mm³ |
| `lower_tissue_by_lung_mean` | 0.08 | 0.22 | 0.45 |
| `upper_tissue_by_lung_mean` | 0.06 | 0.12 | 0.35 |
| `full_mean_mean` | -750 HU | -500 HU | -300 HU |
| `full_texture_entropy_mean` | 3.2 bits | 4.1 bits | 4.8 bits |
| `fibrosis_gradient` (derived) | 1.33 | 1.83 | 1.29 |

**Note:** Severe fibrosis shows high tissue burden but gradient may decrease as upper lung catches up.

---

## 10. CLINICAL INTERPRETATION GUIDE

### Reading a Patient's Features

**High-Risk Pattern:**
- ✓ `approx_lung_volume` < 3,000,000 mm³ (low volume)
- ✓ `lower_tissue_by_lung_mean` > 0.30 (high lower fibrosis)
- ✓ `full_tissue_by_lung_std` > 0.10 (heterogeneous)
- ✓ `full_texture_entropy_mean` > 4.5 (complex patterns)
- ✓ `fibrosis_gradient` > 1.5 (lower predominant)

**Stable/Low-Risk Pattern:**
- ✓ `approx_lung_volume` > 4,500,000 mm³
- ✓ `lower_tissue_by_lung_mean` < 0.15
- ✓ `full_tissue_by_lung_std` < 0.05
- ✓ `fibrosis_gradient` < 1.3

---

## Questions?

If you need help understanding a specific feature or how to use them in your model, let me know!
