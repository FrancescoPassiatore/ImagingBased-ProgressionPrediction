# Feature Descriptions for CT-Based Pulmonary Fibrosis Analysis

## Overview
This document describes the 14 features extracted from chest CT scans for pulmonary fibrosis progression prediction. Features are extracted from the 30-60th percentile range of slices to focus on the mid-lung region.

---

## METADATA FEATURES (2 features)

### 1. Patient
**Type:** Identifier (String)  
**Description:** Unique patient identifier  
**Purpose:** Links CT features to clinical data  
**Example:** `ID00007637202177411956430`

### 2. Data
**Type:** Categorical (String)  
**Description:** Dataset split indicator  
**Values:** `'train'` or `'test'`  
**Purpose:** Identifies which dataset the patient belongs to

---

## SCANNER METADATA FEATURES (2 features)

### 3. SliceThickness
**Type:** Continuous (Float)  
**Unit:** Millimeters (mm)  
**Description:** Distance between consecutive CT slices  
**Typical Range:** 0.5 - 10.0 mm  
**Clinical Significance:**
- Determines spatial resolution in z-axis
- Thinner slices = better detail, more slices
- Affects volume calculations

**Example Values from Your Data:**
- Patient 430: 1.25 mm
- Patient 362: 7.0 mm (thicker, fewer slices)

**Interpretation:**
- **Thin (0.5-2mm):** High-resolution scans, detailed imaging
- **Thick (>5mm):** Lower resolution, faster acquisition

---

### 4. PixelSpacing
**Type:** Continuous (Float)  
**Unit:** Millimeters (mm)  
**Description:** Physical size of each pixel in the image (in-plane resolution)  
**Typical Range:** 0.3 - 1.0 mm  
**Clinical Significance:**
- Determines in-plane spatial resolution
- Smaller spacing = finer detail visible
- Affects area and volume calculations

**Example Values from Your Data:**
- Patient 278: 0.487 mm (fine resolution)
- Patient 317: 0.799 mm (coarser resolution)

**Interpretation:**
- **Fine (<0.6mm):** Can detect small fibrotic changes
- **Coarse (>0.7mm):** May miss subtle patterns

---

## TECHNICAL FEATURES (2 features)

### 5. NumImgBw5Prec
**Type:** Integer  
**Description:** Number of images (slices) between consecutive 5% percentile intervals in the 30-60% range  
**Typical Range:** 1 - 50 slices  
**Clinical Significance:**
- Indicates sampling density in the mid-lung region
- More slices = more detailed coverage
- Used for volume calculation normalization

**Example Values from Your Data:**
- Patient 430: 1 slice (very sparse)
- Patient 278: 12 slices (dense sampling)

**Technical Note:** This is derived from your scan protocol and affects how representative your samples are.

---

### 6. ApproxVol_30_60
**Type:** Continuous (Float)  
**Unit:** Cubic millimeters (mm³)  
**Description:** Approximate lung volume in the 30-60th percentile region  
**Typical Range:** 100,000 - 15,000,000 mm³  
**Clinical Significance:**
- ⭐ **KEY FEATURE** for FVC prediction
- Reflects lung capacity in mid-lung region
- Volume loss = disease progression
- Strongly correlates with pulmonary function

**Example Values from Your Data:**
- Patient 430: 445,809 mm³ (small volume)
- Patient 278: 9,620,161 mm³ (large volume)

**Calculation:**
```
ApproxVol = Σ(lung_pixels × pixel_spacing² × slice_thickness × num_slices)
```

**Interpretation:**
- **Large volume (>5M mm³):** Better preserved lung capacity
- **Medium volume (2-5M mm³):** Moderate lung involvement
- **Small volume (<1M mm³):** Significant lung volume loss

**Clinical Correlation:**
- Inversely correlated with fibrosis severity
- Decreases over time in progressive disease
- Strong predictor of FVC (Forced Vital Capacity)

---

## TISSUE QUANTIFICATION FEATURES (5 features)

These features quantify the amount of fibrotic tissue in the lung.

### 7. Avg_NumTissuePixel_30_60
**Type:** Continuous (Float)  
**Unit:** Pixels  
**Description:** Average number of high-attenuation (bright) tissue pixels per slice  
**Typical Range:** 1,000 - 50,000 pixels  
**Clinical Significance:**
- Raw count of fibrotic/dense tissue pixels
- Higher values = more fibrotic tissue
- Not normalized (depends on image size)

**Example Values from Your Data:**
- Patient 362: 5,153 pixels (low tissue burden)
- Patient 278: 9,616 pixels (higher tissue burden)

**Interpretation:**
- **Low (<5,000):** Minimal fibrosis visible
- **Moderate (5,000-10,000):** Moderate fibrosis
- **High (>10,000):** Extensive fibrotic changes

---

### 8. Avg_Tissue_30_60
**Type:** Continuous (Float)  
**Unit:** Square millimeters (mm²)  
**Description:** Average tissue area per slice (pixel count × pixel spacing)  
**Typical Range:** 500 - 30,000 mm²  
**Clinical Significance:**
- Physical area of fibrotic tissue
- Normalized by pixel size (unlike Avg_NumTissuePixel)
- Directly comparable across different scanners

**Example Values from Your Data:**
- Patient 184: 3,746 mm² (minimal tissue)
- Patient 317: 6,817 mm² (moderate tissue)

**Calculation:**
```
Avg_Tissue = Avg_NumTissuePixel × PixelSpacing
```

**Interpretation:**
- **Minimal (<3,000 mm²):** Early disease or healthy
- **Moderate (3,000-7,000 mm²):** Progressive fibrosis
- **Extensive (>7,000 mm²):** Advanced disease

---

### 9. Avg_Tissue_thickness_30_60
**Type:** Continuous (Float)  
**Unit:** Cubic millimeters (mm³)  
**Description:** Average tissue volume per slice (area × slice thickness)  
**Typical Range:** 500 - 100,000 mm³  
**Clinical Significance:**
- Volumetric measure of tissue burden
- Accounts for slice thickness variation
- Better for comparing across different scan protocols

**Example Values from Your Data:**
- Patient 430: 7,100 mm³
- Patient 362: 22,545 mm³ (thick slices = large value)

**Calculation:**
```
Avg_Tissue_thickness = Avg_Tissue × SliceThickness
```

**Note:** This value is heavily influenced by SliceThickness, so interpret with caution when comparing patients.

---

### 10. Avg_TissueByTotal_30_60
**Type:** Continuous (Float)  
**Unit:** Ratio (0-1)  
**Description:** Average ratio of tissue pixels to total image pixels  
**Typical Range:** 0.005 - 0.15  
**Clinical Significance:**
- Normalized measure of tissue burden
- Accounts for varying image sizes
- Indicates proportion of image occupied by dense tissue

**Example Values from Your Data:**
- Patient 278: 0.0163 (1.6% of image is tissue)
- Patient 430: 0.0332 (3.3% of image is tissue)

**Calculation:**
```
Avg_TissueByTotal = Tissue_Pixels / Total_Image_Pixels
```

**Interpretation:**
- **Low (<0.02):** Minimal tissue visible
- **Moderate (0.02-0.05):** Moderate fibrotic burden
- **High (>0.05):** Extensive fibrosis

**Limitation:** Includes background area, not just lung tissue.

---

### 11. Avg_TissueByLung_30_60 ⭐ MOST IMPORTANT
**Type:** Continuous (Float)  
**Unit:** Ratio (0-1)  
**Description:** Average ratio of tissue pixels to lung pixels (excluding background)  
**Typical Range:** 0.03 - 0.50  
**Clinical Significance:**
- ⭐ **PRIMARY FIBROSIS METRIC**
- Directly measures fibrotic tissue proportion within lung
- Most clinically relevant feature
- Strong predictor of disease severity and progression
- Normalized by lung size (fair comparison across patients)

**Example Values from Your Data:**
- Patient 362: 0.0791 (7.9% of lung is fibrotic tissue)
- Patient 317: 0.1882 (18.8% of lung is fibrotic tissue)

**Calculation:**
```
Avg_TissueByLung = Tissue_Pixels / Lung_Pixels
```

**Clinical Interpretation:**
| Range | Severity | Description |
|-------|----------|-------------|
| < 0.10 | **Minimal/Mild** | Healthy or early disease |
| 0.10 - 0.20 | **Moderate** | Established fibrosis |
| 0.20 - 0.30 | **Severe** | Advanced disease |
| > 0.30 | **Very Severe** | Extensive fibrotic involvement |

**Your Sample Distribution:**
- Minimal (8-10%): Patients 362, 278, 184, 924
- Moderate (14-19%): Patients 671, 430, 317

**Why This Feature Matters:**
1. Directly quantifies disease burden
2. Normalized metric (fair across patients)
3. Correlates with FVC decline
4. Tracks disease progression
5. Independent of scanner settings

**Expected Correlation with Progression:**
- **Positive correlation:** Higher tissue-by-lung = worse outcomes
- Typical r = 0.4-0.6 with FVC decline

---

## HISTOGRAM FEATURES (3 features)

These features describe the distribution of pixel intensities (Hounsfield Units) within the lung.

### 12. Mean_30_60
**Type:** Continuous (Float)  
**Unit:** Hounsfield Units (HU)  
**Description:** Average pixel intensity in the lung region  
**Typical Range:** -900 to +500 HU  
**Clinical Significance:**
- Reflects overall tissue density
- Negative values = air-filled lung (normal)
- Positive values = denser tissue (fibrosis, fluid)
- Shift toward positive values indicates fibrosis

**Example Values from Your Data:**
- Patient 362: -602.7 HU (well-aerated)
- Patient 430: +503.3 HU (dense tissue)

**Hounsfield Unit Reference:**
| HU Range | Tissue Type |
|----------|-------------|
| -1000 | Air |
| -900 to -700 | Normal lung |
| -700 to -400 | Mild ground glass opacity |
| -400 to 0 | Ground glass / fibrosis |
| 0 to +100 | Soft tissue / fibrosis |
| +100 to +500 | Dense fibrotic tissue |

**Interpretation of Your Samples:**
- **Negative values (362, 317):** Still have aerated lung tissue
- **Positive values (430, 671):** More dense fibrotic tissue

**Clinical Correlation:**
- Higher (more positive) mean = more fibrosis
- Negative mean = better preserved lung aeration
- Expected correlation with progression: positive (r ~0.3-0.5)

---

### 13. Skew_30_60
**Type:** Continuous (Float)  
**Unit:** Dimensionless  
**Description:** Asymmetry of the pixel intensity distribution  
**Typical Range:** -2 to +5  
**Clinical Significance:**
- Describes distribution shape
- Positive skew = tail on right (more bright pixels)
- Indicates heterogeneity of lung tissue

**Example Values from Your Data:**
- Patient 671: 0.789 (slight right skew)
- Patient 278: 2.009 (pronounced right skew)

**Interpretation:**
- **Skew < 0:** Left-skewed (more dark/aerated pixels)
  - Unusual in fibrosis
  - May indicate emphysema component

- **Skew ≈ 0:** Symmetric distribution
  - Uniform tissue density
  - Less common in disease

- **Skew > 0:** Right-skewed (more bright/dense pixels)
  - **Common in fibrosis**
  - Indicates patchy disease with dense areas
  - Higher skew = more heterogeneous fibrosis

- **Skew > 2:** Highly right-skewed
  - Very heterogeneous disease
  - Mix of normal and fibrotic areas

**Clinical Correlation:**
- Moderate positive skew typical in IPF
- High skew may indicate rapid progression
- Captures disease heterogeneity

---

### 14. Kurtosis_30_60
**Type:** Continuous (Float)  
**Unit:** Dimensionless  
**Description:** "Peakedness" of the pixel intensity distribution  
**Typical Range:** -2 to +10  
**Clinical Significance:**
- Measures presence of outliers and distribution shape
- High kurtosis = heavy tails + peaked center
- Indicates extreme values (very dense or very aerated areas)

**Example Values from Your Data:**
- Patient 671: 0.812 (low kurtosis)
- Patient 278: 5.935 (high kurtosis)

**Interpretation:**
- **Kurtosis < 3:** Platykurtic (flat distribution)
  - Uniform lung attenuation
  - Homogeneous disease pattern
  - Less variability

- **Kurtosis ≈ 3:** Normal distribution (mesokurtic)
  - Moderate variability
  - Typical in early disease

- **Kurtosis > 3:** Leptokurtic (peaked distribution)
  - **Common in advanced fibrosis**
  - Many pixels at typical values + extreme outliers
  - Very heterogeneous lung tissue

- **Kurtosis > 5:** Highly leptokurtic
  - Extreme heterogeneity
  - Mix of very dense (fibrotic) and very aerated areas
  - Patchy fibrosis pattern

**Clinical Correlation:**
- High kurtosis = heterogeneous disease
- May indicate UIP (Usual Interstitial Pneumonia) pattern
- Correlates with disease complexity

**Histogram Features Together:**
- **Mean:** Overall density (fibrosis burden)
- **Skew:** Distribution asymmetry (heterogeneity)
- **Kurtosis:** Distribution shape (outlier prevalence)

These three features capture different aspects of lung tissue heterogeneity, which is characteristic of fibrotic lung disease.

---

## FEATURE SUMMARY TABLE

| # | Feature | Type | Unit | Range | Importance | Clinical Meaning |
|---|---------|------|------|-------|------------|------------------|
| 1 | Patient | ID | - | - | - | Identifier |
| 2 | Data | Categorical | - | train/test | - | Dataset split |
| 3 | SliceThickness | Scanner | mm | 0.5-10 | Low | Scan protocol |
| 4 | PixelSpacing | Scanner | mm | 0.3-1.0 | Low | Scan protocol |
| 5 | NumImgBw5Prec | Technical | count | 1-50 | Low | Sampling density |
| 6 | ApproxVol_30_60 | Volume | mm³ | 100K-15M | ⭐⭐⭐⭐⭐ | Lung capacity |
| 7 | Avg_NumTissuePixel_30_60 | Tissue | pixels | 1K-50K | ⭐⭐⭐ | Raw tissue count |
| 8 | Avg_Tissue_30_60 | Tissue | mm² | 500-30K | ⭐⭐⭐ | Tissue area |
| 9 | Avg_Tissue_thickness_30_60 | Tissue | mm³ | 500-100K | ⭐⭐ | Tissue volume |
| 10 | Avg_TissueByTotal_30_60 | Tissue Ratio | 0-1 | 0.005-0.15 | ⭐⭐⭐ | Tissue/Total image |
| 11 | **Avg_TissueByLung_30_60** | **Tissue Ratio** | **0-1** | **0.03-0.50** | **⭐⭐⭐⭐⭐** | **PRIMARY FIBROSIS METRIC** |
| 12 | Mean_30_60 | Histogram | HU | -900 to +500 | ⭐⭐⭐⭐ | Lung density |
| 13 | Skew_30_60 | Histogram | - | -2 to +5 | ⭐⭐⭐ | Distribution asymmetry |
| 14 | Kurtosis_30_60 | Histogram | - | -2 to +10 | ⭐⭐⭐ | Distribution peakedness |

---

## FEATURE IMPORTANCE FOR MODELING

### Top Tier (Must Include):
1. **Avg_TissueByLung_30_60** - Direct fibrosis measurement
2. **ApproxVol_30_60** - Lung capacity
3. **Mean_30_60** - Overall tissue density

### Second Tier (Highly Recommended):
4. **Avg_NumTissuePixel_30_60** - Tissue burden
5. **Skew_30_60** - Heterogeneity indicator
6. **Kurtosis_30_60** - Pattern complexity

### Third Tier (Optional):
7. **Avg_TissueByTotal_30_60** - Alternative ratio
8. **Avg_Tissue_30_60** - Physical area measure

### Low Predictive Value:
9. Scanner metadata (SliceThickness, PixelSpacing)
10. Technical features (NumImgBw5Prec)

---

## RECOMMENDED FEATURE ENGINEERING

### Derived Features to Create:

```python
# 1. Tissue density (tissue per unit volume)
df['tissue_density'] = df['Avg_NumTissuePixel_30_60'] / (df['ApproxVol_30_60'] + 1)

# 2. Normalized volume (z-score)
df['volume_zscore'] = (df['ApproxVol_30_60'] - df['ApproxVol_30_60'].mean()) / df['ApproxVol_30_60'].std()

# 3. Fibrosis severity category
df['fibrosis_severity'] = pd.cut(
    df['Avg_TissueByLung_30_60'],
    bins=[0, 0.10, 0.20, 0.30, 1.0],
    labels=['Mild', 'Moderate', 'Severe', 'Very Severe']
)

# 4. Tissue-volume interaction
df['tissue_vol_interaction'] = df['Avg_TissueByLung_30_60'] * df['ApproxVol_30_60']

# 5. HU density ratio (mean relative to typical lung)
df['hu_density_ratio'] = (df['Mean_30_60'] + 750) / 750  # Normalized to typical lung HU
```

---

## FEATURE CORRELATIONS (Expected)

**Strong Positive Correlations:**
- Avg_NumTissuePixel_30_60 ↔ Avg_Tissue_30_60 (r ~0.99)
- Avg_TissueByTotal_30_60 ↔ Avg_TissueByLung_30_60 (r ~0.7)

**Moderate Positive Correlations:**
- Mean_30_60 ↔ Avg_TissueByLung_30_60 (r ~0.5)
- Skew_30_60 ↔ Kurtosis_30_60 (r ~0.4)

**Negative Correlations:**
- ApproxVol_30_60 ↔ Avg_TissueByLung_30_60 (r ~-0.3)

---

## CLINICAL VALIDATION EXAMPLES

### Example 1: Mild Fibrosis Patient
```
Patient 362:
- Avg_TissueByLung_30_60: 0.0791 (7.9%) → Minimal/Mild
- ApproxVol_30_60: 7,202,151 mm³ → Good volume
- Mean_30_60: -602.7 HU → Well-aerated
→ Interpretation: Early disease, good prognosis
```

### Example 2: Moderate Fibrosis Patient
```
Patient 317:
- Avg_TissueByLung_30_60: 0.1882 (18.8%) → Moderate
- ApproxVol_30_60: 510,755 mm³ → Reduced volume
- Mean_30_60: -580.5 HU → Still some aeration
→ Interpretation: Established disease, moderate severity
```

### Example 3: Variable Disease Burden
```
Patient 278:
- Avg_TissueByLung_30_60: 0.0923 (9.2%) → Mild
- ApproxVol_30_60: 9,620,161 mm³ → Large volume
- Kurtosis: 5.94 → Highly heterogeneous
→ Interpretation: Early disease with patchy distribution
```

---

## QUALITY CHECKS FOR YOUR FEATURES

✓ **Check 1:** No zero volumes
✓ **Check 2:** Avg_TissueByLung_30_60 between 0.03-0.50
✓ **Check 3:** Mean_30_60 between -900 and +500 HU
✓ **Check 4:** ApproxVol_30_60 > 0 for all patients

---

## REFERENCES

These features are based on:
1. Standard radiomics approaches for lung CT analysis
2. Clinical understanding of pulmonary fibrosis imaging
3. Quantitative CT metrics used in IPF research

For more details on the scientific basis, see `SCIENTIFIC_REFERENCES_AND_BMI.md`

---

## SUMMARY

You have extracted **14 features** comprising:
- ✓ 2 identifiers
- ✓ 2 scanner metadata
- ✓ 2 technical features
- ✓ 6 tissue quantification features (including the critical Avg_TissueByLung_30_60)
- ✓ 3 histogram features

**Most Important Features:**
1. Avg_TissueByLung_30_60 (fibrosis %)
2. ApproxVol_30_60 (lung volume)
3. Mean_30_60 (tissue density)

These features capture lung volume, fibrotic tissue burden, and tissue heterogeneity - the key factors in pulmonary fibrosis progression!
