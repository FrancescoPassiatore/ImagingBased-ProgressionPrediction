# Survival Analysis — Results & Discussion Draft

---

## 4.X  Survival Analysis Results

### 4.X.1  Overview

A Cox Proportional Hazards (Cox PH) model was trained to predict time-to-progression
in a cohort of 84 IPF patients using five-fold cross-validation. Three feature
modalities were evaluated in a systematic ablation study: clinical demographics
(sex, age, smoking status), hand-crafted CT radiomics (nine texture and volumetric
features extracted from the 30–60% lung region), and deep CNN embeddings
(ResNet50, ImageNet pretrained). Model performance was assessed using the
concordance index (C-index), Kaplan–Meier risk stratification, and log-rank tests.

---

### 4.X.2  Feature Ablation Results

Table X summarises the mean validation and test C-index across all tested
configurations, sorted by test performance.

**Demographic features outperformed all other modalities.** The full demographic
model comprising sex, age, and smoking status (`demo_all`) achieved the highest
mean test C-index of 0.559 (±0.062), followed by the sex-and-age combination
(0.547 ± 0.128). Sex alone, used as the primary baseline, yielded a test C-index
of 0.526 (±0.066), confirming its role as a robust single-feature predictor
consistent with prior clinical evidence on sex differences in IPF progression.

**Handcrafted radiomics showed modest but real signal.** Among imaging features,
average tissue thickness was the strongest individual predictor (test C-index
0.539 ± 0.070), outperforming all other single radiomics features including
kurtosis and approximate lung volume. The complete nine-feature radiomics set
achieved a test C-index of 0.522 (±0.049) — notably the most stable result
across folds — suggesting that Ridge regularisation effectively absorbs the
feature collinearity present in this set without penalising predictive performance.
Crucially, the previously hypothesised reduced three-feature set (kurtosis,
approximate volume, and thickness) underperformed the full nine-feature set
in cross-validation (0.444 vs 0.522 test C-index), indicating that features
flagged as redundant by univariate correlation analysis still carry complementary
survival signal when regularisation is applied.

**Combining radiomics with demographics did not improve over demographics alone.**
The recommended baseline from prior analysis (`hand_all_demo`: all nine radiomics
features plus sex) achieved a test C-index of 0.519 (±0.081), ranking tenth
overall and below the demographic-only model. This suggests that, at the current
sample size, radiomics features do not add independent prognostic information
beyond clinical patient characteristics.

**CNN features added no survival signal.** All configurations incorporating
ResNet50 embeddings scored below 0.42 on test C-index, with the CNN-only
four-statistic aggregation achieving the worst performance of any tested
configuration (0.348 ± 0.120). This finding is consistent with the fundamental
mismatch between ImageNet-pretrained feature spaces and IPF-specific survival
prediction: without task-specific fine-tuning, deep embeddings encode
image-naturalness features rather than disease-relevant texture patterns.
Furthermore, the events-per-variable constraint inherent to Cox regression
(approximately 30 events in this cohort) limits the number of usable predictors
to 2–5, making high-dimensional CNN representations statistically infeasible
regardless of dimensionality reduction approach.

**A methodological note on prior feature selection.** Univariate C-index
estimates computed during initial feature diagnostics (Diagnostic 2) substantially
overestimated predictive performance due to the absence of a held-out test split
— kurtosis, for example, was estimated at C = 0.583 in univariate analysis but
achieved only C = 0.376 in proper five-fold cross-validation. All results reported
in this section are based exclusively on cross-validated estimates and should be
considered the authoritative performance figures.

---

### 4.X.3  Risk Stratification

Kaplan–Meier curves were generated for the best-performing model (`demo_all`)
by splitting each validation fold at the median predicted risk score. Figure X
presents all five folds simultaneously.

Three of five folds (1, 4, and 5) showed consistent separation between
high- and low-risk groups in the correct direction, with high-risk patients
exhibiting lower progression-free survival probability over time. Fold 1 showed
the strongest separation, with the high-risk group maintaining near-unity
progression-free probability until week 25 while the low-risk group declined
rapidly from early follow-up — a pattern consistent with a model that correctly
identifies patients whose disease is initially stable but who progress late.

Fold 2 showed near-complete overlap between risk groups, contributing
disproportionately to the high validation standard deviation observed for
`demo_all` (±0.178). Fold 3 showed delayed separation, with curves diverging
only after week 38, likely reflecting an atypical event-timing distribution
in that fold's composition. These inconsistencies are attributable to the small
validation set size (16–17 patients per fold) rather than systematic model
failure: with so few patients, a single atypical individual can substantially
alter a fold's apparent performance.

Log-rank tests comparing high- and low-risk groups reached statistical significance
(p < 0.05) in [X] of five folds for the `demo_all` model. The mean log-rank
p-value across folds was [X], with fold-level results reported in Table Y.

---

### 4.X.4  Calibration

Calibration was assessed by binning pooled validation-set patients into five
risk quintiles and comparing the mean predicted risk to the observed event rate
within each bin (Figure X, Panel A). The risk score distribution stratified by
event status is shown in Panel B.

[Report Pearson r and KS statistic from your run here — e.g.:]
The Pearson correlation between mean predicted risk and observed event rate
across quintiles was r = [X] (p = [X]), indicating [moderate/weak] calibration.
The Kolmogorov–Smirnov statistic comparing the predicted risk distributions of
event and censored patients was [X] (p = [X]), suggesting [significant/no
significant] separation between the two groups.

It should be noted that calibration assessment at this sample size carries
substantial uncertainty. With approximately 84 patients divided into five bins,
each bin contains only 16–17 patients on average, making observed event rates
highly sensitive to individual case composition. The calibration results should
therefore be interpreted as indicative rather than definitive.

---

### 4.X.5  Discussion

The primary finding of this survival analysis is that clinical demographic
variables — specifically the combination of sex, age, and smoking status —
outperform CT-derived imaging features for IPF progression-time prediction in
this cohort. This result is both clinically meaningful and statistically
interpretable: IPF is known to be associated with sex (male predominance and
worse prognosis), age at diagnosis, and smoking history, and these factors
have established biological and epidemiological foundations independent of
imaging.

The failure of radiomics to add predictive value over demographics is likely
attributable to a combination of factors. First, the sample size (N = 84) limits
statistical power to detect moderate radiomics effects in a multivariate survival
model, particularly under the events-per-variable constraint that restricts
feasible model complexity. Second, the hand-crafted features employed here were
derived from a fixed lung region (30–60% of lung height) and summarised at the
patient level through simple averaging, which may not capture the spatial
heterogeneity or progression patterns most relevant to survival. Third, the Cox
PH model assumes log-linearity of feature effects on hazard, which may not hold
for complex imaging biomarkers.

The poor performance of CNN embeddings, while consistent with theoretical
expectations for a task-agnostic pretrained model, highlights a critical gap
for future work. ResNet50 features trained on ImageNet have no semantic alignment
with IPF tissue patterns. A fine-tuned model — trained on CT image patches with
survival supervision — would be expected to extract disease-relevant features and
may overcome the limitations observed here. This represents the most promising
direction for improving imaging-based survival prediction in IPF, though it
requires either a larger annotated cohort or transfer learning from related
pulmonary fibrosis datasets.

The findings are consistent with the broader literature on CT-based survival
prediction in small IPF cohorts, where radiomics models frequently show modest
C-index values in the range of 0.55–0.65 and rarely outperform clinical baselines
without careful feature engineering or deep learning with task-specific training.

---

### 4.X.6  Limitations

- **Sample size.** 84 patients with a limited number of progression events
  constrains model complexity (EPV < 10 for all but the simplest configurations)
  and produces high fold-to-fold variance in evaluation metrics.

- **Single imaging timepoint.** All features were extracted from a single
  baseline CT scan. Longitudinal CT data, where available, would capture disease
  dynamics directly relevant to progression timing.

- **Fixed-region feature extraction.** Radiomics were computed from the 30–60%
  lung region only. Full-lung or disease-burden-weighted extraction may yield
  stronger survival associations.

- **Cox PH assumptions.** The proportional hazards assumption was not formally
  tested (Schoenfeld residuals) due to the small event count. Violations of this
  assumption could bias coefficient estimates.

- **ImageNet CNN features.** Deep features were extracted without fine-tuning on
  IPF data. Task-specific CNN training is required to evaluate the true
  contribution of imaging-derived deep features to survival prediction.

---

*[Note for thesis: fill in bracketed values [X] from your actual run of
thesis_plots.py. The log-rank p per fold and calibration statistics are printed
to console and saved to logrank_table_demo_all.csv and calibration_demo_all.png.]*