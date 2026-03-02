"""
ctfm_diagnostics.py
===================
Comprehensive diagnostic script to identify why CT-FM ablation results
are at or below random chance (AUC ~0.45).

Runs 4 diagnostic blocks:
  1. Embedding quality checks   – are the vectors degenerate / near-identical?
  2. Label & fold sanity checks – class balance, leakage, fold sizes
  3. Sklearn baseline           – logistic regression on raw embeddings
  4. Pipeline spot-check        – verify HU values, slice ordering, orientation
                                  for a sample of patients

Each block prints a clear PASS / WARNING / FAIL verdict.
Run this before any further training.
"""

import os
import glob
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION – edit these paths
# =============================================================================
CONFIG = {
    # Pre-computed CT-FM embeddings CSV
    "ctfm_csv": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Data_Engineering\CT-FM_extractor\ctfm_embeddings.csv"),

    # Ground truth / labels
    "gt_csv": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\ground_truth.csv"),

    # K-fold splits pickle
    "kfold_pkl": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\kfold_splits_stratified.pkl"),

    # Raw DICOM folder (for pipeline spot-check)
    "dicom_dir": Path(r"D:\FrancescoP\train"),

    # Raw .npy volumes folder (output of Stage 1 in dicom_to_ctfm.py)
    "raw_volume_dir": Path(r"D:\FrancescoP\raw_volumes"),

    # Where to save diagnostic plots and report
    "output_dir": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Data_Engineering\CT-FM_extractor\diagnostics_output"),

    # Number of patients to spot-check in Block 4
    "n_spot_check": 5,

    # Column names
    "patient_col": "patient_id",   # in ctfm_csv
    "label_col":   "label",        # in ctfm_csv  (or 'gt_has_progressed')
}
# =============================================================================

FEAT_PREFIX = "volume_feature_"
SEP = "=" * 70


def header(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def verdict(status, message):
    icons = {"PASS": "✅", "WARNING": "⚠️ ", "FAIL": "❌"}
    print(f"\n  {icons.get(status, '  ')} [{status}]  {message}")


# =============================================================================
# LOADING HELPERS
# =============================================================================

def load_embeddings(cfg):
    path = cfg["ctfm_csv"]
    assert path.exists(), f"CT-FM CSV not found: {path}"
    df = pd.read_csv(path)

    # Normalise label column name
    for candidate in ["label", "gt_has_progressed", "Label", "progression"]:
        if candidate in df.columns and cfg["label_col"] not in df.columns:
            df.rename(columns={candidate: cfg["label_col"]}, inplace=True)

    feat_cols = [c for c in df.columns if c.startswith(FEAT_PREFIX)]
    print(f"\n  Loaded:  {len(df)} patients × {len(feat_cols)} features")
    print(f"  Columns: {df.columns.tolist()[:4]} ... {df.columns.tolist()[-2:]}")
    return df, feat_cols


def load_kfold(cfg):
    path = cfg["kfold_pkl"]
    if not path.exists():
        print(f"  ⚠️  K-fold pickle not found at {path}")
        return None
    with open(path, "rb") as f:
        splits = pickle.load(f)
    return splits


# =============================================================================
# BLOCK 1 – EMBEDDING QUALITY
# =============================================================================

def block1_embedding_quality(df, feat_cols, output_dir):
    header("BLOCK 1 – EMBEDDING QUALITY")

    X = df[feat_cols].values.astype(np.float32)
    n_patients, n_feats = X.shape

    # ── 1a. NaN / Inf check ──────────────────────────────────────────────────
    print("\n  [1a] NaN / Inf check")
    nan_rows  = np.isnan(X).any(axis=1).sum()
    inf_rows  = np.isinf(X).any(axis=1).sum()
    print(f"      Patients with NaN:  {nan_rows}")
    print(f"      Patients with Inf:  {inf_rows}")
    if nan_rows == 0 and inf_rows == 0:
        verdict("PASS", "No NaN or Inf values in embeddings.")
    else:
        verdict("FAIL", f"{nan_rows} NaN rows, {inf_rows} Inf rows – fix extraction first.")
        X = X[~np.isnan(X).any(axis=1)]   # drop for subsequent checks

    # ── 1b. Per-feature variance ─────────────────────────────────────────────
    print("\n  [1b] Per-feature variance (should not be near 0)")
    feat_std = X.std(axis=0)
    zero_var  = (feat_std < 1e-6).sum()
    low_var   = (feat_std < 0.01).sum()
    print(f"      Features with std < 1e-6 (zero variance): {zero_var}")
    print(f"      Features with std < 0.01 (very low):      {low_var}")
    print(f"      Median feature std:  {np.median(feat_std):.4f}")
    print(f"      Min / Max feature std: {feat_std.min():.4f} / {feat_std.max():.4f}")
    if zero_var > 0:
        verdict("FAIL", f"{zero_var} features have zero variance – embeddings are degenerate.")
    elif low_var > n_feats * 0.3:
        verdict("WARNING", f"{low_var}/{n_feats} features have very low variance.")
    else:
        verdict("PASS", "Feature variance looks healthy.")

    # ── 1c. Pairwise cosine similarity ───────────────────────────────────────
    print("\n  [1c] Pairwise cosine similarity (should vary, not all ≈ 1.0)")
    sample = X[:min(30, n_patients)]
    norms  = np.linalg.norm(sample, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    sample_norm = sample / norms
    sim_matrix  = sample_norm @ sample_norm.T
    # Exclude diagonal
    mask    = ~np.eye(len(sample), dtype=bool)
    off_diag = sim_matrix[mask]
    print(f"      Mean cosine sim (off-diagonal): {off_diag.mean():.4f}")
    print(f"      Std  cosine sim:                {off_diag.std():.4f}")
    print(f"      Min / Max:                      {off_diag.min():.4f} / {off_diag.max():.4f}")
    if off_diag.mean() > 0.98:
        verdict("FAIL", "Embeddings are nearly identical (mean cosine sim > 0.98). "
                        "CT-FM is producing degenerate outputs – check preprocessing.")
    elif off_diag.mean() > 0.90:
        verdict("WARNING", "Embeddings are very similar (mean cosine sim > 0.90). "
                           "Very low diversity – check orientation and CropForeground.")
    else:
        verdict("PASS", f"Embedding diversity OK (mean cosine sim = {off_diag.mean():.3f}).")

    # ── 1d. Class separation in PCA space ────────────────────────────────────
    label_col = CONFIG["label_col"]
    if label_col in df.columns:
        print("\n  [1d] Class separation in PCA / t-SNE space")
        labels = df[label_col].values
        valid  = ~np.isnan(X).any(axis=1)
        X_v, y_v = X[valid], labels[valid]

        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X_v)

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_sc)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # PCA plot
        for lbl, color, name in [(0, 'steelblue', 'No Progression'),
                                   (1, 'crimson',   'Progression')]:
            mask = y_v == lbl
            axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                            c=color, label=name, alpha=0.7, s=50)
        axes[0].set_title(f"PCA (var explained: {pca.explained_variance_ratio_.sum():.1%})")
        axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
        axes[0].legend()

        # t-SNE plot (only if enough samples)
        if len(X_v) >= 10:
            perp = min(30, len(X_v) // 3)
            tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
            X_tsne = tsne.fit_transform(X_sc)
            for lbl, color, name in [(0, 'steelblue', 'No Progression'),
                                      (1, 'crimson',   'Progression')]:
                mask = y_v == lbl
                axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                                c=color, label=name, alpha=0.7, s=50)
            axes[1].set_title("t-SNE")
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, 'Not enough samples for t-SNE',
                         ha='center', va='center', transform=axes[1].transAxes)

        plt.suptitle("CT-FM Embedding Visualisation", fontsize=13, fontweight='bold')
        plt.tight_layout()
        save = output_dir / "block1_embedding_visualisation.png"
        plt.savefig(save, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      Plot saved: {save}")

        # Quick linear separability check
        clf = LogisticRegression(C=0.1, max_iter=500, random_state=42)
        pca_full = PCA(n_components=min(50, n_patients - 1), random_state=42)
        X_pca50  = pca_full.fit_transform(X_sc)
        if len(np.unique(y_v)) > 1:
            cv_auc = cross_val_score(clf, X_pca50, y_v, cv=3, scoring='roc_auc').mean()
            print(f"      Logistic Regression on PCA-50 (3-fold CV AUC): {cv_auc:.3f}")
            if cv_auc < 0.52:
                verdict("WARNING", "Embeddings show no linear separability in PCA space. "
                                   "CT-FM signal may not be informative for this task.")
            else:
                verdict("PASS", f"Some linear separability detected (PCA-LR AUC = {cv_auc:.3f}).")

    # ── 1e. Feature distribution plot ────────────────────────────────────────
    print("\n  [1e] Feature value distribution")
    flat = X.flatten()
    print(f"      Mean:   {flat.mean():.4f}")
    print(f"      Std:    {flat.std():.4f}")
    print(f"      Median: {np.median(flat):.4f}")
    print(f"      [1%, 99%] percentile range: [{np.percentile(flat,1):.4f}, {np.percentile(flat,99):.4f}]")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(flat, bins=100, color='steelblue', alpha=0.8)
    ax.set_xlabel("Embedding value"); ax.set_ylabel("Count")
    ax.set_title("Distribution of all CT-FM embedding values")
    ax.grid(True, alpha=0.3)
    save = output_dir / "block1_embedding_distribution.png"
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Plot saved: {save}")


# =============================================================================
# BLOCK 2 – LABEL & FOLD SANITY
# =============================================================================

def block2_label_fold_sanity(df, feat_cols, splits, output_dir):
    header("BLOCK 2 – LABEL & FOLD SANITY")

    label_col  = CONFIG["label_col"]
    pat_col    = CONFIG["patient_col"]

    # ── 2a. Overall label distribution ───────────────────────────────────────
    print("\n  [2a] Overall label distribution")
    if label_col not in df.columns:
        verdict("FAIL", f"Label column '{label_col}' not found in CSV.")
        return
    label_counts = df[label_col].value_counts().sort_index()
    total = len(df)
    for lbl, cnt in label_counts.items():
        print(f"      Label {lbl}: {cnt} patients ({100*cnt/total:.1f}%)")

    pos_rate = label_counts.get(1, 0) / total
    if pos_rate < 0.1 or pos_rate > 0.9:
        verdict("WARNING", f"Severe class imbalance ({pos_rate:.1%} positive). "
                           "AUC estimates on small test folds will be noisy.")
    else:
        verdict("PASS", f"Class balance OK ({pos_rate:.1%} positive).")

    # ── 2b. Missing labels ────────────────────────────────────────────────────
    print("\n  [2b] Missing labels")
    n_missing = df[label_col].isnull().sum()
    print(f"      Missing labels: {n_missing}")
    if n_missing > 0:
        verdict("FAIL", f"{n_missing} patients have no label.")
    else:
        verdict("PASS", "No missing labels.")

    if splits is None:
        print("\n  ⚠️  No K-fold splits loaded – skipping fold checks.")
        return

    # ── 2c. Per-fold class balance ────────────────────────────────────────────
    print("\n  [2c] Per-fold label distribution")
    label_map = dict(zip(df[pat_col].astype(str), df[label_col]))

    fold_summary = []
    has_problem  = False

    for fold_idx, fold_data in sorted(splits.items()):
        for split_name in ['train', 'val', 'test']:
            ids = [str(p) for p in fold_data.get(split_name, [])]
            lbls = [label_map[p] for p in ids if p in label_map]
            n    = len(lbls)
            pos  = sum(lbls)
            neg  = n - pos
            pct  = 100 * pos / n if n > 0 else 0
            fold_summary.append({
                'fold': fold_idx, 'split': split_name,
                'n': n, 'pos': pos, 'neg': neg, 'pos_pct': round(pct, 1)
            })
            flag = ""
            if split_name == 'test':
                if pos == 0 or neg == 0:
                    flag = " ← ⚠️  SINGLE CLASS (AUC undefined!)"
                    has_problem = True
                elif pct < 10 or pct > 90:
                    flag = " ← ⚠️  severe imbalance"
                    has_problem = True
            print(f"      Fold {fold_idx} {split_name:5s}: {n:3d} patients, "
                  f"{pos} pos / {neg} neg ({pct:.0f}%){flag}")

    if has_problem:
        verdict("FAIL", "Some test folds have single-class or severely imbalanced labels. "
                        "AUC is undefined or unreliable for these folds. "
                        "Consider re-stratifying your splits.")
    else:
        verdict("PASS", "All folds have at least some positive and negative examples.")

    # ── 2d. Patient ID overlap between splits ─────────────────────────────────
    print("\n  [2d] Patient ID leakage check (train/val/test overlap per fold)")
    leakage_found = False
    for fold_idx, fold_data in sorted(splits.items()):
        tr  = set(str(p) for p in fold_data.get('train', []))
        val = set(str(p) for p in fold_data.get('val',   []))
        tst = set(str(p) for p in fold_data.get('test',  []))
        tv  = tr & val
        tt  = tr & tst
        vt  = val & tst
        if tv or tt or vt:
            print(f"      Fold {fold_idx}: train∩val={len(tv)}, train∩test={len(tt)}, val∩test={len(vt)}  ← LEAKAGE")
            leakage_found = True
    if not leakage_found:
        verdict("PASS", "No patient ID overlap between train/val/test splits.")
    else:
        verdict("FAIL", "Data leakage detected – patients appear in multiple splits.")

    # ── 2e. CT-FM patients not in splits ──────────────────────────────────────
    print("\n  [2e] Coverage check – CT-FM patients vs K-fold splits")
    all_split_ids = set()
    for fold_data in splits.values():
        for s in ['train', 'val', 'test']:
            all_split_ids.update(str(p) for p in fold_data.get(s, []))
    ctfm_ids = set(df[pat_col].astype(str))
    only_in_ctfm   = ctfm_ids - all_split_ids
    only_in_splits = all_split_ids - ctfm_ids
    print(f"      Patients in CT-FM CSV:    {len(ctfm_ids)}")
    print(f"      Patients in K-fold splits: {len(all_split_ids)}")
    print(f"      In CT-FM only (no split):  {len(only_in_ctfm)}")
    print(f"      In splits only (no embed): {len(only_in_splits)}")
    if only_in_splits:
        verdict("FAIL", f"{len(only_in_splits)} patients in splits have no CT-FM embedding. "
                        "These will be silently dropped during training.")
        print(f"      Missing patient IDs: {sorted(only_in_splits)[:10]} ...")
    else:
        verdict("PASS", "All split patients have a CT-FM embedding.")


# =============================================================================
# BLOCK 3 – SKLEARN BASELINE
# =============================================================================

def block3_sklearn_baseline(df, feat_cols, splits, output_dir):
    header("BLOCK 3 – SKLEARN BASELINE (ceiling check)")

    label_col = CONFIG["label_col"]
    pat_col   = CONFIG["patient_col"]

    if label_col not in df.columns:
        print("  Skipping – label column not found.")
        return

    X_all   = df[feat_cols].values.astype(np.float32)
    y_all   = df[label_col].values
    ids_all = df[pat_col].astype(str).values

    # Remove NaN rows
    valid = ~np.isnan(X_all).any(axis=1) & ~np.isnan(y_all)
    X_all, y_all, ids_all = X_all[valid], y_all[valid], ids_all[valid]

    if len(np.unique(y_all)) < 2:
        print("  Skipping – only one class present after filtering.")
        return

    print(f"\n  Dataset: {len(X_all)} patients, "
          f"{int(y_all.sum())} pos / {int((1-y_all).sum())} neg")

    # ── 3a. Simple cross-validation (no fold structure) ───────────────────────
    print("\n  [3a] 5-fold stratified CV on full dataset")
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_all)

    # PCA to 50 dims first (reduce dimensionality before LR)
    n_comp  = min(50, len(X_all) - 1, len(feat_cols))
    pca     = PCA(n_components=n_comp, random_state=42)
    X_pca   = pca.fit_transform(X_sc)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for name, clf in [
        ("LR (C=0.001)",  LogisticRegression(C=0.001, max_iter=1000, random_state=42)),
        ("LR (C=0.01)",   LogisticRegression(C=0.01,  max_iter=1000, random_state=42)),
        ("LR (C=0.1)",    LogisticRegression(C=0.1,   max_iter=1000, random_state=42)),
        ("SVM (RBF)",     SVC(kernel='rbf', C=1.0, probability=True, random_state=42)),
    ]:
        aucs = cross_val_score(clf, X_pca, y_all, cv=cv, scoring='roc_auc')
        results[name] = aucs
        print(f"      {name:20s}: AUC = {aucs.mean():.3f} ± {aucs.std():.3f}")

    best_name = max(results, key=lambda k: results[k].mean())
    best_auc  = results[best_name].mean()

    if best_auc < 0.52:
        verdict("FAIL", f"Best sklearn AUC = {best_auc:.3f} – CT-FM embeddings contain "
                        "NO discriminative signal for this task. "
                        "Problem is upstream (extraction/preprocessing).")
    elif best_auc < 0.60:
        verdict("WARNING", f"Best sklearn AUC = {best_auc:.3f} – weak but non-zero signal. "
                           "Neural network may be too complex for this dataset size.")
    else:
        verdict("PASS", f"Best sklearn AUC = {best_auc:.3f} – signal exists. "
                        "Neural network training/architecture may be the issue.")

    # ── 3b. Respect your actual K-fold splits ────────────────────────────────
    if splits is not None:
        print("\n  [3b] Logistic regression respecting your K-fold splits")
        id_to_x = dict(zip(ids_all, X_sc))
        id_to_y = dict(zip(ids_all, y_all))

        fold_aucs = []
        for fold_idx, fold_data in sorted(splits.items()):
            tr_ids  = [str(p) for p in fold_data['train'] if str(p) in id_to_x]
            tst_ids = [str(p) for p in fold_data['test']  if str(p) in id_to_x]

            X_tr = np.array([id_to_x[p] for p in tr_ids])
            y_tr = np.array([id_to_y[p] for p in tr_ids])
            X_ts = np.array([id_to_x[p] for p in tst_ids])
            y_ts = np.array([id_to_y[p] for p in tst_ids])

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_ts)) < 2:
                print(f"      Fold {fold_idx}: skipped (single class in train or test)")
                continue

            # PCA fitted on training only
            pca_fold = PCA(n_components=min(n_comp, len(X_tr)-1), random_state=42)
            X_tr_p   = pca_fold.fit_transform(X_tr)
            X_ts_p   = pca_fold.transform(X_ts)

            clf_fold = LogisticRegression(C=0.01, max_iter=1000, random_state=42)
            clf_fold.fit(X_tr_p, y_tr)
            probs    = clf_fold.predict_proba(X_ts_p)[:, 1]
            auc      = roc_auc_score(y_ts, probs)
            fold_aucs.append(auc)
            print(f"      Fold {fold_idx}: {len(tr_ids)} train, {len(tst_ids)} test → AUC = {auc:.3f}")

        if fold_aucs:
            mean_auc = np.mean(fold_aucs)
            print(f"\n      Mean fold AUC: {mean_auc:.3f} ± {np.std(fold_aucs):.3f}")

            if mean_auc < 0.52:
                verdict("FAIL", "Even simple LR on your exact splits gives AUC ≈ chance. "
                                "Embeddings are not informative, or folds are badly stratified.")
            elif mean_auc < 0.60:
                verdict("WARNING", "Weak signal on your exact splits. "
                                   "Neural network is likely overfitting given the dataset size.")
            else:
                verdict("PASS", f"LR baseline AUC = {mean_auc:.3f} on your exact splits. "
                                "Neural network training is the bottleneck.")

    # ── 3c. Majority-class baseline ───────────────────────────────────────────
    print("\n  [3c] Majority-class dummy baseline")
    majority_acc = max(y_all.mean(), 1 - y_all.mean())
    print(f"      Always-predict-majority accuracy: {majority_acc:.3f}")
    print(f"      (If your model accuracy ≈ {majority_acc:.3f}, it is predicting majority class only)")


# =============================================================================
# BLOCK 4 – PIPELINE SPOT-CHECK
# =============================================================================

def block4_pipeline_spotcheck(df, feat_cols, output_dir, n_check=5):
    header("BLOCK 4 – PIPELINE SPOT-CHECK (raw volumes)")

    raw_vol_dir = CONFIG["raw_volume_dir"]
    dicom_dir   = CONFIG["dicom_dir"]

    if not Path(raw_vol_dir).exists():
        print(f"  Raw volume dir not found: {raw_vol_dir}")
        print("  Skipping pipeline spot-check.")
        return

    npy_files = list(Path(raw_vol_dir).glob("*.npy"))
    if not npy_files:
        print("  No .npy files found in raw volume dir. Skipping.")
        return

    sample_files = npy_files[:n_check]
    print(f"\n  Spot-checking {len(sample_files)} patients from {raw_vol_dir}\n")

    fig, axes = plt.subplots(len(sample_files), 3,
                              figsize=(12, 4 * len(sample_files)))
    if len(sample_files) == 1:
        axes = axes[np.newaxis, :]

    hu_issues = []

    for row_idx, npy_path in enumerate(sample_files):
        patient_id = npy_path.stem
        try:
            vol = np.load(str(npy_path))   # (D, H, W), raw HU
            D, H, W = vol.shape

            print(f"  Patient: {patient_id}")
            print(f"    Volume shape: {vol.shape}")
            print(f"    HU range:     [{vol.min():.0f}, {vol.max():.0f}]")
            print(f"    Mean HU:      {vol.mean():.1f}  (chest CT typically ~-500 to -300)")
            print(f"    Std  HU:      {vol.std():.1f}")

            # Expected HU range sanity for chest CT
            expected_min  = -1100
            expected_max  = 1200
            typical_mean_lo = -700
            typical_mean_hi = -100

            if vol.min() < expected_min or vol.max() > expected_max:
                print(f"    ⚠️  HU range outside expected [{expected_min}, {expected_max}]")
                hu_issues.append(patient_id)
            if not (typical_mean_lo < vol.mean() < typical_mean_hi):
                print(f"    ⚠️  Mean HU {vol.mean():.0f} outside typical range "
                      f"[{typical_mean_lo}, {typical_mean_hi}] for chest CT")
                hu_issues.append(patient_id)

            # ── Plot 3 representative slices ──────────────────────────────
            slice_idxs = [D // 4, D // 2, 3 * D // 4]
            titles     = ["Superior (25%)", "Middle (50%)", "Inferior (75%)"]
            for col_idx, (si, title) in enumerate(zip(slice_idxs, titles)):
                sl = vol[si]
                # Display in standard chest window [-1000, 400]
                sl_display = np.clip((sl + 1000) / 1400, 0, 1)
                ax = axes[row_idx, col_idx]
                ax.imshow(sl_display, cmap='gray', origin='upper')
                ax.set_title(f"{patient_id[:12]}\n{title}\nHU µ={sl.mean():.0f}", fontsize=8)
                ax.axis('off')

            print()

        except Exception as e:
            print(f"    ERROR loading {npy_path.name}: {e}\n")

    plt.suptitle("Raw Volume Spot-Check (3 axial slices per patient)\n"
                 "Check: correct orientation, plausible HU, no artefacts",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    save = output_dir / "block4_volume_spotcheck.png"
    plt.savefig(save, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Slice visualisation saved: {save}")

    if hu_issues:
        verdict("WARNING", f"{len(set(hu_issues))} patients had suspicious HU values. "
                           "Check the visualisation plot for artefacts or wrong rescaling.")
    else:
        verdict("PASS", "HU values look plausible for all spot-checked patients.")

    # ── Slice ordering sanity (check via DICOM if available) ─────────────────
    if Path(dicom_dir).exists():
        print("\n  [4b] Slice ordering sanity (checking first patient in DICOM dir)")
        patient_dirs = [d for d in Path(dicom_dir).iterdir() if d.is_dir()]
        if patient_dirs:
            pd_dir  = patient_dirs[0]
            dcm_files = sorted(pd_dir.glob("*.dcm"))
            if dcm_files:
                try:
                    import pydicom
                    positions = []
                    for f in dcm_files[:20]:   # check first 20 slices
                        ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                        if hasattr(ds, 'ImagePositionPatient'):
                            positions.append(float(ds.ImagePositionPatient[2]))
                        elif hasattr(ds, 'InstanceNumber'):
                            positions.append(float(ds.InstanceNumber))

                    if positions:
                        diffs = np.diff(positions)
                        is_monotone = np.all(diffs > 0) or np.all(diffs < 0)
                        print(f"    Patient: {pd_dir.name}")
                        print(f"    First 5 Z positions: {[round(p,1) for p in positions[:5]]}")
                        print(f"    Spacing consistent: {np.allclose(np.abs(diffs), np.abs(diffs[0]), rtol=0.1)}")
                        if is_monotone:
                            verdict("PASS", "Slice ordering is monotonically increasing/decreasing.")
                        else:
                            verdict("WARNING", "Slice ordering is NOT monotone – check get_slice_position logic.")
                except ImportError:
                    print("    pydicom not installed – skipping DICOM check.")


# =============================================================================
# BLOCK 5 – QUICK SUMMARY REPORT
# =============================================================================

def block5_summary_report(df, feat_cols, splits, output_dir):
    header("BLOCK 5 – SUMMARY & RECOMMENDATIONS")

    label_col = CONFIG["label_col"]
    pat_col   = CONFIG["patient_col"]

    print("\n  Dataset at a glance:")
    print(f"    Patients:       {len(df)}")
    print(f"    Feature dims:   {len(feat_cols)}")

    if label_col in df.columns:
        pos = int(df[label_col].sum())
        neg = len(df) - pos
        print(f"    Positive:       {pos} ({100*pos/len(df):.1f}%)")
        print(f"    Negative:       {neg} ({100*neg/len(df):.1f}%)")

    if splits is not None:
        fold_test_sizes = []
        for fold_data in splits.values():
            fold_test_sizes.append(len(fold_data.get('test', [])))
        print(f"    K-fold folds:   {len(splits)}")
        print(f"    Mean test size: {np.mean(fold_test_sizes):.1f} patients")

    print("\n  Interpretation guide for AUC ~0.45:")
    print("    1. If Block 1 shows cosine sim > 0.95  →  fix CT-FM preprocessing")
    print("    2. If Block 2 shows single-class folds  →  re-stratify your splits")
    print("    3. If Block 3 sklearn AUC < 0.52        →  CT-FM embeddings carry no signal")
    print("       └─ Try: different HU window, skip CropForeground, check orientation")
    print("    4. If Block 3 sklearn AUC > 0.55 but NN AUC ≈ 0.45:")
    print("       └─ Problem is the neural network. Try:")
    print("          • Remove the 512→64 reduction layer (too many params for small dataset)")
    print("          • Use L2 regularisation (weight_decay ≥ 0.1)")
    print("          • Reduce epochs, increase early stopping patience")
    print("          • Use sklearn LR as final classifier instead of NN head")
    print("    5. If everything ≈ 0.5  →  task may be too hard / dataset too small")
    print("       └─ Check label definition quality and clinical relevance\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  CT-FM DIAGNOSTIC SUITE")
    print(f"  Output directory: {output_dir}")
    print(f"{'#'*70}")

    # Load data
    df, feat_cols = load_embeddings(CONFIG)
    splits        = load_kfold(CONFIG)

    # Normalise label column
    label_col = CONFIG["label_col"]
    if label_col not in df.columns:
        for candidate in ["label", "gt_has_progressed"]:
            if candidate in df.columns:
                df.rename(columns={candidate: label_col}, inplace=True)
                print(f"  Renamed '{candidate}' → '{label_col}'")
                break

    # Run all blocks
    block1_embedding_quality(df, feat_cols, output_dir)
    block2_label_fold_sanity(df, feat_cols, splits, output_dir)
    block3_sklearn_baseline(df, feat_cols, splits, output_dir)
    block4_pipeline_spotcheck(df, feat_cols, output_dir, n_check=CONFIG["n_spot_check"])
    block5_summary_report(df, feat_cols, splits, output_dir)

    print(f"\n{'#'*70}")
    print(f"  DIAGNOSTICS COMPLETE")
    print(f"  All plots saved to: {output_dir}")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()