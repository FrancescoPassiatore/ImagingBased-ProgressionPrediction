"""
dicom_to_ctfm.py
================
Full pipeline: DICOM slices → stacked 3D volume → CT-FM 512-dim embeddings.

Replaces the old ResNet50 slice-based preprocessing script.

What this script does
---------------------
Stage 1 – DICOM → raw 3D volume (.npy)
    Reads all DICOM slices for each patient, sorts them by ImagePositionPatient
    (the correct anatomical ordering), converts to Hounsfield Units, and saves
    the raw 3D volume as a single .npy file  (D, H, W)  in HU values.
    No masking, no clipping, no normalisation – that is all handled by the
    CT-FM preprocessing pipeline in Stage 2.

Stage 2 – 3D volume → CT-FM embedding
    Applies the exact preprocessing pipeline from the CT-FM model card:
        Orientation  → SPL
        ScaleIntensityRange → HU [-1024, 2048] mapped to [0, 1], clipped
        CropForeground → removes air background
    Runs the SegResEncoder (lighter_zoo) with a SlidingWindowInferer to handle
    large volumes without OOM errors, then global-average-pools the deepest
    feature map to produce one 512-dim vector per patient.

Stage 3 – Save results
    All embeddings are written to a single CSV:
        patient_id | label | volume_feature_0 | ... | volume_feature_511
    Extraction resumes automatically if interrupted (already-processed patients
    are skipped).

Requirements
------------
    pip install pydicom numpy pandas torch monai lighter_zoo -U

Usage
-----
    python dicom_to_ctfm.py

    Edit the CONFIG block at the top to set your paths, device, and label CSV.
"""

import os
import glob
import csv
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F

# MONAI preprocessing transforms
from monai.transforms import (
    Compose,
    EnsureType,
    Orientation,
    ScaleIntensityRange,
    CropForeground,
)
from monai.inferers import SlidingWindowInferer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# CONFIG – edit these paths before running
# =============================================================================
CONFIG = {
    # Folder that contains one sub-folder per patient, each with .dcm files
    # (same INPUT_DIR as your old script)
    "dicom_dir": Path(r"D:\FrancescoP\train"),

    # Where to save the intermediate raw 3D .npy volumes (one per patient)
    # These are in raw HU values – reusable for any future 3D model
    "raw_volume_dir": Path(r"D:\FrancescoP\raw_volumes"),

    # Where to write the final CSV of CT-FM embeddings
    "output_csv": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Data_Engineering\CT-FM_extractor\ctfm_embeddings.csv"),

    # CSV that maps patient_id → label (0 or 1).
    # Must contain columns: 'Patient' and 'label' (or adjust label_col below).
    "ground_truth_csv": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\ground_truth.csv"),
    "patient_col": "PatientID",   # column name for patient ID in ground_truth_csv
    "label_col":   "has_progressed",     # column name for label in ground_truth_csv

    # CT-FM HuggingFace model id
    "ctfm_model_id": "project-lighter/ct_fm_feature_extractor",

    # HU window used by CT-FM (do NOT change – matches model pretraining)
    "hu_min": -1024,
    "hu_max":  2048,

    # SlidingWindowInferer settings
    # roi_size: the 3D patch size fed to the model at each window step.
    # (96,96,96) works well on 8–16 GB VRAM; reduce to (64,64,64) if OOM.
    "sliding_roi_size": (96, 96, 96),
    "sliding_overlap":  0.25,

    # Computation device
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Save a progress checkpoint every N patients during CT-FM inference
    "save_every_n": 5,
}
# =============================================================================


# =============================================================================
# STAGE 1 – DICOM → RAW 3D VOLUME
# =============================================================================

def dicom_to_hu(ds: pydicom.Dataset) -> np.ndarray:
    """Convert a DICOM dataset pixel array to Hounsfield Units (float32)."""
    img = ds.pixel_array.astype(np.float32)
    slope     = float(getattr(ds, "RescaleSlope",     1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return img * slope + intercept


def get_slice_position(ds: pydicom.Dataset) -> float:
    """
    Return the Z-position of a slice for anatomical ordering.
    Uses ImagePositionPatient[2] (most reliable) with a fallback to
    InstanceNumber or SliceLocation.
    """
    if hasattr(ds, "ImagePositionPatient"):
        return float(ds.ImagePositionPatient[2])
    if hasattr(ds, "SliceLocation"):
        return float(ds.SliceLocation)
    if hasattr(ds, "InstanceNumber"):
        return float(ds.InstanceNumber)
    return 0.0


def _is_scout_or_localiser(ds: pydicom.Dataset) -> bool:
    """
    Return True if this DICOM file is a scout, localiser, dose report,
    or any non-volumetric image that should be excluded from the CT stack.
    These were causing 90mm Z-position gaps in the diagnostic check.
    """
    # Check ImageType (e.g. LOCALIZER, SCOUT)
    image_type = list(getattr(ds, "ImageType", []))
    image_type_str = " ".join(str(t).upper() for t in image_type)
    scout_keywords = ["LOCALIZER", "LOCALISER", "SCOUT", "DOSE",
                      "REPORT", "PROJECTION", "SCREEN SAVE"]
    if any(kw in image_type_str for kw in scout_keywords):
        return True

    # Check Modality – must be CT
    if str(getattr(ds, "Modality", "")).upper() not in ("CT", ""):
        return True

    # Check SOPClassUID for known non-volumetric types
    non_ct_sop = {
        "1.2.840.10008.5.1.4.1.1.7",      # Secondary Capture
        "1.2.840.10008.5.1.4.1.1.104.1",  # Encapsulated PDF (dose report)
        "1.2.840.10008.5.1.4.1.1.88.67",  # X-Ray Radiation Dose SR
    }
    if str(getattr(ds, "SOPClassUID", "")) in non_ct_sop:
        return True

    return False


def build_raw_volume(patient_dcm_dir: Path) -> np.ndarray:
    """
    Read DICOM files in a patient folder, select the best volumetric CT series,
    and return a (D, H, W) float32 HU array.

    Series selection strategy (fixes the scout-slice contamination found in
    diagnostics where Z gaps of 90mm were corrupting CT-FM embeddings):
      1. Read metadata only (fast) for all .dcm files
      2. Exclude scouts / localisers / dose files via _is_scout_or_localiser()
      3. Group remaining slices by SeriesInstanceUID
      4. Select the series with the most slices (= the main volumetric CT)
      5. Sort by Z position and load pixel data for the selected series only
    """
    from collections import defaultdict

    patient_dcm_dir = Path(patient_dcm_dir)
    dcm_files = sorted(patient_dcm_dir.glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files found in {patient_dcm_dir}")

    # ── Step 1: read metadata only (no pixel data, fast) ─────────────────────
    all_meta = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            all_meta.append((f, ds))
        except Exception as e:
            log.debug(f"  Metadata read failed {f.name}: {e}")

    if not all_meta:
        raise ValueError(f"No readable DICOM files in {patient_dcm_dir}")

    # ── Step 2: exclude scouts and localisers ─────────────────────────────────
    volumetric = [(f, ds) for f, ds in all_meta if not _is_scout_or_localiser(ds)]
    n_excluded = len(all_meta) - len(volumetric)
    if n_excluded:
        log.info(f"    Excluded {n_excluded} scout/localiser/non-CT file(s)")
    if not volumetric:
        raise ValueError(f"No volumetric CT slices after exclusion in {patient_dcm_dir}")

    # ── Step 3: group by SeriesInstanceUID ────────────────────────────────────
    series_groups: dict = defaultdict(list)
    for f, ds in volumetric:
        uid = str(getattr(ds, "SeriesInstanceUID", "UNKNOWN"))
        series_groups[uid].append((f, ds))

    if len(series_groups) > 1:
        log.info(f"    {len(series_groups)} series found – selecting largest:")
        for uid, slices in sorted(series_groups.items(), key=lambda x: -len(x[1])):
            desc = str(getattr(slices[0][1], "SeriesDescription", "N/A"))
            log.info(f"      UID …{uid[-12:]}: {len(slices):3d} slices  [{desc}]")

    # ── Step 4: pick the series with the most slices ──────────────────────────
    best_uid, best_slices = max(series_groups.items(), key=lambda x: len(x[1]))
    best_desc = str(getattr(best_slices[0][1], "SeriesDescription", "N/A"))
    log.info(f"    → Selected: UID …{best_uid[-12:]} "
             f"({len(best_slices)} slices, '{best_desc}')")

    if len(best_slices) < 50:
        log.warning(f"    Only {len(best_slices)} slices – volume may be too thin for CT-FM")

    # ── Step 5: sort by Z, load pixel data for selected series only ───────────
    best_slices.sort(key=lambda x: get_slice_position(x[1]))

    hu_slices = []
    for f, _ in best_slices:
        try:
            ds_full = pydicom.dcmread(str(f))
            hu_slices.append(dicom_to_hu(ds_full))
        except Exception as e:
            log.error(f"    Pixel load failed {f.name}: {e}")

    if not hu_slices:
        raise ValueError(f"Failed to load pixel data from chosen series in {patient_dcm_dir}")

    volume = np.stack(hu_slices, axis=0)  # (D, H, W)
    log.info(f"    Volume: {volume.shape}  HU [{volume.min():.0f}, {volume.max():.0f}]")
    return volume.astype(np.float32)


def stage1_dicom_to_raw_volumes(config: dict, allowed_patient_ids: set = None) -> dict:
    """
    Iterate over patient folders, convert DICOM stacks to raw HU volumes using
    series-aware build_raw_volume(), and save each as raw_volume_dir/<patient_id>.npy.

    Args:
        config              : CONFIG dict
        allowed_patient_ids : if provided, only process these patient IDs

    Returns a dict: {patient_id: Path_to_npy}
    """
    dicom_dir   = Path(config["dicom_dir"])
    raw_vol_dir = Path(config["raw_volume_dir"])
    raw_vol_dir.mkdir(parents=True, exist_ok=True)

    all_dirs = sorted([d for d in dicom_dir.iterdir() if d.is_dir()])

    if allowed_patient_ids is not None:
        patient_dirs = [d for d in all_dirs if d.name in allowed_patient_ids]
        log.info(f"Stage 1 – {len(all_dirs)} folders found, "
                 f"processing {len(patient_dirs)} matching ground truth")
    else:
        patient_dirs = all_dirs
        log.info(f"Stage 1 – {len(patient_dirs)} patient folders in {dicom_dir}")

    volume_paths  = {}
    thin_patients = []   # < 50 slices after series selection
    failed        = []

    for i, patient_dir in enumerate(patient_dirs, 1):
        patient_id = patient_dir.name
        out_path   = raw_vol_dir / f"{patient_id}.npy"

        if out_path.exists():
            log.info(f"  [{i}/{len(patient_dirs)}] {patient_id} – already exists, skipping")
            volume_paths[patient_id] = out_path
            continue

        log.info(f"  [{i}/{len(patient_dirs)}] {patient_id} – building volume …")
        try:
            volume = build_raw_volume(patient_dir)
            if volume.shape[0] < 50:
                thin_patients.append((patient_id, volume.shape[0]))
            np.save(str(out_path), volume)
            log.info(f"    Saved {volume.shape}  →  {out_path.name}")
            volume_paths[patient_id] = out_path
        except Exception as e:
            log.error(f"    FAILED: {e}")
            failed.append(patient_id)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info(f"\nStage 1 complete – {len(volume_paths)} volumes saved.")
    if thin_patients:
        log.warning("  ⚠ Thin volumes (<50 slices) – CT-FM quality may be poor:")
        for pid, d in thin_patients:
            log.warning(f"    {pid}: {d} slices")
    if failed:
        log.error(f"  ✗ Failed patients: {failed}")
    log.info("")
    return volume_paths


# =============================================================================
# STAGE 2 – RAW 3D VOLUME → CT-FM EMBEDDING
# =============================================================================

def load_ctfm_model(model_id: str, device: str):
    """Download and return the CT-FM SegResEncoder in eval mode."""
    try:
        from lighter_zoo import SegResEncoder
    except ImportError:
        raise ImportError(
            "lighter_zoo is required.\n"
            "Install with:  pip install lighter_zoo -U"
        )
    log.info(f"Loading CT-FM model from '{model_id}' …")
    model = SegResEncoder.from_pretrained(model_id)
    model.eval()
    model.to(device)
    log.info("CT-FM model loaded and ready.\n")
    return model


def build_ctfm_preprocess(hu_min: float, hu_max: float) -> Compose:
    """
    Exact preprocessing pipeline from the CT-FM model card.
    Input:  torch.Tensor of shape (1, D, H, W) in raw HU values
    Output: preprocessed tensor ready for the model
    """
    return Compose([
        EnsureType(),                        # ensure float tensor
        Orientation(axcodes="SPL"),          # standardise orientation
        ScaleIntensityRange(                 # HU → [0, 1]
            a_min=hu_min,
            a_max=hu_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForeground(allow_smaller=True),  # remove air background
    ])


def extract_ctfm_embedding(
    volume_npy: np.ndarray,
    model,
    preprocess: Compose,
    inferer: SlidingWindowInferer,
    device: str,
) -> np.ndarray:
    """
    Run the full CT-FM inference on one 3D volume.

    Args:
        volume_npy : (D, H, W) float32 numpy array in HU
        model      : loaded SegResEncoder
        preprocess : MONAI Compose pipeline
        inferer    : SlidingWindowInferer
        device     : 'cuda' or 'cpu'

    Returns:
        embedding : (512,) float32 numpy array
    """
    # Add channel dim → (1, D, H, W)
    tensor = torch.from_numpy(volume_npy).unsqueeze(0)

    # Apply preprocessing (Orientation, ScaleIntensityRange, CropForeground)
    tensor = preprocess(tensor)             # (1, D', H', W')

    # Add batch dim → (1, 1, D', H', W')
    batch = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        # SlidingWindowInferer calls model(patch) for each window.
        # CT-FM returns a list of feature maps; we need the last (deepest) one.
        # We wrap the model to extract only that final map for the inferer.
        def model_last_feature(x):
            outputs = model(x)   # list of tensors
            return outputs[-1]   # (B, 512, d, h, w)

        # Run sliding window inference → (1, 512, d, h, w)
        feature_map = inferer(batch, model_last_feature)

        # Global average pool over spatial dims → (1, 512, 1, 1, 1) → (512,)
        embedding = F.adaptive_avg_pool3d(feature_map, 1).squeeze().cpu().numpy()

    return embedding.astype(np.float32)


def stage2_extract_embeddings(
    volume_paths: dict,
    label_map: dict,
    config: dict,
) -> pd.DataFrame:
    """
    For every patient in volume_paths, run CT-FM inference and collect a
    512-dim embedding.  Results are written to config['output_csv'] incrementally.

    Args:
        volume_paths : {patient_id: Path_to_raw_npy}
        label_map    : {patient_id: int label}
        config       : CONFIG dict

    Returns:
        DataFrame with columns: patient_id | label | volume_feature_0..511
    """
    output_csv = Path(config["output_csv"])
    device     = config["device"]
    log.info(f"Stage 2 – CT-FM inference on {len(volume_paths)} patients  (device: {device})\n")

    # ---- Resume: load already-processed patients ----
    done_ids = set()
    existing_rows = []
    if output_csv.exists():
        existing_df = pd.read_csv(output_csv)
        done_ids    = set(existing_df["patient_id"].astype(str).tolist())
        existing_rows = existing_df.to_dict("records")
        log.info(f"  Resuming – {len(done_ids)} patients already in CSV.\n")

    # ---- Load model and build pipelines ----
    model      = load_ctfm_model(config["ctfm_model_id"], device)
    preprocess = build_ctfm_preprocess(config["hu_min"], config["hu_max"])
    inferer    = SlidingWindowInferer(
        roi_size    = config["sliding_roi_size"],
        sw_batch_size = 1,
        overlap     = config["sliding_overlap"],
        mode        = "gaussian",       # smoother blending at patch boundaries
        progress    = False,
    )

    # ---- Feature column names ----
    feat_cols = [f"volume_feature_{i}" for i in range(512)]

    all_rows   = list(existing_rows)   # start with already-done patients
    todo       = [(pid, path) for pid, path in sorted(volume_paths.items())
                  if str(pid) not in done_ids]
    total_todo = len(todo)

    for idx, (patient_id, npy_path) in enumerate(todo, 1):
        label = label_map.get(str(patient_id), -1)
        log.info(f"  [{idx}/{total_todo}] {patient_id}  (label={label}) …")

        try:
            volume    = np.load(str(npy_path))               # (D, H, W) HU
            embedding = extract_ctfm_embedding(
                volume, model, preprocess, inferer, device
            )
            row = {"patient_id": patient_id, "label": label}
            row.update(dict(zip(feat_cols, embedding.tolist())))
            all_rows.append(row)
            log.info(f"    ✓ embedding shape: {embedding.shape}")

        except Exception as e:
            log.error(f"    ✗ FAILED: {e}")
            row = {"patient_id": patient_id, "label": label}
            row.update({c: float("nan") for c in feat_cols})
            all_rows.append(row)

        # Save progress every N patients
        if idx % config["save_every_n"] == 0 or idx == total_todo:
            pd.DataFrame(all_rows).to_csv(output_csv, index=False)
            log.info(f"    → progress saved to {output_csv}  ({idx}/{total_todo} done)")

    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)

    # ---- Sanity checks ----
    n_nan = df[feat_cols].isnull().any(axis=1).sum()
    log.info(f"\nStage 2 complete:")
    log.info(f"  Total patients:  {len(df)}")
    log.info(f"  Failed (NaN):    {n_nan}")
    log.info(f"  Output CSV:      {output_csv}")
    log.info(f"  CSV shape:       {df.shape}\n")

    return df


# =============================================================================
# STAGE 3 – VALIDATION & QUICK REPORT
# =============================================================================

def stage3_validate(df: pd.DataFrame, config: dict):
    """
    Print a brief validation report so you can sanity-check the embeddings
    before passing them to the ablation study.
    """
    feat_cols = [f"volume_feature_{i}" for i in range(512)]

    print("\n" + "="*60)
    print("CT-FM EMBEDDING VALIDATION REPORT")
    print("="*60)
    print(f"Total patients:      {len(df)}")
    print(f"Embedding dimension: {len(feat_cols)}")

    # Label distribution
    if "label" in df.columns:
        print(f"\nLabel distribution:")
        for lbl, cnt in df["label"].value_counts().sort_index().items():
            print(f"  {lbl}: {cnt} patients")

    # NaN check
    nan_mask = df[feat_cols].isnull().any(axis=1)
    if nan_mask.any():
        print(f"\n⚠ Patients with NaN embeddings ({nan_mask.sum()}):")
        print(df[nan_mask]["patient_id"].tolist())
    else:
        print("\n✓ No NaN embeddings.")

    # Basic embedding stats (across all patients and features)
    flat = df[feat_cols].values.flatten()
    flat = flat[~np.isnan(flat)]
    print(f"\nEmbedding value statistics:")
    print(f"  mean:  {flat.mean():.4f}")
    print(f"  std:   {flat.std():.4f}")
    print(f"  min:   {flat.min():.4f}")
    print(f"  max:   {flat.max():.4f}")
    print("="*60 + "\n")


# =============================================================================
# HELPERS
# =============================================================================

def load_label_map(config: dict) -> dict:
    """
    Load the ground-truth CSV and return {patient_id_str: label_int}.
    """
    gt_path = Path(config["ground_truth_csv"])
    if not gt_path.exists():
        log.warning(f"Ground-truth CSV not found: {gt_path}  – labels will be -1")
        return {}

    df = pd.read_csv(gt_path)
    p_col = config["patient_col"]
    l_col = config["label_col"]

    if p_col not in df.columns or l_col not in df.columns:
        log.warning(f"Expected columns '{p_col}' and '{l_col}' in {gt_path}. "
                    f"Found: {df.columns.tolist()}")
        return {}

    label_map = {str(row[p_col]): int(row[l_col]) for _, row in df.iterrows()}
    log.info(f"Loaded {len(label_map)} labels from {gt_path}")
    return label_map


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*60)
    print("  DICOM → RAW VOLUME → CT-FM EMBEDDINGS")
    print("="*60 + "\n")

    log.info(f"Device: {CONFIG['device']}")
    log.info(f"Sliding window ROI: {CONFIG['sliding_roi_size']}")
    log.info(f"HU window: [{CONFIG['hu_min']}, {CONFIG['hu_max']}]\n")

    # --- Load label map first so Stage 1 only processes relevant patients ---
    label_map = load_label_map(CONFIG)
    allowed_patient_ids = set(label_map.keys()) if label_map else None

    if allowed_patient_ids:
        log.info(f"Will process {len(allowed_patient_ids)} patients from ground truth\n")

    # --- Stage 1: DICOM → raw .npy volumes (series-aware, ground-truth filtered) ---
    volume_paths = stage1_dicom_to_raw_volumes(CONFIG, allowed_patient_ids)

    # --- Stage 2: raw volume → CT-FM 512-dim embedding ---
    embeddings_df = stage2_extract_embeddings(volume_paths, label_map, CONFIG)

    # --- Stage 3: validation report ---
    stage3_validate(embeddings_df, CONFIG)

    print(f"Done!  Embeddings saved to:\n  {CONFIG['output_csv']}\n")
    print("Pass this CSV as 'volume_features_df' to your ablation study.")


if __name__ == "__main__":
    main()