import numpy as np
import pandas as pd
import os
import math
import cv2
import pydicom
from tqdm import tqdm
import glob
from sklearn.cluster import KMeans
from skimage import morphology, measure
from scipy.stats import skew, kurtosis
from scipy.ndimage import label
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for CT feature extraction"""
    N_ROWS = 512
    N_COLS = 512
    
    # Regional percentile ranges for analysis
    UPPER_LUNG = (0.15, 0.35)   # Upper 15-35%
    MIDDLE_LUNG = (0.35, 0.65)  # Middle 35-65%
    LOWER_LUNG = (0.65, 0.85)   # Lower 65-85%
    
    # Tissue threshold for fibrosis detection
    TISSUE_THRESHOLD = 0.35
    
    # Tissue mask shift percentage
    TISSUE_SHIFT_PERC = 0.02


# ============================================================================
# IMAGE LOADING & PREPROCESSING
# ============================================================================

def crop_image(img: np.ndarray) -> np.ndarray:
    """Remove uniform borders from image"""
    if img.size == 0:
        return img
    edge_pixel_value = img[0, 0]
    mask = img != edge_pixel_value
    rows = mask.any(1)
    cols = mask.any(0)
    if not rows.any() or not cols.any():
        return img
    return img[np.ix_(rows, cols)]


def load_dicom_image(path: str) -> np.ndarray:
    """Load and preprocess DICOM image"""
    try:
        dataset = pydicom.dcmread(path)
        img = dataset.pixel_array.astype(float)
        img = crop_image(img)
        return img
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def get_dicom_metadata(path: str) -> dict:
    """Extract metadata from DICOM file"""
    try:
        dataset = pydicom.dcmread(path)
        return {
            "SliceThickness": float(dataset.SliceThickness),
            "PixelSpacing": float(dataset.PixelSpacing[0]),
        }
    except Exception as e:
        print(f"Error reading metadata from {path}: {e}")
        return {"SliceThickness": 0.0, "PixelSpacing": 0.0}


# ============================================================================
# PATIENT SCAN UTILITIES
# ============================================================================

def get_patient_slices(patient_id: str, data_folder: str, base_path: str) -> list:
    """Get sorted list of slice numbers for a patient"""
    pattern = os.path.join(base_path, data_folder, patient_id, "*.dcm")
    files = glob.glob(pattern)
    
    slice_ids = []
    for filepath in files:
        filename = os.path.basename(filepath)
        try:
            slice_id = int(filename.split('.')[0])
            slice_ids.append(slice_id)
        except ValueError:
            continue
    
    return sorted(slice_ids)


def get_slice_at_percentile(slice_ids: list, percentile: float) -> int:
    """Get slice ID at given percentile"""
    if not slice_ids:
        return None
    idx = max(0, min(len(slice_ids) - 1, math.ceil(percentile * len(slice_ids)) - 1))
    return slice_ids[idx]


def get_slices_in_range(slice_ids: list, perc_start: float, perc_end: float) -> list:
    """Get all slice IDs between two percentiles"""
    if not slice_ids:
        return []
    
    start_idx = max(0, math.ceil(perc_start * len(slice_ids)) - 1)
    end_idx = min(len(slice_ids), math.ceil(perc_end * len(slice_ids)))
    
    return slice_ids[start_idx:end_idx]


# ============================================================================
# LUNG SEGMENTATION
# ============================================================================

def segment_lung(img: np.ndarray, display: bool = False) -> tuple:
    """
    Segment lung from CT image using K-means clustering
    
    Returns:
        mask: Binary lung mask
        lung_pixels: Number of lung pixels
        lung_area_ratio: Ratio of lung pixels to total image pixels
    """
    if img is None or img.size == 0:
        return None, 0, 0.0
    
    img = img.astype(float)
    row_size, col_size = img.shape
    
    # Normalize image
    mean = np.mean(img)
    std = np.std(img)
    if std > 0:
        img = (img - mean) / std
    
    # Focus on middle region for threshold finding
    r_start, r_end = int(row_size/5), int(row_size*4/5)
    c_start, c_end = int(col_size/5), int(col_size*4/5)
    middle = img[r_start:r_end, c_start:c_end]
    
    mean = np.mean(middle)
    max_val = np.max(img)
    min_val = np.min(img)
    
    # Replace extreme values with mean
    img[img == max_val] = mean
    img[img == min_val] = mean
    
    # K-means clustering to separate foreground and background
    try:
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(
            middle.reshape(-1, 1)
        )
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
    except:
        threshold = 0
    
    # Threshold image
    thresh_img = (img < threshold).astype(float)
    
    # Morphological operations
    eroded = morphology.erosion(thresh_img, np.ones([3, 3]))
    dilated = morphology.dilation(eroded, np.ones([8, 8]))
    
    # Label connected components
    labels = measure.label(dilated)
    regions = measure.regionprops(labels)
    
    # Filter regions based on size and position
    good_labels = []
    for prop in regions:
        bbox = prop.bbox
        if ((bbox[2] - bbox[0] < row_size * 0.9) and 
            (bbox[3] - bbox[1] < col_size * 0.9) and
            (bbox[2] - bbox[0] > row_size * 0.20) and
            (bbox[3] - bbox[1] > col_size * 0.10) and
            (bbox[0] > row_size * 0.03) and 
            (bbox[2] < row_size * 0.97) and
            (bbox[1] > col_size * 0.03) and 
            (bbox[3] < col_size * 0.97)):
            good_labels.append(prop.label)
    
    # Create mask
    mask = np.zeros([row_size, col_size], dtype=np.int8)
    for label_id in good_labels:
        mask = mask + (labels == label_id).astype(np.int8)
    
    # Final dilation
    mask = morphology.dilation(mask, np.ones([10, 10]))
    
    # Calculate lung area
    lung_pixels = np.count_nonzero(mask)
    total_pixels = row_size * col_size
    lung_area_ratio = lung_pixels / total_pixels if total_pixels > 0 else 0
    
    return mask, lung_pixels, lung_area_ratio


# ============================================================================
# TISSUE SEGMENTATION
# ============================================================================

def create_tissue_mask(img: np.ndarray, lung_mask: np.ndarray, 
                       shift_perc: float = 0.02) -> np.ndarray:
    """
    Create tissue mask by removing lung border pixels
    FIXED: Bug in del_top_rows calculation (was using c_dim instead of r_dim)
    """
    if lung_mask is None or lung_mask.size == 0:
        return None
    
    r_dim, c_dim = lung_mask.shape
    
    # Shift left
    del_left_cols = int(shift_perc * c_dim)
    mask_left = np.zeros((r_dim, c_dim), dtype=int)
    mask_left[:, :c_dim - del_left_cols] = lung_mask[:, del_left_cols:]
    
    # Shift right
    del_right_cols = int(shift_perc * c_dim)
    mask_right = np.zeros((r_dim, c_dim), dtype=int)
    mask_right[:, del_right_cols:] = lung_mask[:, :c_dim - del_right_cols]
    
    # Shift top (FIXED: now using r_dim)
    del_top_rows = int(shift_perc * r_dim)
    mask_top = np.zeros((r_dim, c_dim), dtype=int)
    mask_top[:r_dim - del_top_rows, :] = lung_mask[del_top_rows:, :]
    
    # Shift bottom
    del_bottom_rows = int(shift_perc * r_dim)
    mask_bottom = np.zeros((r_dim, c_dim), dtype=int)
    mask_bottom[del_bottom_rows:, :] = lung_mask[:r_dim - del_bottom_rows, :]
    
    # Intersection of all shifted masks
    tissue_mask = ((mask_left == 1) & (mask_right == 1) & 
                   (mask_top == 1) & (mask_bottom == 1)).astype(int)
    
    return tissue_mask


def extract_tissue_features(img: np.ndarray, lung_mask: np.ndarray, 
                           tissue_mask: np.ndarray, threshold: float = 0.35) -> dict:
    """
    Extract tissue-related features (potential fibrosis indicators)
    
    Returns dictionary with tissue metrics
    """
    if tissue_mask is None or tissue_mask.size == 0:
        return {
            "num_tissue_pixels": 0,
            "tissue_by_total": 0.0,
            "tissue_by_lung": 0.0
        }
    
    # Apply tissue mask to image
    tissue_img = tissue_mask * img
    
    # Count high-attenuation pixels (potential fibrosis)
    high_attenuation = (tissue_img >= threshold).astype(int)
    num_tissue_pixels = np.count_nonzero(high_attenuation)
    
    # Calculate ratios
    total_pixels = img.shape[0] * img.shape[1]
    lung_pixels = np.count_nonzero(lung_mask)
    
    tissue_by_total = num_tissue_pixels / total_pixels if total_pixels > 0 else 0
    tissue_by_lung = num_tissue_pixels / lung_pixels if lung_pixels > 0 else 0
    
    return {
        "num_tissue_pixels": num_tissue_pixels,
        "tissue_by_total": tissue_by_total,
        "tissue_by_lung": tissue_by_lung
    }


# ============================================================================
# HISTOGRAM & TEXTURE FEATURES
# ============================================================================

def extract_histogram_features(img: np.ndarray, mask: np.ndarray) -> dict:
    """
    Extract statistical features from lung pixel histogram
    """
    if mask is None or not mask.any():
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "p10": 0.0,
            "p90": 0.0
        }
    
    lung_pixels = img[mask == 1]
    
    if lung_pixels.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "p10": 0.0,
            "p90": 0.0
        }
    
    return {
        "mean": float(np.mean(lung_pixels)),
        "std": float(np.std(lung_pixels)),
        "median": float(np.median(lung_pixels)),
        "skewness": float(skew(lung_pixels)),
        "kurtosis": float(kurtosis(lung_pixels)),
        "p10": float(np.percentile(lung_pixels, 10)),
        "p90": float(np.percentile(lung_pixels, 90))
    }


def extract_texture_features(img: np.ndarray, mask: np.ndarray) -> dict:
    """
    Extract simple texture features (variance, entropy)
    """
    if mask is None or not mask.any():
        return {
            "texture_variance": 0.0,
            "texture_entropy": 0.0
        }
    
    lung_region = img[mask == 1]
    
    if lung_region.size == 0:
        return {
            "texture_variance": 0.0,
            "texture_entropy": 0.0
        }
    
    # Local variance (simple approximation)
    variance = float(np.var(lung_region))
    
    # Entropy calculation
    hist, _ = np.histogram(lung_region, bins=50, density=True)
    hist = hist[hist > 0]  # Remove zero bins
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    return {
        "texture_variance": variance,
        "texture_entropy": float(entropy)
    }


# ============================================================================
# SLICE-LEVEL FEATURE EXTRACTION
# ============================================================================

def extract_slice_features(img_path: str, img: np.ndarray, metadata: dict) -> dict:
    """
    Extract all features from a single CT slice
    """
    features = {
        "slice_path": img_path,
        "slice_thickness": metadata.get("SliceThickness", 0.0),
        "pixel_spacing": metadata.get("PixelSpacing", 0.0)
    }
    
    if img is None:
        return {**features, **get_empty_slice_features()}
    
    # Lung segmentation
    lung_mask, lung_pixels, lung_area_ratio = segment_lung(img)
    
    features["lung_pixels"] = lung_pixels
    features["lung_area_ratio"] = lung_area_ratio
    
    if lung_mask is None:
        return {**features, **get_empty_slice_features()}
    
    # Tissue segmentation
    tissue_mask = create_tissue_mask(img, lung_mask, Config.TISSUE_SHIFT_PERC)
    tissue_feats = extract_tissue_features(img, lung_mask, tissue_mask, 
                                          Config.TISSUE_THRESHOLD)
    
    # Histogram features
    hist_feats = extract_histogram_features(img, lung_mask)
    
    # Texture features
    texture_feats = extract_texture_features(img, lung_mask)
    
    # Combine all features
    features.update(tissue_feats)
    features.update(hist_feats)
    features.update(texture_feats)
    
    return features


def get_empty_slice_features() -> dict:
    """Return empty feature dictionary for failed slices"""
    return {
        "lung_pixels": 0,
        "lung_area_ratio": 0.0,
        "num_tissue_pixels": 0,
        "tissue_by_total": 0.0,
        "tissue_by_lung": 0.0,
        "mean": 0.0,
        "std": 0.0,
        "median": 0.0,
        "skewness": 0.0,
        "kurtosis": 0.0,
        "p10": 0.0,
        "p90": 0.0,
        "texture_variance": 0.0,
        "texture_entropy": 0.0
    }


# ============================================================================
# PATIENT-LEVEL AGGREGATION
# ============================================================================

def aggregate_slice_features(slice_features_list: list, region_name: str) -> dict:
    """
    Aggregate features from multiple slices into patient-level statistics
    """
    if not slice_features_list:
        return get_empty_aggregated_features(region_name)
    
    df = pd.DataFrame(slice_features_list)
    
    # Remove rows with zero lung pixels (failed segmentation)
    df = df[df['lung_pixels'] > 0]
    
    if len(df) == 0:
        return get_empty_aggregated_features(region_name)
    
    aggregated = {
        f"{region_name}_num_slices": len(df),
    }
    
    # Features to aggregate
    features_to_agg = [
        'lung_pixels', 'lung_area_ratio', 'num_tissue_pixels',
        'tissue_by_total', 'tissue_by_lung', 'mean', 'std', 'median',
        'skewness', 'kurtosis', 'p10', 'p90', 
        'texture_variance', 'texture_entropy'
    ]
    
    for feat in features_to_agg:
        if feat in df.columns:
            values = df[feat].dropna()
            if len(values) > 0:
                aggregated[f"{region_name}_{feat}_mean"] = float(values.mean())
                aggregated[f"{region_name}_{feat}_std"] = float(values.std())
                aggregated[f"{region_name}_{feat}_min"] = float(values.min())
                aggregated[f"{region_name}_{feat}_max"] = float(values.max())
    
    return aggregated


def get_empty_aggregated_features(region_name: str) -> dict:
    """Return empty aggregated features for a region"""
    return {f"{region_name}_num_slices": 0}


def calculate_volume_metrics(slice_features_list: list, metadata: dict) -> dict:
    """
    Calculate approximate lung volume from slice features
    """
    if not slice_features_list:
        return {"approx_lung_volume": 0.0}
    
    pixel_spacing = metadata.get("PixelSpacing", 0.0)
    slice_thickness = metadata.get("SliceThickness", 0.0)
    
    if pixel_spacing == 0 or slice_thickness == 0:
        return {"approx_lung_volume": 0.0}
    
    # Calculate voxel volume
    voxel_volume = pixel_spacing * pixel_spacing * slice_thickness
    
    # Sum lung pixels across all slices
    total_lung_pixels = sum([s.get('lung_pixels', 0) for s in slice_features_list])
    
    # Approximate volume
    approx_volume = total_lung_pixels * voxel_volume
    
    return {"approx_lung_volume": approx_volume}


# ============================================================================
# MAIN PATIENT FEATURE EXTRACTION
# ============================================================================

def extract_patient_features(patient_id: str, data_folder: str, 
                            base_path: str) -> dict:
    """
    Extract comprehensive features for a single patient
    Processes all slices and returns regional + global features
    """
    print(f"Processing patient: {patient_id}")
    
    # Get all slices for patient
    slice_ids = get_patient_slices(patient_id, data_folder, base_path)
    
    if not slice_ids:
        print(f"  No slices found for {patient_id}")
        return get_empty_patient_features(patient_id, data_folder)
    
    # Get metadata from first slice
    first_slice_path = os.path.join(base_path, data_folder, patient_id, 
                                    f"{slice_ids[0]}.dcm")
    metadata = get_dicom_metadata(first_slice_path)
    
    # Define regions
    regions = {
        'upper': get_slices_in_range(slice_ids, *Config.UPPER_LUNG),
        'middle': get_slices_in_range(slice_ids, *Config.MIDDLE_LUNG),
        'lower': get_slices_in_range(slice_ids, *Config.LOWER_LUNG),
        'full': slice_ids
    }
    
    patient_features = {
        "patient_id": patient_id,
        "data_split": data_folder,
        "total_slices": len(slice_ids),
        "slice_thickness": metadata["SliceThickness"],
        "pixel_spacing": metadata["PixelSpacing"]
    }
    
    # Process each region
    for region_name, region_slices in regions.items():
        slice_features_list = []
        
        for slice_id in region_slices:
            slice_path = os.path.join(base_path, data_folder, patient_id, 
                                     f"{slice_id}.dcm")
            img = load_dicom_image(slice_path)
            
            if img is not None:
                slice_feats = extract_slice_features(slice_path, img, metadata)
                slice_features_list.append(slice_feats)
        
        # Aggregate region features
        region_agg = aggregate_slice_features(slice_features_list, region_name)
        patient_features.update(region_agg)
        
        # Calculate volume for full lung
        if region_name == 'full':
            volume_metrics = calculate_volume_metrics(slice_features_list, metadata)
            patient_features.update(volume_metrics)
    
    return patient_features


def get_empty_patient_features(patient_id: str, data_folder: str) -> dict:
    """Return empty feature dictionary for failed patients"""
    return {
        "patient_id": patient_id,
        "data_split": data_folder,
        "total_slices": 0
    }


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def extract_features_for_cohort(patient_ids: list, data_folder: str, 
                               base_path: str) -> pd.DataFrame:
    """
    Extract features for all patients in a cohort (train or test)
    """
    all_patient_features = []
    
    for i, patient_id in enumerate(tqdm(patient_ids, desc=f"Processing {data_folder}")):
        try:
            patient_feats = extract_patient_features(patient_id, data_folder, base_path)
            all_patient_features.append(patient_feats)
        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            all_patient_features.append(get_empty_patient_features(patient_id, data_folder))
    
    return pd.DataFrame(all_patient_features)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    BASE_PATH = "C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/CodeDevelopment/osic-pulmonary-fibrosis-progression/"
    
    # Load patient IDs
    train_csv = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
    train_patient_ids = train_csv['Patient'].unique().tolist()
    
    print(f"Found {len(train_patient_ids)} patients in training set")
    
    # Extract features
    print("\n=== Extracting Training Features ===")
    train_features_df = extract_features_for_cohort(
        train_patient_ids, 
        'train', 
        BASE_PATH
    )
    
    # Save results
    output_path = os.path.join(BASE_PATH, "patient_features_improved.csv")
    train_features_df.to_csv(output_path, index=False)
    print(f"\nFeatures saved to: {output_path}")
    
    # Display summary
    print("\n=== Feature Extraction Summary ===")
    print(f"Total patients processed: {len(train_features_df)}")
    print(f"Total features extracted: {len(train_features_df.columns)}")
    print(f"\nSample features:")
    print(train_features_df.head())
    
    print("\n=== Feature Statistics ===")
    print(train_features_df.describe())