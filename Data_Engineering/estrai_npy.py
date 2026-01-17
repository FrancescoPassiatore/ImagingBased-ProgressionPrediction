import os
import glob
import numpy as np
import pydicom
import cv2
from skimage import morphology, measure
import matplotlib.pyplot as plt

# ==========================
# CONFIGURAZIONE PERCORSI
# ==========================
# Cartella con le slice già selezionate (output del tuo script precedente)
INPUT_DIR = r"C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/extracted"

# Cartella dove salveremo le .npy preprocessate
OUTPUT_DIR = r"C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/extracted_npy"

# Dimensione target per il modello
TARGET_SIZE = (224, 224)  # (width, height)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# FUNZIONI DI SUPPORTO
# ==========================

def dicom_to_hu(ds):
    """Convert DICOM pixel array to Hounsfield Units."""
    img = ds.pixel_array.astype(np.float32)
    
    slope = float(ds.RescaleSlope) if 'RescaleSlope' in ds else 1.0
    intercept = float(ds.RescaleIntercept) if 'RescaleIntercept' in ds else 0.0
    
    img = img * slope + intercept
    return img

def lung_seg_pixel_ratio(img_array):
    """Calcola numero pixel non-zero e area relativa."""
    c = np.count_nonzero(img_array)
    return c, round(c / (img_array.shape[0] * img_array.shape[1]), 4)

def make_lungmask(img):
    """
    Genera una maschera polmonare da immagine in HU.
    Ritorna: mask binaria
    """
    img_norm = (img - np.mean(img)) / (np.std(img) + 1e-6)
    
    row_size, col_size = img.shape
    middle = img_norm[int(col_size/5):int(col_size/5*4),
                      int(row_size/5):int(row_size/5*4)]
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(
        middle.reshape(-1, 1)
    )
    threshold = np.mean(sorted(kmeans.cluster_centers_.flatten()))
    thresh_img = np.where(img_norm < threshold, 1.0, 0.0)
    
    eroded = morphology.erosion(thresh_img, np.ones((3, 3)))
    dilated = morphology.dilation(eroded, np.ones((8, 8)))
    
    labels = measure.label(dilated)
    mask = np.zeros_like(img, dtype=np.uint8)
    
    for prop in measure.regionprops(labels):
        y0, x0, y1, x1 = prop.bbox
        if 0.2 * row_size < (y1 - y0) < 0.9 * row_size:
            mask[labels == prop.label] = 1
    
    mask = morphology.dilation(mask, np.ones((10, 10)))
    return mask

def preprocess_slice_hu(img_hu, mask, target_size=(224, 224)):
    """
    Preprocess CT slice in Hounsfield Units with soft masking.
    
    Args:
        img_hu: immagine in Hounsfield Units
        mask: lung mask binaria
        target_size: dimensione output (width, height)
    
    Returns:
        immagine float32 normalizzata [0,1] e ridimensionata
    """
    # 1. Clip HU a range clinicamente significativo
    img_hu = np.clip(img_hu, -1000, 400)
    
    # 2. Soft masking: mantiene contesto al 10% fuori dai polmoni
    img_soft = img_hu * mask + img_hu * 0.1 * (1 - mask)
    
    # 3. Normalizzazione globale HU → [0,1]
    # -1000 HU (aria) → 0, +400 HU (osso) → 1
    img_norm = (img_soft + 1000) / 1400
    img_norm = np.clip(img_norm, 0.0, 1.0)
    
    # 4. Resize
    img_resized = cv2.resize(img_norm, target_size, interpolation=cv2.INTER_AREA)
    
    return img_resized.astype(np.float32)

# ==========================
# MAIN: DICOM -> NPY
# ==========================

def process_all_patients():
    patients = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    print(f"Trovati {len(patients)} pazienti in {INPUT_DIR}")

    for idx, patient_id in enumerate(patients, 1):
        print(f"\n[{idx}/{len(patients)}] Paziente: {patient_id}")
        patient_folder = os.path.join(INPUT_DIR, patient_id)

        # Cartella output per questo paziente
        patient_out = os.path.join(OUTPUT_DIR, patient_id)
        os.makedirs(patient_out, exist_ok=True)

        dcm_files = sorted(glob.glob(os.path.join(patient_folder, "*.dcm")))
        print(f"  Slice trovate: {len(dcm_files)}")

        for dcm_path in dcm_files:
            fname = os.path.basename(dcm_path)
            base_name = os.path.splitext(fname)[0]
            out_path = os.path.join(patient_out, base_name + ".npy")

            # Se esiste già, salta (utile se riesegui lo script)
            if os.path.exists(out_path):
                continue

            try:
                ds = pydicom.dcmread(dcm_path)
                
                # 1. Converti a Hounsfield Units
                img_hu = dicom_to_hu(ds)
                
                # 2. Genera lung mask
                mask = make_lungmask(img_hu)
                
                # 3. Preprocessa con soft masking e normalizzazione globale HU
                img_pre = preprocess_slice_hu(img_hu, mask, target_size=TARGET_SIZE)

                # 4. Salva come .npy
                np.save(out_path, img_pre)

            except Exception as e:
                print(f"  ⚠️ Errore su {fname}: {e}")

    print("\n✅ Preprocessing completato! Tutte le slice sono state salvate in formato .npy.")

# ==========================
# VISUALIZZAZIONE DI ALCUNE IMMAGINI
# ==========================

def show_some_examples(n_patients=2, n_slices_per_patient=2):
    """
    Mostra qualche immagine preprocessata (le .npy) come controllo visivo.
    """
    patients = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
    patients = sorted(patients)[:n_patients]

    samples = []

    for p in patients:
        p_folder = os.path.join(OUTPUT_DIR, p)
        npy_files = sorted(glob.glob(os.path.join(p_folder, "*.npy")))[:n_slices_per_patient]
        for npy_path in npy_files:
            img = np.load(npy_path)
            samples.append((p, os.path.basename(npy_path), img))

    if not samples:
        print("Nessuna immagine trovata da mostrare.")
        return

    n = len(samples)
    cols = 2
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(6 * cols, 4 * rows))
    for i, (patient_id, fname, img) in enumerate(samples, 1):
        plt.subplot(rows, cols, i)
        plt.imshow(img, cmap="gray")
        plt.title(f"{patient_id} - {fname}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# ==========================
# ESECUZIONE
# ==========================

if __name__ == "__main__":
    process_all_patients()
    print("\n🔍 Visualizzo qualche immagine preprocessata...")
    show_some_examples(n_patients=2, n_slices_per_patient=2)
