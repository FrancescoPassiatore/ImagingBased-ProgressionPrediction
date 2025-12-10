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

def lung_seg_pixel_ratio(img_array):
    """Calcola numero pixel non-zero e area relativa."""
    c = np.count_nonzero(img_array)
    return c, round(c / (img_array.shape[0] * img_array.shape[1]), 4)

def make_lungmask(img):
    """
    Genera una maschera polmonare semplice (stessa logica del tuo script di selezione).
    Ritorna: img_norm, mask
    """
    img = img.astype(float)
    row_size, col_size = img.shape

    # Normalizzazione base
    mean = np.mean(img)
    std = np.std(img) if np.std(img) > 0 else 1.0
    img = (img - mean) / std

    # Regione centrale
    middle = img[int(col_size/5):int(col_size/5*4), int(row_size/5):int(row_size/5*4)]
    mean = np.mean(middle)
    max_val = np.max(img)
    min_val = np.min(img)

    img[img == max_val] = mean
    img[img == min_val] = mean

    # KMeans per soglia
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(
        np.reshape(middle, [np.prod(middle.shape), 1])
    )
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)

    # Pulizia
    eroded = morphology.erosion(thresh_img, np.ones([3, 3]))
    dilation = morphology.dilation(eroded, np.ones([8, 8]))

    labels = measure.label(dilation)
    regions = measure.regionprops(labels)
    good_labels = []

    for prop in regions:
        B = prop.bbox
        if ((B[2]-B[0] < row_size*0.9) and (B[3]-B[1] < col_size*0.9) and
            (B[2]-B[0] > row_size*0.20) and (B[3]-B[1] > col_size*0.10) and
            (B[0] > row_size*0.03) and (B[2] < row_size*0.97) and
            (B[1] > col_size*0.03) and (B[3] < col_size*0.97)):
            good_labels.append(prop.label)

    mask = np.zeros([row_size, col_size], dtype=np.int8)
    for N in good_labels:
        mask += np.where(labels == N, 1, 0)

    mask = morphology.dilation(mask, np.ones([10, 10]))

    return img, mask

def preprocess_slice(img, mask=None, target_size=(224, 224)):
    """
    img: array 2D (CT raw)
    mask: opzionale, stessa shape di img. Se presente, applichiamo img * mask.
    Ritorna immagine float32 normalizzata in [0,1] e ridimensionata.
    """
    if mask is not None:
        img = img * mask

    # Normalizzazione robusta usando percentili
    # (così sei meno sensibile agli outlier)
    v1, v99 = np.percentile(img, (1, 99))
    if v99 - v1 < 1e-6:
        # Evita divisione per 0: fallback a semplice scaling
        img_norm = img - v1
    else:
        img_norm = (img - v1) / (v99 - v1)

    img_norm = np.clip(img_norm, 0.0, 1.0)

    # Resize con OpenCV (cv2 usa shape (h, w))
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
                img = ds.pixel_array

                # Genera mask e preprocessa
                img_norm, mask = make_lungmask(img)
                img_pre = preprocess_slice(img_norm, mask=mask, target_size=TARGET_SIZE)

                # Salva come .npy
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
