import pandas as pd
import numpy as np
import cv2
import pydicom 
import os
import ast
import matplotlib.pyplot as plt

#-----------------------------------------
import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
import seaborn as sns
import math
import cv2
import pydicom
import os
from tqdm import tqdm
import glob
import pickle as pkl
import matplotlib.image as mpimg
from tabulate import tabulate
import missingno as msno
from IPython.display import display_html
from PIL import Image
import gc
from skimage.transform import resize
import copy
import re
from scipy.stats import pearsonr

# Segmentation
import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *
init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings("ignore")

#------------------------------------------------------------

pd.set_option("display.max_columns", 100)

custom_colors = ['#74a09e','#86c1b2','#98e2c6','#f3c969','#f2a553', '#d96548', '#c14953']
#sns.palplot(sns.color_palette(custom_colors))

#Define resolution of image 512*512
N_ROWS = 512
N_COLS = 512
path = 'C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/CodeDevelopment/osic-pulmonary-fibrosis-progression/'
#Define Batch size
BATCH_SIZE=128

# Areas with the same number of pixels on the edges are not required. Crop it.
def crop_image(img: np.ndarray):
    edge_pixel_value = img[0, 0]
    mask = img != edge_pixel_value
    return img[np.ix_(mask.any(1),mask.any(0))]

# Load images, crop thick borders(if any) and resize
def load_image(path):
    dataset = pydicom.dcmread(path)
    img = dataset.pixel_array
    img = crop_image(img)
#   img = cv2.resize(img, (512,512))
    return img

# Get Nth percentile image
def get_img(perc, patient_id, data):

    l = glob.glob(path+'{0}/{1}/*.dcm'.format(data, patient_id))
    img_ids = []
    for x in l:
        y = x.split('\\')[-1]
        z = int(y.split('.')[0])
        img_ids.append(z)

    img_ids.sort()

    return img_ids[math.ceil(perc*(len(img_ids)))-1]

# Get num of slices bw two percentiles
def num_img_bw_perc(p1, p2, patient_id, data):

    l = glob.glob(path+'{0}/{1}/*.dcm'.format(data, patient_id))
    img_ids = []
    for x in l:
        y = x.split('\\')[-1]
        z = int(y.split('.')[0])
        img_ids.append(z)

    img_ids.sort()

    return len(img_ids[math.ceil(p1*(len(img_ids)))-1:math.ceil(p2*(len(img_ids)))])-1


# Get number of images per patient
def get_num_images(patient_id, data):

    return len(glob.glob(path+'{0}/{1}/*.dcm'.format(data, patient_id)))

# Get the lung area in the image slice
def lung_seg_pixel_ratio(img_array):

    c = 0
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            if img_array[i][j] != 0:
                c+=1

    return c, round(c/(img_array.shape[0]*img_array.shape[1]),4)

# Get dicom meta data
def get_dicom_meta(path):

    '''Get information from the .dcm files.
    path: complete path to the .dcm file'''

    image_data = pydicom.dcmread(path)

    # Dictionary to store the information from the image
    observation_data = {
        "SliceThickness" : float(image_data.SliceThickness),
        "PixelSpacing" : float(image_data.PixelSpacing[0]),
    }

    return observation_data

# Get tissue mask
# To extract the tissues from the segmented lung all we need to do is get rid of the border parts from the segmented lung
# Grey pixels present within the border of the lung is assumed to be tissue.
# Inorder to get rid of the border pixels of the lung we slightly perturb the segmented lung to the right, left, top and bottom
# The intersection of all the perturbed images gets rid of the border lung pixels
# This resultant image serves as the mask for the tissue segmentation
def tissue_mask(img, mask, shift_perc):

    r_dim, c_dim = img.shape[0], img.shape[1]

    # Move the image by shift_perc to the left
    del_left_cols = int(shift_perc*c_dim)

    mask1, mask2 = mask.copy(), np.zeros((r_dim, c_dim)).astype(int)
    mask1 = mask1[:,del_left_cols:]
    mask2[:,:c_dim-del_left_cols] = mask1

    # Move the image by shift_perc to the right
    del_right_cols = int(shift_perc*c_dim)

    mask3, mask4 = mask.copy(), np.zeros((r_dim, c_dim)).astype(int)
    mask3 = mask3[:,:c_dim-del_right_cols]
    mask4[:,del_right_cols:] = mask3

    # Move the image by shift_perc to the top
    del_top_rows = int(shift_perc*c_dim) #BUG C_DIM SHOULD BE R_DIM

    mask5, mask6 = mask.copy(), np.zeros((r_dim, c_dim)).astype(int)
    mask5 = mask5[del_top_rows:,:]
    mask6[:r_dim-del_top_rows,:] = mask5

    # Move the image by shift_perc to the bottom
    del_bottom_rows = int(shift_perc*r_dim)

    mask7, mask8 = mask.copy(), np.zeros((r_dim, c_dim)).astype(int)
    mask7 = mask7[:r_dim-del_bottom_rows,:]
    mask8[del_bottom_rows:,:] = mask7

    #Obtain the final mask
    final_mask = ((mask2==1) & (mask4==1) & (mask6==1) & (mask8==1)).astype(int)

    return final_mask

# Get tissue features
def tissue_features(tissue_mask, img, thresh = 0.35):

    final_img = tissue_mask*img

    checker = np.zeros((final_img.shape[0], final_img.shape[1]))
    counter, other_counter = 0, 0
    for i in range(final_img.shape[0]):
        for j in range(final_img.shape[1]):
            if final_img[i][j]>=thresh:
                checker[i][j] = 1
                counter+=1
            else:
                checker[i][j] = 0
                other_counter+=1

    tissue_by_total = counter/(final_img.shape[0]**2)
    tissue_by_lung = counter/((tissue_mask==1).sum())

    return counter, tissue_by_total, tissue_by_lung, checker

# https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
def make_lungmask(img, display=False):

    img = img.astype(float)
    row_size= img.shape[0]
    col_size = img.shape[1]


    #Normalize image pixels
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std

    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)]
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)

    # To improve threshold finding, I'm moving the
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean
    #Substitutes too light/dark pixels with mean
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.
    # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img,np.ones([3,3]))
    dilation = morphology.dilation(eroded,np.ones([8,8]))

    labels = measure.label(dilation) # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if ((B[2]-B[0]<row_size*0.9) and (B[3]-B[1]<col_size*0.9) and (B[2]-B[0]>row_size*0.20)
            and (B[3]-B[1]>col_size*0.10) and (B[0]>row_size*0.03) and (B[2]<row_size*0.97)
            and (B[1]>col_size*0.03) and (B[3]<col_size*0.97)):
            good_labels.append(prop.label)
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0

    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask

    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
 # Compute Lung Area in the slice
    lung_pixels, slice_lung_area = lung_seg_pixel_ratio(mask)

    # Tissue Mask
    t_mask = tissue_mask(img, mask, shift_perc = 0.02)

    # Extract tissue features
    num_t_pixels, tissue_by_total, tissue_by_lung, checker = tissue_features(t_mask, img, thresh = 0.35)

    if (display):
        fig, ax = plt.subplots(4, 2, figsize=[18, 18])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask*img, cmap='gray')
        ax[2, 1].axis('off')
        ax[3, 0].set_title("Inner Lung Mask")
        ax[3, 0].imshow(t_mask, cmap='gray')
        ax[3, 0].axis('off')
        ax[3, 1].set_title("Segmented Tissue")
        ax[3, 1].imshow(checker, cmap='gray')
        ax[3, 1].axis('off')

        plt.show()

    return img,thresh_img,dilation,labels,mask,mask*img,t_mask,checker

#----------------------------------------
data_path = 'C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/CodeDevelopment/Code/patient_event_slice.csv'
data_path2 = 'C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/extracted'

#df -> patient_event_slice.csv --- | Patient | Event | Time | Slice files extracted from percentile |
df_event_time = pd.read_csv(data_path)

df_event_time['slice_files'] = df_event_time['slice_files'].apply(ast.literal_eval)

# ============================================================
# PREPROCESSING CORRETTO PER CT SCAN
# ============================================================

def preprocess_ct_correct(dcm_path, apply_mask=True):
    """
    Preprocessing corretto per CT scan
    
    Steps:
    1. Carica DICOM raw (HU units)
    2. Applica lung mask
    3. Window/Level per polmoni (-1000 a +400 HU)
    4. Normalizza a [0, 1]
    5. Converti a RGB (3 canali)
    6. Normalizza con ImageNet stats (per pre-trained model)
    """
    
    # STEP 1: Carica DICOM
    ds = pydicom.dcmread(dcm_path)
    img_raw = ds.pixel_array.astype(np.float32)
    
    # STEP 2: Segmentazione polmonare
    if apply_mask:
        img_norm, _, _, _, mask, img_masked, _, _ = make_lungmask(img_raw.copy())
        # ✅ USA img_masked (non img_norm)
        img_to_process = img_masked
    else:
        img_to_process = img_raw
    
    # STEP 3: Lung Window (-1000 to +400 HU)
    # Nota: make_lungmask già normalizza, quindi potremmo avere valori strani
    # MEGLIO: Applica mask a img_raw PRIMA di normalizzare
    
    # Riapplica mask a img_raw
    if apply_mask:
        img_raw_masked = img_raw * mask  # ✅ Applica mask a raw HU values
    else:
        img_raw_masked = img_raw
    
    # Lung window clipping
    img_windowed = np.clip(img_raw_masked, -1000, 400)
    
    # STEP 4: Normalizza a [0, 1]
    # (img - min) / (max - min)
    img_normalized = (img_windowed + 1000) / 1400  # Range [-1000, 400] → [0, 1]
    
    # STEP 5: Resize
    img_resized = cv2.resize(img_normalized, (224, 224))
    
    # STEP 6: Converti a RGB (3 canali)
    img_rgb = np.stack([img_resized, img_resized, img_resized], axis=0)  # [3, 224, 224]
    
    # STEP 7: Normalizza con ImageNet stats
    # (Necessario per transfer learning con EfficientNet pre-trained)
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]
    img_final = (img_rgb - mean) / std
    
    return img_final, mask


# ============================================================
# VISUALIZZAZIONE PER DEBUGGING
# ============================================================

def visualize_preprocessing_steps(dcm_path):
    """
    Visualizza tutti gli step di preprocessing
    """
    # Carica raw
    ds = pydicom.dcmread(dcm_path)
    img_raw = ds.pixel_array.astype(np.float32)
    
    # Segmentazione
    img_norm, thresh_img, dilation, labels, mask, img_masked, t_mask, checker = make_lungmask(img_raw.copy())
    
    # Applica mask a raw
    img_raw_masked = img_raw * mask
    
    # Window
    img_windowed = np.clip(img_raw_masked, -1000, 400)
    img_normalized = (img_windowed + 1000) / 1400
    
    # Resize
    img_resized = cv2.resize(img_normalized, (224, 224))
    
    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    axes[0, 0].imshow(img_norm, cmap='gray')
    axes[0, 0].set_title('1. DICOM Raw', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title('2. Lung Mask', fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img_raw_masked, cmap='gray')
    axes[0, 2].set_title('3. Masked Raw', fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(img_windowed, cmap='gray')
    axes[0, 3].set_title('4. Windowed [-1000,400]', fontweight='bold')
    axes[0, 3].axis('off')
    
    axes[1, 0].imshow(img_normalized, cmap='gray')
    axes[1, 0].set_title('5. Normalized [0,1]', fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img_resized, cmap='gray')
    axes[1, 1].set_title('6. Resized 224x224', fontweight='bold')
    axes[1, 1].axis('off')
    
    # ImageNet normalized (mostra solo 1 canale)
    img_rgb = np.stack([img_resized, img_resized, img_resized], axis=0)
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]
    img_final = (img_rgb - mean) / std
    
    axes[1, 2].imshow(img_final[0], cmap='gray', vmin=-2, vmax=2)
    axes[1, 2].set_title('7. ImageNet Normalized', fontweight='bold')
    axes[1, 2].axis('off')
    
    # Histogram
    axes[1, 3].hist(img_final[0].flatten(), bins=100, alpha=0.7)
    axes[1, 3].set_title('8. Final Distribution', fontweight='bold')
    axes[1, 3].set_xlabel('Pixel Value')
    axes[1, 3].set_ylabel('Frequency')
    axes[1, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 STATISTICS:")
    print(f"Raw range: [{img_raw.min():.0f}, {img_raw.max():.0f}]")
    print(f"Masked range: [{img_raw_masked.min():.0f}, {img_raw_masked.max():.0f}]")
    print(f"Windowed range: [{img_windowed.min():.0f}, {img_windowed.max():.0f}]")
    print(f"Normalized range: [{img_normalized.min():.3f}, {img_normalized.max():.3f}]")
    print(f"Final range: [{img_final.min():.3f}, {img_final.max():.3f}]")


# ============================================================
# TEST SU UN PAZIENTE
# ============================================================

# Seleziona un paziente casuale
test_patient = df_event_time.iloc[0]
test_slice = test_patient['slice_files'][0]
test_path = os.path.join(data_path2, test_patient['Patient'], test_slice)

print(f"Testing preprocessing on: {test_patient['Patient']}/{test_slice}")
visualize_preprocessing_steps(test_path)
