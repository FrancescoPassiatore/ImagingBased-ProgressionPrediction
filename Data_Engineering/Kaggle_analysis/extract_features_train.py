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
path = 'C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/osic-pulmonary-fibrosis-progression/'
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

    return lung_pixels, slice_lung_area, num_t_pixels, tissue_by_total, tissue_by_lung


#Extract features patients
#Path train folder
tr_csv = pd.read_csv("C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/osic-pulmonary-fibrosis-progression/train.csv")

train_dicoms = "C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/osic-pulmonary-fibrosis-progression/train"

tr_csv['Where'] = 'train'

#Unique patient list
tr_patient_ids = tr_csv.Patient.unique().tolist()

#Image Feature extraction
percentile_range_1 = np.linspace(0.3,0.6,11)
percentile_range_2 = range(30,63, 3)
percentile_range = [(round(x,2),y) for x,y in zip(percentile_range_1,percentile_range_2)]

def build_image_features_for_split(split_name, patient_ids):
    """
    Calcola le feature delle immagini (30–60 percentile) per ogni paziente in train o test.
    
    split_name : 'train' o 'test'
    patient_ids: lista di ID pazienti
    """
    image_data = []
    counter = 1

    for pid in patient_ids:
        lung_list, num_t_pixels_list, tissue_by_total_list, tissue_by_lung_list = [], [], [], []
        flag = -1

        for perc in percentile_range:
            img_no = get_img(perc[0], pid, split_name)
            if img_no == flag:  # stessa slice già usata
                continue
            flag = img_no

            dcm_path = os.path.join(path, f"{split_name}/{pid}/{img_no}.dcm")

            try:
                img = load_image(dcm_path)
            except Exception as e:
                print(f"{pid}/{img_no} Err load_image:", e)
                continue

            try:
                lung_pixels, area_ratio, num_t_pixels, tissue_by_total, tissue_by_lung = make_lungmask(img, display=False)
                if not math.isnan(lung_pixels):
                    lung_list.append((perc[0], lung_pixels))
                if not math.isnan(num_t_pixels):
                    num_t_pixels_list.append(num_t_pixels)
                if not math.isnan(tissue_by_total):
                    tissue_by_total_list.append(tissue_by_total)
                if not math.isnan(tissue_by_lung):
                    tissue_by_lung_list.append(tissue_by_lung)
            except Exception as e:
                print(split_name, pid, img_no, e)
                pass

        # Metadati DICOM
        try:
            meta_dcm = get_dicom_meta(dcm_path)
            slice_thickness = meta_dcm["SliceThickness"]
            pixel_spacing   = meta_dcm["PixelSpacing"]
        except:
            slice_thickness = 0.0
            pixel_spacing   = 0.0

        # Medie
        try:
            Avg_NumTissuePixel_30_60   = round(sum(num_t_pixels_list)/len(num_t_pixels_list),4)
            Avg_Tissue_30_60           = round((sum(num_t_pixels_list)/len(num_t_pixels_list))*pixel_spacing,4)
            Avg_Tissue_thickness_30_60 = round((sum(num_t_pixels_list)/len(num_t_pixels_list))*pixel_spacing*slice_thickness,4)
            Avg_TissueByTotal_30_60    = round(sum(tissue_by_total_list)/len(tissue_by_total_list),4)
            Avg_TissueByLung_30_60     = round(sum(tissue_by_lung_list)/len(tissue_by_lung_list),4)
        except:
            Avg_NumTissuePixel_30_60 = Avg_Tissue_30_60 = Avg_Tissue_thickness_30_60 = 0
            Avg_TissueByTotal_30_60 = Avg_TissueByLung_30_60 = 0

        # Numero di slice tra due percentili
        try:
            num_im = num_img_bw_perc(percentile_range[0][0], percentile_range[1][0], pid, split_name)
        except:
            num_im = 0
        if num_im == 0:
            num_im = 1

        # Volume approssimato
        approx_vol = 0
        for x in lung_list[:-1]:
            approx_vol += (x[1] * pixel_spacing * slice_thickness * num_im)

        patient_dict = {
            "Patient": pid,
            "Data": split_name,
            "SliceThickness": slice_thickness,
            "PixelSpacing": pixel_spacing,
            "NumImgBw5Prec": num_im,
            "ApproxVol_30_60": round(approx_vol,4),
            "Avg_NumTissuePixel_30_60": Avg_NumTissuePixel_30_60,
            "Avg_Tissue_30_60": Avg_Tissue_30_60,
            "Avg_Tissue_thickness_30_60": Avg_Tissue_thickness_30_60,
            "Avg_TissueByTotal_30_60": Avg_TissueByTotal_30_60,
            "Avg_TissueByLung_30_60": Avg_TissueByLung_30_60,
        }
        image_data.append(patient_dict)

        if counter % 20 == 0:
            print(f"[{split_name}] processed: {counter}")
        counter += 1

    return pd.DataFrame(image_data)


image_data_df_train = build_image_features_for_split('train', tr_patient_ids)
tr = tr_csv.merge(image_data_df_train, on='Patient', how='left')
tr.to_csv("train_df_with_img_feat.csv", index=False)
print("Saved train_df_with_img_feat.csv:", tr.shape)


