# -----------------------------------------------------Regular Imports
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


# Regular Imports
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
import warnings
warnings.filterwarnings("ignore")

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

# Model imports
import tensorflow as tf
from tensorflow.keras.layers import (
                                    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, GlobalAveragePooling2D,
                                    Add, Conv2D, AveragePooling2D, LeakyReLU, Concatenate , Lambda
                                    )
from tensorflow.keras import Model
import tensorflow.keras.models as M
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K
import tensorflow.keras.applications as tfa
import efficientnet.tfkeras as efn
from sklearn.model_selection import train_test_split, KFold
import seaborn as sns


pd.set_option("display.max_columns", 100)
custom_colors = ['#74a09e','#86c1b2','#98e2c6','#f3c969','#f2a553', '#d96548', '#c14953']
sns.palplot(sns.color_palette(custom_colors))

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

EPOCHS = 30
BATCH_SIZE = 8
NFOLD = 5
LR = 0.003
SAVE_BEST = True
MODEL_CLASS = 'b1'
path = 'C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/osic-pulmonary-fibrosis-progression/'

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
#     img = cv2.resize(img, (512,512))
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

# https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
def make_lungmask(img, display=False):

    img = img.astype(float)

    row_size= img.shape[0]
    col_size = img.shape[1]

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
#         if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:
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
    num_t_pixels, tissue_by_total, tissue_by_lung = tissue_features(t_mask, img, thresh = 0.35)

    if (display):
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
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

        plt.show()

    return lung_pixels, slice_lung_area, num_t_pixels, tissue_by_total, tissue_by_lung

# Get tissue mask
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
    del_top_rows = int(shift_perc*c_dim)

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

#     if tissue_by_lung>1:
#         tissue_by_total = (counter-(tissue_mask==0).sum())/(rescaled.shape[0]**2)
#         tissue_by_lung = (counter-(tissue_mask==0).sum())/((tissue_mask==1).sum())
#         counter = counter-(tissue_mask==0).sum()

    return counter, tissue_by_total, tissue_by_lung

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

def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_tab(df):
    vector = [(df.Age.values[0] - 30) / 30]

    if df.Sex.values[0].lower() == 'male':
        vector.append(0)
    else:
        vector.append(1)

    if df.SmokingStatus.values[0] == 'Never smoked':
        vector.extend([0,0])
    elif df.SmokingStatus.values[0] == 'Ex-smoker':
        vector.extend([1,1])
    elif df.SmokingStatus.values[0] == 'Currently smokes':
        vector.extend([0,1])
    else:
        vector.extend([1,0])

    if df.Avg_Tissue_30_60_Quartile.values[0] == 'Q1':
        vector.extend([0])
    elif df.Avg_Tissue_30_60_Quartile.values[0] == 'Q2':
        vector.extend([1])
    elif df.Avg_Tissue_30_60_Quartile.values[0] == 'Q3':
        vector.extend([2])
    elif df.Avg_Tissue_30_60_Quartile.values[0] == 'Q4':
        vector.extend([3])
    else:
        vector.extend([4])

    vector.extend([df.Avg_Tissue_30_60.values[0]])

    return np.array(vector)

def get_img_1(path):
    d = pydicom.dcmread(path)
    return cv2.resize((d.pixel_array - d.RescaleIntercept) / (d.RescaleSlope * 1000), (512, 512))

def score(fvc_true, fvc_pred, sigma):
    sigma_clip = np.maximum(sigma, 70) # changed from 70, trie 66.7 too
    delta = np.abs(fvc_true - fvc_pred)
    delta = np.minimum(delta, 1000)
    sq2 = np.sqrt(2)
    metric = (delta / sigma_clip)*sq2 + np.log(sigma_clip* sq2)
    return np.mean(metric)

seed_everything(42)

tr = pd.read_csv('C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/ProgressionCode/train_df_with_img_feat.csv')

ts = pd.read_csv(f"{path}/test.csv")

ts['Where'] = 'test'

# Fetch unique patient ids
ts_patient_ids = ts.Patient.unique().tolist()

percentile_range_1 = np.linspace(0.3,0.6,11)
percentile_range_2 = range(30,63, 3)
percentile_range = [(round(x,2),y) for x,y in zip(percentile_range_1,percentile_range_2)]
percentile_range


image_data = []
counter = 1


for data in [('test',ts_patient_ids)]:

    half_img_path = []
    flag = -1

    for i in data[1]:

        lung_list,num_t_pixels_list,tissue_by_total_list,tissue_by_lung_list = [], [], [], []

        for perc in percentile_range:

            img_no = get_img(perc[0], i, data[0])

            if img_no == flag:
                print(perc[0], i, data[0])
                continue

            flag = img_no

            try:
                img= load_image(path+'{0}/{1}/{2}.dcm'.format(data[0],i,img_no))
            except:
                print('{0}/{1}'.format(i,img_no), 'Err')
                continue

            try:
                lung_pixels, area_ratio, num_t_pixels, tissue_by_total, tissue_by_lung = make_lungmask(img, display=False)

                if math.isnan(lung_pixels):
                    print('nan lung_pixels',data[0],i,img_no)
                else:
                    lung_list.append((perc[0],lung_pixels))

                if math.isnan(num_t_pixels):
                    print('nan num_t_pixels',data[0],i,img_no)
                else:
                    num_t_pixels_list.append(num_t_pixels)

                if math.isnan(tissue_by_total):
                    print('nan tissue_by_total',data[0],i,img_no)
                else:
                    tissue_by_total_list.append(tissue_by_total)

                if math.isnan(tissue_by_lung):
                    print('nan tissue_by_lung',data[0],i,img_no)
                else:
                    tissue_by_lung_list.append(tissue_by_lung)

#                 lung_list.append((perc[0],lung_pixels))
#                 num_t_pixels_list.append(num_t_pixels)
#                 tissue_by_total_list.append(tissue_by_total)
#                 tissue_by_lung_list.append(tissue_by_lung)
            except Exception as e:
                print(data[0], i, img_no, e)
                pass

        slice_thickness = get_dicom_meta(path+'{0}/{1}/{2}.dcm'.format(data[0], i, img_no))['SliceThickness']
        pixel_spacing = get_dicom_meta(path+'{0}/{1}/{2}.dcm'.format(data[0], i, img_no))['PixelSpacing']

        try:
            Avg_NumTissuePixel_30_60 = round(sum(num_t_pixels_list)/len(num_t_pixels_list),4)
            Avg_Tissue_30_60 = round((sum(num_t_pixels_list)/len(num_t_pixels_list))*pixel_spacing,4)
            Avg_Tissue_thickness_30_60 = round((sum(num_t_pixels_list)/len(num_t_pixels_list))*pixel_spacing*slice_thickness,4)
            Avg_TissueByTotal_30_60 = round(sum(tissue_by_total_list)/len(tissue_by_total_list),4)
            Avg_TissueByLung_30_60 = round(sum(tissue_by_lung_list)/len(tissue_by_lung_list),4)
        except Exception as e:
            print(data[0], i, img_no, e)
            Avg_NumTissuePixel_30_60 = 0
            Avg_Tissue_30_60 = 0
            Avg_Tissue_thickness_30_60 = 0
            Avg_TissueByTotal_30_60 = 0
            Avg_TissueByLung_30_60 = 0

        num_im = num_img_bw_perc(percentile_range[0][0], percentile_range[1][0], i, data[0])
        if num_im == 0:
            num_im = 1

        approx_vol = 0
        for x in lung_list[:-1]:
            approx_vol+=(x[1]*pixel_spacing*slice_thickness*num_im)

        patient_dict = {'Patient':i,
                        'Data':data[0],
                        'SliceThickness': slice_thickness,
                        'PixelSpacing': pixel_spacing,
                        'NumImgBw5Prec': num_im,
                        'ApproxVol_30_60':round(approx_vol,4),
                        'Avg_NumTissuePixel_30_60':Avg_NumTissuePixel_30_60,
                        'Avg_Tissue_30_60':Avg_Tissue_30_60,
                        'Avg_Tissue_thickness_30_60':Avg_Tissue_thickness_30_60,
                        'Avg_TissueByTotal_30_60':Avg_TissueByTotal_30_60,
                        'Avg_TissueByLung_30_60':Avg_TissueByLung_30_60}

        image_data.append(patient_dict)

        if counter%20 == 0:
            print(counter)
        counter+=1
        
image_data_df = pd.DataFrame(image_data)
image_data_df.head()

# Merge test data with image features
ts = ts.merge(image_data_df, left_on = ['Patient'], right_on = ['Patient'], how = 'left')

# Merge train and test data
df = pd.concat([tr,ts])
f, axes = plt.subplots(3, 2, figsize=(16, 12))

sns.kdeplot(df.ApproxVol_30_60, shade = True, ax=axes[0, 0])
axes[0,0].set_title('Average approximate lung volume')
sns.kdeplot(df.Avg_NumTissuePixel_30_60, shade = True, ax=axes[0, 1])
axes[0,1].set_title('Average number of tissue pixels')
sns.kdeplot(df.Avg_Tissue_30_60, shade = True, ax=axes[1, 0])
axes[1,0].set_title('Average tissue pixel area')
sns.kdeplot(df.Avg_Tissue_thickness_30_60, shade = True, ax=axes[1, 1])
axes[1,1].set_title('Average tissue pixel volume')
sns.kdeplot(df.Avg_TissueByTotal_30_60, shade = True, ax=axes[2, 0])
axes[2,0].set_title('Tissue area by Total area ratio')
sns.kdeplot(df.Avg_TissueByLung_30_60, shade = True, ax=axes[2, 1])
axes[2,1].set_title('Tissue area by Lung area ratio')


# Impute 0 values with median
variables = ['ApproxVol_30_60','Avg_NumTissuePixel_30_60','Avg_Tissue_30_60','Avg_Tissue_thickness_30_60','Avg_TissueByTotal_30_60','Avg_TissueByLung_30_60']
for var in variables:
    median = df[var].quantile(q = 0.5, interpolation='linear')
    df.loc[df[var]==0,var] = median

# Impute extreme outliers in Avg_TissueByLung_20_60 by median
median = df['Avg_TissueByLung_30_60'].quantile(q = 0.5, interpolation='linear')
df.loc[df['Avg_TissueByLung_30_60']>=0.3,'Avg_TissueByLung_30_60'] = median

# Impute extreme outliers in Avg_TissueByLung_20_60 by median
median = df['Avg_Tissue_thickness_30_60'].quantile(q = 0.5, interpolation='linear')
df.loc[df['Avg_Tissue_thickness_30_60']>=20000,'Avg_Tissue_thickness_30_60'] = median

#Conver Avg_Tissue_30_60 to quartiles
df["Avg_Tissue_30_60_Quartile"] = pd.qcut(df.Avg_Tissue_30_60, q = 4, labels = ['Q1','Q2','Q3','Q4'])

f, axes = plt.subplots(3, 2, figsize=(16, 12))

sns.kdeplot(df.ApproxVol_30_60, shade = True, ax=axes[0, 0])
axes[0,0].set_title('Average approximate lung volume')
sns.kdeplot(df.Avg_NumTissuePixel_30_60, shade = True, ax=axes[0, 1])
axes[0,1].set_title('Average number of tissue pixels')
sns.kdeplot(df.Avg_Tissue_30_60, shade = True, ax=axes[1, 0])
axes[1,0].set_title('Average tissue pixel area')
sns.kdeplot(df.Avg_Tissue_thickness_30_60, shade = True, ax=axes[1, 1])
axes[1,1].set_title('Average tissue pixel volume')
sns.kdeplot(df.Avg_TissueByTotal_30_60, shade = True, ax=axes[2, 0])
axes[2,0].set_title('Tissue area by Total area ratio')
sns.kdeplot(df.Avg_TissueByLung_30_60, shade = True, ax=axes[2, 1])
axes[2,1].set_title('Tissue area by Lung area ratio')

l = ['SliceThickness','PixelSpacing','ApproxVol_30_60','Avg_NumTissuePixel_30_60','Avg_Tissue_30_60',
     'Avg_Tissue_thickness_30_60','Avg_TissueByTotal_30_60','Avg_TissueByLung_30_60']

for var in l:
    df[var] = (df[var] - df[var].min() ) / ( df[var].max() - df[var].min() )
    

f, axes = plt.subplots(3, 2, figsize=(16, 12))

sns.kdeplot(df.ApproxVol_30_60, shade = True, ax=axes[0, 0])
axes[0,0].set_title('Average approximate lung volume')
sns.kdeplot(df.Avg_NumTissuePixel_30_60, shade = True, ax=axes[0, 1])
axes[0,1].set_title('Average number of tissue pixels')
sns.kdeplot(df.Avg_Tissue_30_60, shade = True, ax=axes[1, 0])
axes[1,0].set_title('Average tissue pixel area')
sns.kdeplot(df.Avg_Tissue_thickness_30_60, shade = True, ax=axes[1, 1])
axes[1,1].set_title('Average tissue pixel volume')
sns.kdeplot(df.Avg_TissueByTotal_30_60, shade = True, ax=axes[2, 0])
axes[2,0].set_title('Tissue area by Total area ratio')
sns.kdeplot(df.Avg_TissueByLung_30_60, shade = True, ax=axes[2, 1])
axes[2,1].set_title('Tissue area by Lung area ratio')

train = df[df.Where == 'train']
test = df[df.Where == 'test']
print(train.shape)
print(test.shape)

quartile = ['Q1','Q2','Q3','Q4']
for q in quartile:
    temp_df = train.loc[train.Avg_Tissue_30_60_Quartile==q,:]
    patients = temp_df.Patient.unique()
    fvc_diff_lst = []
    print('Quartile: ', q)
    for patient in patients:
        fvc_lst = temp_df.loc[temp_df.Patient == patient, 'FVC'].tolist()
        fvc_diff = fvc_lst[-1]-fvc_lst[0]
        fvc_diff_lst.append(fvc_diff)
    print('FVC Diff Mean: ', sum(fvc_diff_lst)/len(fvc_diff_lst))
    print('FVC Diff Q1: ', np.quantile(fvc_diff_lst, .25))
    print('FVC Diff Q2/Median: ', np.quantile(fvc_diff_lst, .5))
    print('FVC Diff Q3: ', np.quantile(fvc_diff_lst, .75))
    print('FVC Diff Q4: ', np.quantile(fvc_diff_lst, .90))
    
    
    
#Modeling

A = {} 
TAB = {} 
P = [] 
for i, p in tqdm(enumerate(train.Patient.unique()), total=train.Patient.nunique()):
    sub = train.loc[train.Patient == p, :] 
    fvc = sub.FVC.values
    weeks = sub.Weeks.values
    c = np.vstack([weeks, np.ones(len(weeks))]).T
    a, b = np.linalg.lstsq(c, fvc)[0]
    
    A[p] = a
    TAB[p] = get_tab(sub)
    P.append(p)
    

from tensorflow.keras.utils import Sequence

class IGenerator(Sequence):
    BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']
    def __init__(self, keys, a, tab, batch_size=32):
        self.keys = [k for k in keys if k not in self.BAD_ID]
        self.a = a
        self.tab = tab
        self.batch_size = batch_size
        
        self.train_data = {}
        for p in train.Patient.values:
            self.train_data[p] = os.listdir(f'C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/osic-pulmonary-fibrosis-progression/train/{p}/')
    
    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        x = []
        a, tab = [], [] 
        keys = np.random.choice(self.keys, size = self.batch_size)
        for k in keys:
            try:
                i = np.random.choice(self.train_data[k], size=1)[0]
                img = get_img_1(f'C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/osic-pulmonary-fibrosis-progression/train/{k}/{i}')
                x.append(img)
                a.append(self.a[k])
                tab.append(self.tab[k])
            except:
                print(k, i)
       
        x,a,tab = np.array(x), np.array(a), np.array(tab)
        x = np.expand_dims(x, axis=-1)
        return [x, tab] , a
    
    
from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, GlobalAveragePooling2D, Add, Conv2D, AveragePooling2D, 
    LeakyReLU, Concatenate 
)
import efficientnet.tfkeras as efn

def get_efficientnet(model, shape):
    models_dict = {
        'b0': efn.EfficientNetB0(input_shape=shape,weights=None,include_top=False),
        'b1': efn.EfficientNetB1(input_shape=shape,weights=None,include_top=False),
        'b2': efn.EfficientNetB2(input_shape=shape,weights=None,include_top=False),
        'b3': efn.EfficientNetB3(input_shape=shape,weights=None,include_top=False),
        'b4': efn.EfficientNetB4(input_shape=shape,weights=None,include_top=False),
        'b5': efn.EfficientNetB5(input_shape=shape,weights=None,include_top=False),
        'b6': efn.EfficientNetB6(input_shape=shape,weights=None,include_top=False),
        'b7': efn.EfficientNetB7(input_shape=shape,weights=None,include_top=False)
    }
    return models_dict[model]

def build_model(shape=(512, 512, 1), model_class = None, fold=None):
    inp = Input(shape=shape)
    base = get_efficientnet(model_class, shape)
    x = base(inp)
    x = GlobalAveragePooling2D()(x)
    inp2 = Input(shape=(6,))
    x2 = tf.keras.layers.GaussianNoise(0.2)(inp2)
    x = Concatenate()([x, x2]) 
    x = Dropout(0.5)(x) 
    x = Dense(1)(x)
    model = Model([inp, inp2] , x)
    

    return model


bst_quantile = [0.8, 0.5, 0.1, 0.1, 0.1]

model_class = 'b1'
NFOLD = 5

kf = KFold(n_splits=NFOLD, random_state=42,shuffle=True)
P = np.array(P)
subs = []
folds_history = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(P)):
    print('#####################')
    print('####### Fold %i ######'%fold)
    print('#####################')
    print('Predicting...')
    
    model = build_model(shape=(512, 512, 1), model_class = model_class, fold = fold)
    q = bst_quantile[fold]
    
    sub = pd.read_csv('C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/osic-pulmonary-fibrosis-progression/sample_submission.csv') 
    A_test, B_test, P_test,W, FVC= {}, {}, {}, {}, {} 
    STD, WEEK = {}, {} 
    for p in test.Patient.unique():
        x = [] 
        tab = [] 
        ldir = os.listdir(f'C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/osic-pulmonary-fibrosis-progression/test/{p}/')
        counter = 0
        for i in ldir:
            if int(i[:-4]) / len(ldir) < 0.7 and int(i[:-4]) / len(ldir) > 0.20:
                x.append(get_img_1(f'C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/osic-pulmonary-fibrosis-progression/test/{p}/{i}')) 
                tab.append(get_tab(test.loc[test.Patient == p, :])) 
                counter+=1
            if counter == 20:
                break
        if len(x) <= 1:
            continue
        tab = np.array(tab) 

        x = np.expand_dims(x, axis=-1) 
        _a = model.predict([x, tab]) 
        a = np.quantile(_a, q)
        A_test[p] = a
        B_test[p] = test.FVC.values[test.Patient == p] - a*test.Weeks.values[test.Patient == p]
        P_test[p] = test.Percent.values[test.Patient == p] 
        WEEK[p] = test.Weeks.values[test.Patient == p]

    for k in sub.Patient_Week.values:
        p, w = k.split('_')
        w = int(w) 

        fvc = A_test[p] * w + B_test[p]
        sub.loc[sub.Patient_Week == k, 'FVC'] = fvc
        sub.loc[sub.Patient_Week == k, 'Confidence'] = (
            P_test[p] - A_test[p] * abs(WEEK[p] - w) 
    ) 

    _sub = sub[["Patient_Week","FVC","Confidence"]].copy()
    subs.append(_sub)
    
    K.clear_session()
    del model
    gc.collect()
    
    
N = len(subs)
sub = subs[0].copy() # ref
sub["FVC"] = 0
sub["Confidence"] = 0
for i in range(N):
    sub["FVC"] += subs[i]["FVC"] * (1/N)
    sub["Confidence"] += subs[i]["Confidence"] * (1/N)
    
    
    
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
import warnings
warnings.filterwarnings("ignore")

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

# Model imports
import tensorflow as tf
from tensorflow.keras.layers import (
                                    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, GlobalAveragePooling2D,
                                    Add, Conv2D, AveragePooling2D, LeakyReLU, Concatenate , Lambda
                                    )
from tensorflow.keras import Model
import tensorflow.keras.models as M
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K
import tensorflow.keras.applications as tfa
import efficientnet.tfkeras as efn
from sklearn.model_selection import train_test_split, KFold
import seaborn as sns


pd.set_option("display.max_columns", 100)
custom_colors = ['#74a09e','#86c1b2','#98e2c6','#f3c969','#f2a553', '#d96548', '#c14953']
sns.palplot(sns.color_palette(custom_colors))

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

EPOCHS = 30
BATCH_SIZE = 8
NFOLD = 5
LR = 0.003
SAVE_BEST = True
MODEL_CLASS = 'b1'
path = 'C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/osic-pulmonary-fibrosis-progression/'

#------------------------------------------
tr = pd.read_csv(f"{path}/train.csv")
tr.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])
chunk = pd.read_csv(f"{path}/test.csv")

# Fetch unique patient ids
tr_patient_ids = tr.Patient.unique().tolist()
ts_patient_ids = chunk.Patient.unique().tolist()

sub = pd.read_csv(f"{path}/sample_submission.csv")
sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])
sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]
sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient")

tr['WHERE'] = 'train'
chunk['WHERE'] = 'val'
sub['WHERE'] = 'test'
meta_data = pd.concat([tr, chunk, sub], ignore_index=True)

print(tr.shape, chunk.shape, sub.shape, meta_data.shape)
print(tr.Patient.nunique(), chunk.Patient.nunique(), sub.Patient.nunique(),meta_data.Patient.nunique())

#ADDS TO THE PATIENT A VALUE train/test confidence if it has it and patient_week

meta_data['min_week'] = meta_data['Weeks']
meta_data.loc[meta_data.WHERE=='test','min_week'] = np.nan
meta_data['min_week'] = meta_data.groupby('Patient')['min_week'].transform('min')

base = meta_data.loc[meta_data.Weeks == meta_data.min_week]
base = base[['Patient','FVC']].copy()
base.columns = ['Patient','min_FVC']
base['nb'] = 1
base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')
base = base[base.nb==1]
base.drop('nb', axis=1, inplace=True)
meta_data = meta_data.merge(base, on='Patient', how='left')

base = meta_data.loc[meta_data.Weeks == meta_data.min_week]
base = base[['Patient','Percent']].copy()
base.columns = ['Patient','min_Percent']
base['nb'] = 1
base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')
base = base[base.nb==1]
base.drop('nb', axis=1, inplace=True)
meta_data = meta_data.merge(base, on='Patient', how='left')

meta_data['base_week'] = meta_data['Weeks'] - meta_data['min_week']
del base

COLS = ['Sex','SmokingStatus'] #,'Age'
FE = []
for col in COLS:
    for mod in meta_data[col].unique():
        print(col, mod)
        FE.append(mod)
        meta_data[mod] = (meta_data[col] == mod).astype(int)
#=================

print(meta_data.shape)


image_data = []
counter = 1

percentile_range_1 = np.linspace(0.3,0.6,11)
percentile_range_2 = range(30,63, 3)
percentile_range = [(round(x,2),y) for x,y in zip(percentile_range_1,percentile_range_2)]


for data in [('test',ts_patient_ids)]:

    half_img_path = []
    flag = -1

    for i in data[1]:

        lung_list,num_t_pixels_list,tissue_by_total_list,tissue_by_lung_list = [], [], [], []

        for perc in percentile_range:

            img_no = get_img(perc[0], i, data[0])

            if img_no == flag:
                print(perc[0], i, data[0])
                continue

            flag = img_no

            try:
                img= load_image(path+'{0}/{1}/{2}.dcm'.format(data[0],i,img_no))
            except:
                print('{0}/{1}'.format(i,img_no), 'Err')
                continue

            try:
                lung_pixels, area_ratio, num_t_pixels, tissue_by_total, tissue_by_lung = make_lungmask(img, display=False)

                if math.isnan(lung_pixels):
                    print('nan lung_pixels',data[0],i,img_no)
                else:
                    lung_list.append((perc[0],lung_pixels))

                if math.isnan(num_t_pixels):
                    print('nan num_t_pixels',data[0],i,img_no)
                else:
                    num_t_pixels_list.append(num_t_pixels)

                if math.isnan(tissue_by_total):
                    print('nan tissue_by_total',data[0],i,img_no)
                else:
                    tissue_by_total_list.append(tissue_by_total)

                if math.isnan(tissue_by_lung):
                    print('nan tissue_by_lung',data[0],i,img_no)
                else:
                    tissue_by_lung_list.append(tissue_by_lung)

#                 lung_list.append((perc[0],lung_pixels))
#                 num_t_pixels_list.append(num_t_pixels)
#                 tissue_by_total_list.append(tissue_by_total)
#                 tissue_by_lung_list.append(tissue_by_lung)
            except Exception as e:
                print(data[0], i, img_no, e)
                pass

        slice_thickness = get_dicom_meta(path+'{0}/{1}/{2}.dcm'.format(data[0], i, img_no))['SliceThickness']
        pixel_spacing = get_dicom_meta(path+'{0}/{1}/{2}.dcm'.format(data[0], i, img_no))['PixelSpacing']

        try:
            Avg_NumTissuePixel_30_60 = round(sum(num_t_pixels_list)/len(num_t_pixels_list),4)
            Avg_Tissue_30_60 = round((sum(num_t_pixels_list)/len(num_t_pixels_list))*pixel_spacing,4)
            Avg_Tissue_thickness_30_60 = round((sum(num_t_pixels_list)/len(num_t_pixels_list))*pixel_spacing*slice_thickness,4)
            Avg_TissueByTotal_30_60 = round(sum(tissue_by_total_list)/len(tissue_by_total_list),4)
            Avg_TissueByLung_30_60 = round(sum(tissue_by_lung_list)/len(tissue_by_lung_list),4)
        except Exception as e:
            print(data[0], i, img_no, e)
            Avg_NumTissuePixel_30_60 = 0
            Avg_Tissue_30_60 = 0
            Avg_Tissue_thickness_30_60 = 0
            Avg_TissueByTotal_30_60 = 0
            Avg_TissueByLung_30_60 = 0

        num_im = num_img_bw_perc(percentile_range[0][0], percentile_range[1][0], i, data[0])
        if num_im == 0:
            num_im = 1

        approx_vol = 0
        for x in lung_list[:-1]:
            approx_vol+=(x[1]*pixel_spacing*slice_thickness*num_im)

        patient_dict = {'Patient':i,
                        'Data':data[0],
                        'SliceThickness': slice_thickness,
                        'PixelSpacing': pixel_spacing,
                        'NumImgBw5Prec': num_im,
                        'ApproxVol_30_60':round(approx_vol,4),
                        'Avg_NumTissuePixel_30_60':Avg_NumTissuePixel_30_60,
                        'Avg_Tissue_30_60':Avg_Tissue_30_60,
                        'Avg_Tissue_thickness_30_60':Avg_Tissue_thickness_30_60,
                        'Avg_TissueByTotal_30_60':Avg_TissueByTotal_30_60,
                        'Avg_TissueByLung_30_60':Avg_TissueByLung_30_60}

        image_data.append(patient_dict)

        if counter%20 == 0:
            print(counter)
        counter+=1
        

image_data_df = pd.DataFrame(image_data)

tr1 = pd.read_csv('C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/ProgressionCode/train_df_with_img_feat.csv')
cols = image_data_df.columns
tr1 = tr1[cols]
tr1 = tr1.drop_duplicates(keep = 'first')


# Merge train and test data
image_data_df = pd.concat([tr1,image_data_df])

# Impute 0 values with median
variables = ['ApproxVol_30_60','Avg_NumTissuePixel_30_60','Avg_Tissue_30_60','Avg_Tissue_thickness_30_60','Avg_TissueByTotal_30_60','Avg_TissueByLung_30_60']
for var in variables:
    median = image_data_df[var].quantile(q = 0.5, interpolation='linear')
    image_data_df.loc[image_data_df[var]==0,var] = median

# Impute extreme outliers in Avg_TissueByLung_20_60 by median
median = image_data_df['Avg_TissueByLung_30_60'].quantile(q = 0.5, interpolation='linear')
image_data_df.loc[image_data_df['Avg_TissueByLung_30_60']>=0.3,'Avg_TissueByLung_30_60'] = median

# Impute extreme outliers in Avg_TissueByLung_20_60 by median
median = image_data_df['Avg_Tissue_thickness_30_60'].quantile(q = 0.5, interpolation='linear')
image_data_df.loc[image_data_df['Avg_Tissue_thickness_30_60']>=20000,'Avg_Tissue_thickness_30_60'] = median

final_data = meta_data.merge(image_data_df, left_on = ['Patient','WHERE'], right_on = ['Patient', 'Data'], how = 'left')
print(final_data.shape)
# final_data = final_data.loc[final_data.Patient != 'ID00011637202177653955184',:]
print(final_data.shape)


# Min-Max normalization

final_data['age'] = (final_data['Age'] - final_data['Age'].min() ) / ( final_data['Age'].max() - final_data['Age'].min() )
final_data['BASE_FVC'] = (final_data['min_FVC'] - final_data['min_FVC'].min() ) / ( final_data['min_FVC'].max() - final_data['min_FVC'].min() )
final_data['week'] = (final_data['base_week'] - final_data['base_week'].min() ) / ( final_data['base_week'].max() - final_data['base_week'].min() )
final_data['BASE_percent'] = (final_data['min_Percent'] - final_data['min_Percent'].min() ) / ( final_data['min_Percent'].max() - final_data['min_Percent'].min() )


l = ['SliceThickness','PixelSpacing','ApproxVol_30_60','Avg_NumTissuePixel_30_60','Avg_Tissue_30_60',
     'Avg_Tissue_thickness_30_60','Avg_TissueByTotal_30_60','Avg_TissueByLung_30_60']

for var in l:
    final_data[var] = (final_data[var] - final_data[var].min() ) / ( final_data[var].max() - final_data[var].min() )

FE += ['age','week','BASE_FVC','BASE_percent']
FE.extend(l)

tr = final_data.loc[final_data.WHERE=='train']
chunk = final_data.loc[final_data.WHERE=='val']
sub = final_data.loc[final_data.WHERE=='test']

C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")
#=============================#
def score(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    
    #sigma_clip = sigma + C1
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
    return K.mean(metric)
#============================#
def qloss(y_true, y_pred):
    # Pinball loss for multiple quantiles
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return K.mean(v)
#=============================#
def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)
    return loss
#=================
def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence, return_values = False):
    """
    Calculates the modified Laplace Log Likelihood score for this competition.
    """
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    metric = - np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)

    if return_values:
        return metric
    else:
        return np.mean(metric)
#=================
def make_model(FE):
    z = Input((len(FE),), name="Patient")
    x = Dense(100, activation="relu", name="d1")(z)
#     x = L.Dropout(0.05)(x)
    x = Dense(100, activation="relu", name="d2")(x)
#     x = L.Dense(100, activation="relu", name="d3")(x)
    p1 = Dense(3, activation="linear", name="p1")(x)
    p2 = Dense(3, activation="relu", name="p2")(x)
    preds = Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), 
                     name="preds")([p1, p2])
    
    model = M.Model(z, preds, name="CNN")
    #model.compile(loss=qloss, optimizer="adam", metrics=[score])
    model.compile(loss=mloss(0.65), optimizer=tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False), metrics=[score])
    return model

net = make_model(FE)
print(net.summary())
print(net.count_params())

## GET TRAINING DATA AND TARGET VALUE

# get target value
y = tr['FVC'].values.astype(float)


# get training & test data
X_train = tr[FE].values
X_test = sub[FE].values

# instantiate target arrays
train_preds = np.zeros((X_train.shape[0], 3))
test_preds = np.zeros((X_test.shape[0], 3))


FE = ['Male', 'Female', 'Ex-smoker', 'Never smoked', 'Currently smokes', 'age', 'week', 'BASE_FVC', 'BASE_percent']
image_features = ['SliceThickness','PixelSpacing','ApproxVol_30_60','Avg_NumTissuePixel_30_60','Avg_Tissue_30_60',
                  'Avg_TissueByTotal_30_60','Avg_TissueByLung_30_60']
FE1 = FE
FE2 = FE+['ApproxVol_30_60']
FE3 = FE+['Avg_Tissue_thickness_30_60']
FE4 = FE+['Avg_TissueByLung_30_60']
FE5 = FE+['ApproxVol_30_60','Avg_Tissue_thickness_30_60','Avg_TissueByLung_30_60']

model_cnt = 1
for FE in [FE1, FE2, FE3, FE4, FE5]:
    
    print('Model Count: ', model_cnt, '\nFeatures Used: ', FE)
    
    # get target value
    y = tr['FVC'].values.astype(float)

    # get training & test data
    X_train = tr[FE].values
    X_test = sub[FE].values

    # instantiate target arrays
    globals()['train_preds_{}'.format(model_cnt)] = np.zeros((X_train.shape[0], 3))
    globals()['test_preds_{}'.format(model_cnt)] = np.zeros((X_test.shape[0], 3))

    NFOLD = 5
    gkf = GroupKFold(n_splits=NFOLD)
    groups = tr['Patient'].values

    cnt = 0
    EPOCHS = 800
    BATCH_SIZE=128
    for tr_idx, val_idx in gkf.split(X_train, y, groups):
        cnt += 1
        print(f"FOLD {cnt}")
        net = make_model(FE)
        net.fit(X_train[tr_idx], y[tr_idx], batch_size=BATCH_SIZE, epochs=EPOCHS, 
                validation_data=(X_train[val_idx], y[val_idx]), verbose=0) #
        print("train", net.evaluate(X_train[tr_idx], y[tr_idx], verbose=0, batch_size=BATCH_SIZE))
        print("val", net.evaluate(X_train[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE))
        print("predict val...")
        globals()['train_preds_{}'.format(model_cnt)][val_idx] = net.predict(X_train[val_idx], batch_size=BATCH_SIZE, verbose=0)
        print("predict test...")
        globals()['test_preds_{}'.format(model_cnt)] += net.predict(X_test, batch_size=BATCH_SIZE, verbose=0) / NFOLD

    predicted_fvc = globals()['train_preds_{}'.format(model_cnt)][:,1]
    confidence = globals()['train_preds_{}'.format(model_cnt)][:,2]-globals()['train_preds_{}'.format(model_cnt)][:,0]
    model_score = laplace_log_likelihood(actual_fvc = y, predicted_fvc = predicted_fvc, confidence = confidence,
                           return_values = False)
    print('Overall Score: ', model_score)
    model_cnt+=1
    #==============
    
    scores = []

for i in range(1,6):
    globals()['predicted_fvc_{}'.format(i)] = globals()['train_preds_{}'.format(i)][:,1]
    globals()['confidence_{}'.format(i)] = globals()['train_preds_{}'.format(i)][:,2]-globals()['train_preds_{}'.format(i)][:,0]
    
for i1 in np.linspace(0,0.5,11):
    for i2 in np.linspace(0,0.5,11):
        for i3 in np.linspace(0,0.5,11):
            for i4 in np.linspace(0,0.5,11):
                for i5 in np.linspace(0,0.5,11):
                    if i1+i2+i3+i4+i5 == 1:
                        train_preds = (i1*train_preds_1+
                                       i2*train_preds_2+
                                       i3*train_preds_3+
                                       i4*train_preds_4+
                                       i5*train_preds_5)
                        
                        predicted_fvc = train_preds[:,1]
                        confidence = train_preds[:,2]-train_preds[:,0]
                        score = laplace_log_likelihood(actual_fvc = y, predicted_fvc = predicted_fvc, confidence = confidence, return_values = False)
                        scores.append((i1,i2,i3,i4,i5,score))
            
scores = sorted(scores, key = lambda x: x[-1])
scores[-10:]

best_weights = scores[-1]
train_preds = (best_weights[0]*train_preds_1+best_weights[1]*train_preds_2+best_weights[2]*train_preds_3+
               best_weights[3]*train_preds_4+best_weights[4]*train_preds_5)

test_preds = (best_weights[0]*test_preds_1+best_weights[1]*test_preds_2+best_weights[2]*test_preds_3+
               best_weights[3]*test_preds_4+best_weights[4]*test_preds_5)

## FIND OPTIMIZED STANDARD-DEVIATION
sigma_opt = mean_absolute_error(y, train_preds[:,1])
sigma_uncertain = train_preds[:,2] - train_preds[:,0]
sigma_mean = np.mean(sigma_uncertain)
print(sigma_opt, sigma_mean)

# 161.8837622626985 237.76367962865177

idxs = np.random.randint(0, y.shape[0], 100)
plt.plot(y[idxs], label="ground truth")
plt.plot(train_preds[idxs, 0], label="q25")
plt.plot(train_preds[idxs, 1], label="q50")
plt.plot(train_preds[idxs, 2], label="q75")
plt.legend(loc="best")
plt.show()

## PREPARE SUBMISSION FILE WITH OUR PREDICTIONS
sub['FVC1'] = test_preds[:, 1]
sub['Confidence1'] = test_preds[:,2] - test_preds[:,0]

# get rid of unused data and show some non-empty data
submission = sub[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()
submission.loc[~submission.FVC1.isnull()].head(10)

submission.loc[~submission.FVC1.isnull(),'FVC'] = submission.loc[~submission.FVC1.isnull(),'FVC1']

if sigma_mean < 70:
    submission['Confidence'] = sigma_opt
else:
    submission.loc[~submission.FVC1.isnull(),'Confidence'] = submission.loc[~submission.FVC1.isnull(),'Confidence1']
    
    org_test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

for i in range(len(org_test)):
    submission.loc[submission['Patient_Week']==org_test.Patient[i]+'_'+str(org_test.Weeks[i]), 'FVC'] = org_test.FVC[i]
    submission.loc[submission['Patient_Week']==org_test.Patient[i]+'_'+str(org_test.Weeks[i]), 'Confidence'] = 70
    
submission[["Patient_Week","FVC","Confidence"]].to_csv("submission_meta.csv", index = False)

reg_sub = submission[["Patient_Week","FVC","Confidence"]].copy()

df1 = img_sub.sort_values(by=['Patient_Week'], ascending=True).reset_index(drop=True)
df2 = reg_sub.sort_values(by=['Patient_Week'], ascending=True).reset_index(drop=True)

df = df1[['Patient_Week']].copy()
df['FVC'] = 0.4*df1['FVC'] + 0.6*df2['FVC']
df['Confidence'] = 0.4*df1['Confidence'] + 0.6*df2['Confidence']

df.to_csv('submission.csv', index=False)