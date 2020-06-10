import numpy as np
import cv2
import os
import glob
import re
from tqdm import tqdm
import scipy.ndimage
import matplotlib.pyplot as plt

from argparse import ArgumentParser

""" python zoom_resize.py --data_path '../data/' --size 128 --out_path '../output/' """

def parse_args():
    """ Take arguments from user inputs."""
    parser = ArgumentParser(description='Type of Dataset must be .npy')
    parser.add_argument('--data_path', help='Directory path for data', 
            default='../data/',  type=str)
    # parser.add_argument('--hu', dset='hu_cut') # You can customizing this part
    parser.add_argument('--size', help='Output zoom image size',
            default=128, type=int)
    parser.add_argument('--out_path', help='Output path of preprocessed data',
            default='../output/', type=str)

    args = parser.parse_args()
    return args

# Axial image
def ax_resize(img_set, size):
    ax = img_set[:, :, :32]
    # Default zoom image size is 128
    new_img = np.zeros((size, size, 32), dtype=np.float32)
    for i in range(32):
        try:
            # Convert to binary pixel
            _, thresh_ax = cv2.threshold(ax[:,:,i], 1, 255, cv2.THRESH_BINARY)
            
            y_ax, x_ax = np.nonzero(thresh_ax)
            # Cutting unnecessary part
            ax_cut = ax[np.min(y_ax):np.max(y_ax), np.min(x_ax):np.max(x_ax), i]
            # Set resize factor
            ax_resize_factor = np.array([size, size]) / ax_cut.shape
            # Zoom interpolation
            ax_resized_img = scipy.ndimage.interpolation.zoom(ax_cut, ax_resize_factor)

            new_img[:,:,i] = ax_resized_img
        except:
            new_img[:,:,i] = np.zeros((size, size))

    # Hu-cutting
    img_fat = new_img.copy()
    # Fat Extraction
    img_fat[img_fat < -110] = 0
    img_fat[img_fat > -50] = 0

    img_msc = new_img.copy()
    # Muscle Extraction
    img_msc[img_msc < 10] = 0
    img_msc[img_msc > 40] = 0

    # Merge
    new_img = img_fat + img_msc
    # MinMax Normalization
    new_img = (new_img + 110) / (40 + 110)
    return new_img

# Coronal image
def co_resize(img_set, size):
    co = img_set[:, :, 32:64]
    # Default zoom image size is 128
    new_img = np.zeros((size, size, 32), dtype=np.float32)
    for i in range(32):
        try:
            # Convert to binary pixel
            _, thresh_co = cv2.threshold(co[:,:,i], 1, 255, cv2.THRESH_BINARY)
            
            y_co, x_co = np.nonzero(thresh_co)
            # Cutting unnecessary part
            co_cut = co[np.min(y_co):np.max(y_co), np.min(x_co):np.max(x_co), i]
            # Set resize factor
            co_resize_factor = np.array([size/2, size]) / co_cut.shape
            # Zoom interpolation
            co_resized_img = scipy.ndimage.interpolation.zoom(co_cut, co_resize_factor)
            
            new_img[:64,:,i] = co_resized_img
        except:
            new_img[:,:,i] = np.zeros((size, size))

   # Hu-cutting
    img_fat = new_img.copy()
    # Fat Extraction
    img_fat[img_fat < -110] = 0
    img_fat[img_fat > -50] = 0

    img_msc = new_img.copy()
    # Muscle Extraction
    img_msc[img_msc < 10] = 0
    img_msc[img_msc > 40] = 0

    # Merge
    new_img = img_fat + img_msc
    # MinMax Normalization
    new_img = (new_img + 110) / (40 + 110)
    return new_img

# Sagittal image
def sa_resize(img_set, size):
    sa = img_set[:, :, 64:]
    # Default zoom image size is 128
    new_img = np.zeros((size, size, 32), dtype=np.float32)
    for i in range(32):
        try:
            # Convert to binary pixel
            _, thresh_sa = cv2.threshold(sa[:,:,i], 1, 255, cv2.THRESH_BINARY)
            
            y_sa, x_sa = np.nonzero(thresh_sa)
            # Cutting unnecessary part
            sa_cut = sa[np.min(y_sa):np.max(y_sa), np.min(x_sa):np.max(x_sa), i]
            # Set resize factor
            sa_resize_factor = np.array([size, size]) / sa_cut.shape
            # Zoom interpolation
            sa_resized_img = scipy.ndimage.interpolation.zoom(sa_cut, sa_resize_factor)
            
            new_img[:,:,i] = sa_resized_img
        except:
            new_img[:,:,i] = np.zeros((size, size))

   # Hu-cutting
    img_fat = new_img.copy()
    # Fat Extraction
    img_fat[img_fat < -110] = 0
    img_fat[img_fat > -50] = 0

    img_msc = new_img.copy()
    # Muscle Extraction
    img_msc[img_msc < 10] = 0
    img_msc[img_msc > 40] = 0

    # Merge
    new_img = img_fat + img_msc
    # MinMax Normalization
    new_img = (new_img + 110) / (40 + 110)
    return new_img

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    fn_list = glob.glob(args.data_path + '*.npy')
    conf = re.compile('[0-9]+')


    for fn in tqdm(fn_list):
        print(fn)
        pid = conf.findall(fn)[0]
        img = np.load(fn)
        
        ax_img = ax_resize(img, args.size)
        co_img = co_resize(img, args.size)
        sa_img = sa_resize(img, args.size)
        
        resized_img = np.concatenate([ax_img, co_img, sa_img], axis=2)
        np.save(args.out_path + pid + '.npy', resized_img) 

