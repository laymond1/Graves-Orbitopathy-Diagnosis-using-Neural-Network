from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import glob
import random

def load_img(data_path, case):
    if case == "S_C":
        b_list = glob.glob(data_path + 'control/*.npy')
        p_list = glob.glob(data_path + 'severe/*.npy')        
    elif case == "M_C":
        b_list = glob.glob(data_path + 'control/*.npy')
        p_list = glob.glob(data_path + 'mild/*.npy')
    elif case == "S_M":
        b_list = glob.glob(data_path + 'mild/*.npy')
        p_list = glob.glob(data_path + 'severe/*.npy')
    else:
        b_list = glob.glob(data_path + 'control/*.npy')
        p_list = glob.glob(data_path + 'severe/*.npy')
        m_list = glob.glob(data_path + 'mild/*.npy')

        num_data = len(b_list) + len(p_list) + len(m_list)
        const_pixel_dims = (num_data, 128, 128, 96)
        img_set = np.zeros(const_pixel_dims, dtype=np.float32)
        label = np.zeros((num_data,3), dtype=np.float32)

        for i, fn in enumerate(b_list):
            img_set[i,:,:,:] = np.load(fn)
            label[i,0] = 1.0

        for i, fn in enumerate(p_list):
            img_set[i+len(b_list),:,:,:] = np.load(fn)
            label[i+len(b_list),1] = 1.0

        for i, fn in enumerate(m_list):
            img_set[i+len(b_list)+len(p_list),:,:,:] = np.load(fn)
            label[i+len(b_list)+len(p_list),2] = 1.0

        return img_set, label

    num_data = len(b_list) + len(p_list)
    const_pixel_dims = (num_data, 128, 128, 96)
    img_set = np.zeros(const_pixel_dims, dtype=np.float32)
    label = np.zeros((num_data,2), dtype=np.float32)

    for i, fn in enumerate(b_list):
        img_set[i,:,:,:] = np.load(fn)
        label[i,0] = 1.0

    for i, fn in enumerate(p_list):
        img_set[i+len(b_list),:,:,:] = np.load(fn)
        label[i+len(b_list),1] = 1.0

    return img_set, label
