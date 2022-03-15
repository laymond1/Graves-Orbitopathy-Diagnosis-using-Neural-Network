from __future__ import absolute_import, division, print_function, unicode_literals

import os
import glob
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import tensorflow as tf

from utils import load_img
from cnn_model import CNN_Model

from argparse import ArgumentParser

""" 0: Control, 1: Severe, 2: Mild """

""" python test_multiclass.py --data_path './sample_data/' --model 'axcosa' --load_path './saved_model/S_M_C/' """

def parse_args():
    """ Take arguments from user inputs."""
    parser = ArgumentParser(description='Multiclass Classification')
    parser.add_argument('--data_path', help='Directory path for preprocessed data', 
            default='./sample_data/',  type=str)
    parser.add_argument('--model', help='Model: ax, co, sa, axco, axsa, cosa, axcosa',
            default='axcosa', type=str)
    parser.add_argument('--load_path', help='Path of Saved Model Checkpoint',
            default='./saved_model/S_M_C/', type=str)

    args = parser.parse_args()
    return args

def main(data_path, mode, load_path):
    # Model
    model = CNN_Model(mode=mode, multiclass=True)
    model.load_weights(load_path + 'my_chekpoint').expect_partial()

    # Load Dataset
    fn_list = glob.glob(data_path + '*.npy')
    for fn in fn_list:
        pid = fn.split('/')[-1][:-4]
        img = np.load(fn)
        img = img.reshape((1, 128, 128, 96))

        # Predict
        pred = model(img)
        pred = np.argmax(pred, 1)
        if pred == 0:
            label = 'Control'
        elif pred ==1:
            label = 'Severe'
        else:
            label = 'Mild'

        # Result
        print("ID: {} Label: {}".format(pid, label))

if __name__ == "__main__":
    args = parse_args()

    data_path = args.data_path
    mode = args.model
    load_path = args.load_path

    main(data_path, mode, load_path)
