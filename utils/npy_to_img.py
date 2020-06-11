import os
import numpy as np
import glob
import re
import matplotlib.pylab as plt

from argparse import ArgumentParser

""" python npy_to_img.py --data_path '../data/' --out_path '../output/' """

def parse_args():
    """ Take arguments from user inputs."""
    parser = ArgumentParser(description='Convert .npy data to png image file')
    parser.add_argument('--data_path', help='Directory path for npy data', 
            default='../data/',  type=str)
    parser.add_argument('--out_path', help='Output path of png image file',
            default='../output/', type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    data_path = args.data_path
    output_path = args.out_path

    conf = re.compile('[0-9]+')

    for fn in glob.glob(data_path + '*.npy'): 
        os.makedirs(os.path.join(output_path + 'png_image' + '/'+ fn.split('/')[-1][:-4] +'/'))
        print(fn)
        imgs = np.load(fn)

        for j in range(96):
                plt.imsave(output_path + 'png_image' + '/'+ fn.split('/')[-1][:-4] +'/{}.png'.format(j), imgs[:,:,j], cmap="gray")