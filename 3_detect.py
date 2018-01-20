#-*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import os
from src.net import text_activation_map

WEIGHT_FILE = "weights//weights.07-0.05.h5"


if __name__ == "__main__":

    img_files = os.listdir("images")
    img_files = [os.path.join("images", fname) for fname in img_files]
    
    for fname in img_files:
        text_map, im = text_activation_map(WEIGHT_FILE, fname)
    
        fig, ax = plt.subplots()
        ax.imshow(im, alpha=0.5)
        ax.imshow(text_map, cmap='jet', alpha=0.5)
        
        weight_name = os.path.split(WEIGHT_FILE)[1]
        dname = os.path.join("images", "detected_{}".format(weight_name))
        
        if not os.path.isdir(dname):
            os.mkdir(dname)
    
        write_path = os.path.join(dname, os.path.split(fname)[1])
        plt.savefig(write_path, bbox_inches='tight')
        print(write_path)
