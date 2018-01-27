#-*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.utils import list_files, files_to_images
from src.activate import ClsActWorker

DATASET_TEXT = "dataset//train//text"
N_SAMPLES = 100
DATASET_KAIST = "..//dataset//kaist"
        
if __name__ == "__main__":
    
    # 1. create worker
    worker = ClsActWorker(cls_weights=np.load("weights//kaist_text_weights.npy"))

    # 2. get images
    files = list_files(DATASET_KAIST, "*.jpg", N_SAMPLES, random_order=False)
    images = files_to_images(files)
    
    # 3. 
    maps = worker.run(images)
    print(maps.shape)
 
    for i, (img, conv_map) in enumerate(zip(images, maps)):
        conv_map = cv2.resize(conv_map, (img.shape[1], img.shape[0]))
        fig, ax = plt.subplots(nrows=1, ncols=2)
        plt.subplot(2, 1, 1)
        plt.imshow(img)
        plt.subplot(2, 1, 2)
        plt.imshow(img, alpha=0.7)
        plt.imshow(conv_map, cmap='jet', alpha=0.3)
        # plt.show()
        plt.savefig("{}.png".format(i), bbox_inches='tight')
        print("{}.png saved".format(i))


