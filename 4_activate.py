#-*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.utils import get_list_images
from src.activate import ClsActWorker

DATASET_TEXT = "dataset//train//text"

        
if __name__ == "__main__":
    
    # 1. create worker
    worker = ClsActWorker(cls_weights=np.load("cls_weights.npy"))

    # 2. get images
    images = get_list_images(DATASET_TEXT, 10, random_order=False)
    
    # 3. 
    maps = worker.run(images)
    print(maps.shape)
 
    for img, conv_map in zip(images, maps):
        conv_map = cv2.resize(conv_map, (img.shape[1], img.shape[0]))
        fig, ax = plt.subplots(nrows=1, ncols=2)
        plt.subplot(1, 2, 1)
        plt.imshow(img, alpha=0.6)
        plt.imshow(conv_map, cmap='jet', alpha=0.4)
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.show()
        # plt.savefig("{}.png".format(i), bbox_inches='tight')



