#-*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.feature import FeatureExtractor
from src.utils import get_list_images, load_model
from src.activate import activate_label

DATASET_TEXT = "dataset//train//text"


class ClsActWorker(object):
    
    def __init__(self):
        self._fe = FeatureExtractor()
    
    def run(self, images):
        feature_images = self._fe.to_feature_image(images)

        # get AMP layer weights
        model = load_model("cls.pkl")
        activation_maps = []
        for feature_image in feature_images:
            map_ = activate_label(feature_image,
                                  0,
                                  model.coef_.reshape(-1,1),
                                  image_size=(224,224))
            activation_maps.append(map_)
        return np.array(activation_maps)
        
if __name__ == "__main__":


    # Get postive features
    images = get_list_images(DATASET_TEXT)[30:32]
    
    worker = ClsActWorker()
    maps = worker.run(images)
 
    for img, conv_map in zip(images, maps):
        conv_map = cv2.resize(conv_map, (img.shape[1], img.shape[0]))
        fig, ax = plt.subplots(nrows=2, ncols=2)
        plt.subplot(1, 2, 1)
        plt.imshow(img, alpha=0.6)
        plt.imshow(conv_map, cmap='jet', alpha=0.4)
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.show()
        # plt.savefig("{}.png".format(i), bbox_inches='tight')



