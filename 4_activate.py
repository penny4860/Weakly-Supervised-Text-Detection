#-*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt

from src.feature import FeatureExtractor
from src.utils import get_list_images, load_model
from src.activate import activate_label

DATASET_TEXT = "dataset//train//text"

if __name__ == "__main__":


    # Get postive features
    positive_images = get_list_images(DATASET_TEXT)[30:32]
 
    fe = FeatureExtractor()
    conv_maps = fe.get_image_feature(positive_images)
    print(conv_maps.shape)
 
    # get AMP layer weights
    model = load_model("cls.pkl")
    for i, (img, conv_map) in enumerate(zip(positive_images, conv_maps)):
        map_ = activate_label(conv_map, 0, model.coef_.reshape(-1,1), image_size=(224,224))
        map_ = cv2.resize(map_, (img.shape[1], img.shape[0]))
        
        fig, ax = plt.subplots(nrows=2, ncols=2)
        plt.subplot(1, 2, 1)
        plt.imshow(img, alpha=0.6)
        plt.imshow(map_, cmap='jet', alpha=0.4)
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.show()
        # plt.savefig("{}.png".format(i), bbox_inches='tight')



