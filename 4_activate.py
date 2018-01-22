#-*- coding: utf-8 -*-
import numpy as np
from src.feature import FeatureExtractor
from src.utils import get_list_images, load_model

DATASET_TEXT = "dataset//train//text"
DATASET_NEGATIVE = "dataset//train//negative"

# It takes about 15 minutes on the CPU.
if __name__ == "__main__":


    # Get postive features
    positive_images = get_list_images(DATASET_TEXT)
 
    fe = FeatureExtractor()
    conv_output = fe.get_image_feature(positive_images[:32])
    print(conv_output.shape)
 
    # get AMP layer weights
    model = load_model("cls.pkl")
    from src.activate import activate_label
    import cv2
    import matplotlib.pyplot as plt
    for i, conv_map in enumerate(conv_output):
        map_ = activate_label(conv_map, 0, model.coef_.reshape(-1,1), image_size=(224,224))
        fig, ax = plt.subplots()
        im = cv2.resize(positive_images[i], (224, 224))
        ax.imshow(im, alpha=0.8)
        ax.imshow(map_, cmap='jet', alpha=0.2)
        plt.show()


