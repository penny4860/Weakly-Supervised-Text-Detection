#-*- coding: utf-8 -*-
import numpy as np
from src.feature import FeatureExtractor
from src.utils import get_list_images, load_model

DATASET_TEXT = "dataset//train//text"
DATASET_NEGATIVE = "dataset//train//negative"

# It takes about 15 minutes on the CPU.
if __name__ == "__main__":


    # Get postive features
    positive_images = get_list_images(DATASET_TEXT)[30:62]
 
    fe = FeatureExtractor()
    conv_maps = fe.get_image_feature(positive_images)
    print(conv_maps.shape)
 
    # get AMP layer weights
    model = load_model("cls.pkl")
    from src.activate import activate_label
    import cv2
    import matplotlib.pyplot as plt
    
    # Todo : original image + activation map 을 subplot 으로 출력
    # Todo : original image size (또는 그 비율로 출력)
    for i, (img, conv_map) in enumerate(zip(positive_images, conv_maps)):
        map_ = activate_label(conv_map, 0, model.coef_.reshape(-1,1), image_size=(224,224))
        fig, ax = plt.subplots()
        im = cv2.resize(img, (224, 224))
        ax.imshow(im, alpha=0.6)
        ax.imshow(map_, cmap='jet', alpha=0.4)
        # plt.show()
        
        plt.savefig("{}.png".format(i), bbox_inches='tight')
#         print(write_path)



