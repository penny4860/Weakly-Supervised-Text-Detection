#-*- coding: utf-8 -*-
import numpy as np
from src.feature import FeatureExtractor
from src.utils import get_list_images, list_files

DATASET_TEXT = "dataset//train//text"

DATASET_KAIST = "..//dataset//kaist"
DATASET_NEGATIVE = "..//dataset//images"


def to_images(files):
    import cv2
    images = []
    for fname in files:
        img = cv2.imread(fname)
        print(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    return images

# It takes about 15 minutes on the CPU.
if __name__ == "__main__":

    fe = FeatureExtractor()


    # Get negative features
    files = list_files(DATASET_KAIST, "*.jpg", n_files_to_sample=400, random_order=False)
    positive_images = to_images(files)
    positive_features = fe.to_feature_vector(positive_images)
    np.save("kaist_text_features", positive_features)
    print(positive_features.shape) # (350, 2048)
    
#     # Get negative features
#     negative_images = get_list_images(DATASET_NEGATIVE, 1800)
#     negative_features = fe.to_feature_vector(negative_images)
#     np.save("negative_features", negative_features)
#     print(negative_features.shape) # (1800, 2048)

    

