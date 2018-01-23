#-*- coding: utf-8 -*-
import numpy as np
from src.feature import FeatureExtractor
from src.utils import get_list_images

DATASET_TEXT = "dataset//train//text"
DATASET_NEGATIVE = "..//dataset//images"

# It takes about 15 minutes on the CPU.
if __name__ == "__main__":

    fe = FeatureExtractor()

    # Get negative features
    positive_images = get_list_images(DATASET_TEXT, 1800)
    positive_features = fe.to_feature_vector(positive_images)
    np.save("positive_features", positive_features)
    print(positive_features.shape) # (350, 2048)
    
    # Get negative features
    negative_images = get_list_images(DATASET_NEGATIVE, 1800)
    negative_features = fe.to_feature_vector(negative_images)
    np.save("negative_features", negative_features)
    print(negative_features.shape) # (1800, 2048)

    

