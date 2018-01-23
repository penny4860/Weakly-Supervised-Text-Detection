#-*- coding: utf-8 -*-
import numpy as np
from src.feature import FeatureExtractor
from src.utils import get_list_images

DATASET_TEXT = "dataset//train//text"
DATASET_NEGATIVE = "dataset//train//negative"

# It takes about 15 minutes on the CPU.
if __name__ == "__main__":

    fe = FeatureExtractor()
    
    # Get postive features
    positive_images = get_list_images(DATASET_TEXT)
    positive_features = fe.to_feature_vector(positive_images)
    np.save("positive_features", positive_features)
    print(positive_features.shape) # (350, 2048)

    # Get negative features
    negative_images = get_list_images(DATASET_NEGATIVE)
    negative_features = fe.to_feature_vector(negative_images)
    np.save("negative_features", negative_features)
    print(negative_features.shape) # (1225, 2048)

    

