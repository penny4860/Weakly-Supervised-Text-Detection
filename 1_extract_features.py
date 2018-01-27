#-*- coding: utf-8 -*-
import numpy as np
from src.feature import FeatureExtractor
from src.utils import files_to_images, list_files

DATASET_TEXT = "dataset//train//text"
DATASET_NEGATIVE = "dataset//train//negative"

# It takes about 15 minutes on the CPU.
if __name__ == "__main__":
    fe = FeatureExtractor()

    # Get negative features
    files = list_files(DATASET_TEXT, random_order=False)
    positive_images = files_to_images(files)
    positive_features = fe.to_feature_vector(positive_images)
    np.save("svhn_features", positive_features)
    print(positive_features.shape) # (350, 2048)
    
    # Get negative features
    files = list_files(DATASET_NEGATIVE, random_order=False)
    negative_images = files_to_images(files)
    negative_features = fe.to_feature_vector(negative_images)
    np.save("negative_features", negative_features)
    print(negative_features.shape) # (1225, 2048)

    

