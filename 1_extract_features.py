#-*- coding: utf-8 -*-
import numpy as np
from src.feature import FeatureExtractor
from src.utils import files_to_images, list_files

DATASET_TEXT = "dataset//train//text"
DATASET_SVT = "..//dataset//svt1"
DATASET_KAIST = "..//dataset//kaist"
DATASET_NEGATIVE = "..//dataset//images"

# It takes about 15 minutes on the CPU.
if __name__ == "__main__":

    fe = FeatureExtractor()


    # Get negative features
    files = list_files(DATASET_SVT, "*.jpg", n_files_to_sample=350, random_order=False)
    positive_images = files_to_images(files)
    positive_features = fe.to_feature_vector(positive_images)
    np.save("svt_features", positive_features)
    print(positive_features.shape) # (350, 2048)
    
#     # Get negative features
#     negative_images = get_list_images(DATASET_NEGATIVE, 1800)
#     negative_features = fe.to_feature_vector(negative_images)
#     np.save("negative_features", negative_features)
#     print(negative_features.shape) # (1800, 2048)

    

