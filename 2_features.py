#-*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from src.feature import FeatureExtractor

DATASET_TEXT = "dataset//train//text"
DATASET_NEGATIVE = "dataset//train//negative"

def get_list_images(folder):
    def _get_files(folder):
        files = os.listdir(folder)
        files = [os.path.join(folder, fname) for fname in files]
        return files
    
    files = _get_files(folder)
    images = []
    for fname in files:
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    return images

if __name__ == "__main__":

    fe = FeatureExtractor()
    
    # Get list of images
    positive_images = get_list_images(DATASET_TEXT)
    positive_features = fe.run(positive_images)
    np.save("positive_features", positive_features)
    print(positive_features.shape)

    negative_images = get_list_images(DATASET_NEGATIVE)
    negative_features = fe.run(negative_images)
    np.save("negative_features", negative_features)
    print(negative_features.shape)

