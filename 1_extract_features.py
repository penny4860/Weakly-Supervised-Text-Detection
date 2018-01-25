#-*- coding: utf-8 -*-
import numpy as np
from src.feature import FeatureExtractor
from src.utils import files_to_images, list_files
import shutil
import os

DATASET_TEXT = "dataset//train//text"
DATASET_SVT = "..//dataset//svt1"
DATASET_KAIST = "..//dataset//kaist"
DATASET_NEGATIVE = "..//dataset//sun_dataset"

# It takes about 15 minutes on the CPU.
if __name__ == "__main__":

    # Get negative features
    negative_images = list_files(DATASET_NEGATIVE, "*.jpg", 1800)
    dname = "dataset//train//negative"
    
    for fname in negative_images:
        new_name = os.path.join(dname, os.path.split(fname)[1])
        shutil.copyfile(fname, new_name)
        print(new_name)

