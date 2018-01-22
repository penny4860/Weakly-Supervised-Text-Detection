#-*- coding: utf-8 -*-
import os

DATASET_TEXT = "dataset//train//text"
DATASET_NEGATIVE = "dataset//train//negative"

if __name__ == "__main__":
    text_imgs = os.listdir(DATASET_TEXT)            # 350
    negative_imgs = os.listdir(DATASET_NEGATIVE)    # 1225
    print(len(text_imgs), len(negative_imgs))

