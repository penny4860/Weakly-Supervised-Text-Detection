#-*- coding: utf-8 -*-
from src.feature import FeatureExtractor

from keras.applications.resnet50 import preprocess_input
import numpy as np


# It takes about 15 minutes on the CPU.
if __name__ == "__main__":
    fe = FeatureExtractor()
    detector = fe.get_cam_model()
    detector.load_weights("weights.04-0.02.h5", by_name=True)

    import cv2
    img_path = "dataset//train//text//200.png"
    original_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(original_img, (224, 224))
    img = np.expand_dims(img, 0).astype(np.float64)

    conv_map = detector.predict(preprocess_input(img))
    print(conv_map.shape)
    conv_map = conv_map[0, :, :, 1]
    conv_map = cv2.resize(conv_map, (original_img.shape[1], original_img.shape[0]))
     
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=3)
    plt.subplot(3, 1, 1)
    plt.imshow(original_img)
    plt.subplot(3, 1, 2)
    plt.imshow(conv_map)
    plt.subplot(3, 1, 3)
    plt.imshow(original_img, alpha=0.7)
    plt.imshow(conv_map, cmap='jet', alpha=0.3)
    plt.show()
