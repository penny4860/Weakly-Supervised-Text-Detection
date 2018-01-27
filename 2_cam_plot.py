#-*- coding: utf-8 -*-
from src.feature import CamModelBuilder

from keras.applications.resnet50 import preprocess_input
import numpy as np


def plot_img(image, cam_map):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=3)
    plt.subplot(3, 1, 1)
    plt.imshow(image)
    plt.subplot(3, 1, 2)
    plt.imshow(cam_map)
    plt.subplot(3, 1, 3)
    plt.imshow(image, alpha=0.7)
    plt.imshow(cam_map, cmap='jet', alpha=0.3)
    plt.show()


if __name__ == "__main__":
    fe = CamModelBuilder()
    detector = fe.get_cam_model()
    detector.load_weights("weights.h5", by_name=True)

    import cv2
    img_path = "dataset//train//text//10.png"
    original_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(original_img, (224, 224))
    img = np.expand_dims(img, 0).astype(np.float64)

    cam_map = detector.predict(preprocess_input(img))
    cam_map = cam_map[0, :, :, 1]
    cam_map = cv2.resize(cam_map, (original_img.shape[1], original_img.shape[0]))
    
    plot_img(original_img, cam_map)
    
    
    
