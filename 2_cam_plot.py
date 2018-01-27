#-*- coding: utf-8 -*-
from src.feature import CamModelBuilder
from src.utils import plot_img, list_files
from keras.applications.resnet50 import preprocess_input
import numpy as np
import cv2


from src.feature import BinearUpSampling2D
from keras.layers import Reshape, Dense, AveragePooling2D, Flatten, Conv2D
from keras.models import Model

from src.exp import get_model_14x14

if __name__ == "__main__":
    model = get_model_14x14()
    last_conv_output = model.layers[-4].output
    x = BinearUpSampling2D((224, 224))(last_conv_output)
    x = Reshape((224 * 224,
                 1024))(x)
    x = Dense(2, name="cam_cls")(x)
    x = Reshape((224, 224, 2))(x)
    
    detector = Model(inputs=model.input,
                  outputs=x)
    detector.load_weights("weights.07-0.02.h5", by_name=True)
    detector.summary()

    imgs = list_files("dataset//train//text")
     
    for i, img_path in enumerate(imgs):
        original_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
         
        img = cv2.resize(original_img, (224, 224))
        img = np.expand_dims(img, 0).astype(np.float64)
     
        cam_map = detector.predict(preprocess_input(img))
        cam_map = cam_map[0, :, :, 1]
        cam_map = cv2.resize(cam_map, (original_img.shape[1], original_img.shape[0]))
         
        plot_img(original_img, cam_map, show=False, save_filename="{}.png".format(i+1))
    
    
    
