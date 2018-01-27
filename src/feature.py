#-*- coding: utf-8 -*-
import numpy as np
import cv2
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.engine.topology import Layer
import tensorflow as tf
from keras.layers import Dense, Reshape

_CLASSIFICATION_LAYER = "cam_cls"
_N_LABELS = 2
_INPUT_SIZE = 224

class CamModelBuilder(object):
    def __init__(self):
        model = ResNet50(weights='imagenet')
        self._resnet = Model(inputs=model.input, 
                             outputs=model.layers[-2].output)
    
    def get_cls_model(self):
        x = self._resnet.output
        x = Dense(_N_LABELS,
                  activation='softmax',
                  name=_CLASSIFICATION_LAYER)(x)
        model = Model(self._resnet.input, x)
        return model
    
    def get_cam_model(self):
        model = self.get_cls_model()
        last_conv_output = model.layers[-4].output
        x = BinearUpSampling2D((_INPUT_SIZE, _INPUT_SIZE))(last_conv_output)
        x = Reshape((_INPUT_SIZE * _INPUT_SIZE,
                     2048))(x)
        x = Dense(_N_LABELS, name=_CLASSIFICATION_LAYER)(x)
        x = Reshape((_INPUT_SIZE, _INPUT_SIZE, _N_LABELS))(x)
        
        model = Model(inputs=model.input,
                      outputs=x)
        return model


class BinearUpSampling2D(Layer):

    def __init__(self, size=(224,224), **kwargs):
        super(BinearUpSampling2D, self).__init__(**kwargs)
        self._size = size

    def call(self, x):
        return tf.image.resize_images(x, self._size)

    def compute_output_shape(self, input_shape):
        height = self._size[0] if input_shape[1] is not None else None
        width = self._size[1] if input_shape[2] is not None else None
        return (input_shape[0],
                height,
                width,
                input_shape[3])

def preprocess(images):
    xs = resize_imgs(images)
    xs = xs.astype(np.float64)
    xs = preprocess_input(xs)
    return xs

def resize_imgs(imgs):
    resized = []
    for img in imgs:
        resized.append(cv2.resize(img, (224,224)))
    resized = np.array(resized)
    return resized

if __name__ == "__main__":
    pass


