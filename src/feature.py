#-*- coding: utf-8 -*-
import numpy as np
import cv2
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.engine.topology import Layer
import tensorflow as tf
from keras.layers import AveragePooling2D, Flatten, Dense, Conv2D, BatchNormalization, Activation, Reshape

_CLASSIFICATION_LAYER = "cam_cls"
_N_LABELS = 2
_INPUT_SIZE = 224

class CamModelBuilder(object):
    def __init__(self):
        pass
    
    def get_cls_model(self):
        model = ResNet50()
        model = Model(inputs=model.input, outputs=model.get_layer("activation_40").output)
    
        x = model.output
        x = Conv2D(1024, (3,3), padding='same')(x)
        x = BatchNormalization(axis = 3)(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=(14, 14),
                             name='cam_average_pooling')(x)
        x = Flatten()(x)
        x = Dense(2, activation='softmax', name='cam_cls')(x)
        model = Model(model.input, x)
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


