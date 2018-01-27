#-*- coding: utf-8 -*-
from src.feature import FeatureExtractor

from keras.models import Model
from keras.layers import Dense

# It takes about 15 minutes on the CPU.
if __name__ == "__main__":
    fe = FeatureExtractor()
    model = fe._resnet
    
    x = model.output
    x = Dense(2, activation='softmax', name='cam_cls')(x)
    model = Model(model.input, x)
    model.summary()

    for layer in model.layers[:-1]:
        layer.trainable = False
        print(layer.name)
    
