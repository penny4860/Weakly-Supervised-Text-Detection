#-*- coding: utf-8 -*-
from src.feature import FeatureExtractor

from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50, preprocess_input

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

    optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = optimizer,
                  metrics = ['accuracy'])
    
    from src.utils import build_generator, create_callbacks
    train_generator = build_generator("dataset//train", preprocess_input, augment=True)
    model.fit_generator(train_generator,
                        steps_per_epoch = len(train_generator),
                        callbacks = create_callbacks("resnet_cls.h5"),
                        epochs=20)

