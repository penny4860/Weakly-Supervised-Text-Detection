#-*- coding: utf-8 -*-
from src.feature import CamModelBuilder

from keras.optimizers import Adam
from keras.applications.resnet50 import preprocess_input
from src.keras_utils import build_generator, create_callbacks

from keras.models import Model
from keras.layers import AveragePooling2D, Flatten, Dense

def get_model():
    builder = CamModelBuilder()
    model = builder._resnet
    model = Model(inputs=model.input, outputs=model.get_layer("activation_40").output)

    x = model.output
    x = AveragePooling2D(pool_size=(14, 14),
                         name='cam_average_pooling')(x)
    x = Flatten()(x)
    x = Dense(2, activation='softmax', name='cam_cls')(x)
    model = Model(model.input, x)
    return model

if __name__ == "__main__":
    model = get_model()

    fixed_layers = []
    for layer in model.layers[:-1]:
        layer.trainable = False
        fixed_layers.append(layer.name)
    print(fixed_layers)

    optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = optimizer,
                  metrics = ['accuracy'])
      
    train_generator = build_generator("dataset//train", preprocess_input, augment=True)
    model.fit_generator(train_generator,
                        steps_per_epoch = len(train_generator),
                        callbacks = create_callbacks(),
                        epochs=20)

