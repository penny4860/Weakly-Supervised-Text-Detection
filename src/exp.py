

from keras.models import Model
from keras.layers import AveragePooling2D, Flatten, Dense, Conv2D, BatchNormalization, Activation
from keras.applications.resnet50 import ResNet50, preprocess_input

def get_model_14x14():
    model = ResNet50()
    model = Model(inputs=model.input, outputs=model.get_layer("activation_40").output)

    x = model.output
    x = AveragePooling2D(pool_size=(14, 14),
                         name='cam_average_pooling')(x)
    x = Flatten()(x)
    x = Dense(2, activation='softmax', name='cam_cls')(x)
    model = Model(model.input, x)
    return model

def get_model_conv_14x14():
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

