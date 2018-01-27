

from keras.models import Model
from keras.layers import AveragePooling2D, Flatten, Dense
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
