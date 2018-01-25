
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Dropout, Flatten, Dense, Conv2D, AveragePooling2D


def resnet50_cam(spatial_size=14, n_classes=2, froze_pretrained_layer=True):
    resnet = ResNet50(input_shape=(224, 224, 3), weights='imagenet')

    if spatial_size == 7:
        model = Model(inputs=resnet.input, outputs=resnet.get_layer("activation_49").output)
    elif spatial_size == 14:
        model = Model(inputs=resnet.input, outputs=resnet.get_layer("activation_40").output)
    else:
        raise ValueError('spatial size {} is not allowed.'.format(spatial_size))

    x = model.output
    
    # Add another conv layer with ReLU + GAP
    x = Conv2D(1024, (3,3), activation='relu', padding='same', name='cam_conv')(x)
    x = AveragePooling2D(pool_size=(spatial_size,spatial_size),
                         name='cam_average_pooling')(x)
    x = Flatten()(x)
    x = Dense(n_classes, activation='softmax', name='cam_cls')(x)
    model = Model(model.input, x)

    if froze_pretrained_layer:
        for layer in model.layers[:-4]:
            layer.trainable = False
    return model

model = resnet50_cam()
model.compile()
model.summary()




