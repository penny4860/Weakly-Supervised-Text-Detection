

from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Dropout, Flatten, Dense, Conv2D, AveragePooling2D, BatchNormalization, Activation
from keras import backend as K

def resnet50_cam(spatial_size=14, n_classes=2, froze_pretrained_layer=True):
    resnet = ResNet50(input_shape=(224, 224, 3), weights='imagenet')

    if spatial_size == 7:
        model = Model(inputs=resnet.input, outputs=resnet.get_layer("activation_49").output)
    elif spatial_size == 14:
        model = Model(inputs=resnet.input, outputs=resnet.get_layer("activation_40").output)
    else:
        raise ValueError('spatial size {} is not allowed.'.format(spatial_size))

    x = model.output
    
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    # Add another conv layer with ReLU + GAP
    x = Conv2D(1024, (3,3), padding='same', name='cam_conv')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(spatial_size,spatial_size),
                         name='cam_average_pooling')(x)
    x = Flatten()(x)
    x = Dense(n_classes, activation='softmax', name='cam_cls')(x)
    model = Model(model.input, x)
    model.summary()

    if froze_pretrained_layer:
        for layer in model.layers[:-6]:
            layer.trainable = False
            
            print(layer.name)
    
    return model


