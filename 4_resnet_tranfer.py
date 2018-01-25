
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Dropout, Flatten, Dense, Conv2D, AveragePooling2D


def create_callbacks(weight_file="mobilenet_cls2.h5"):
    from keras.callbacks import ModelCheckpoint
    checkpoint = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.h5", 
                                 monitor='loss', 
                                 verbose=1, 
                                 save_best_only=False,
                                 mode='min', 
                                 period=1)
    return [checkpoint]


def build_generator(train_directory="dataset",
                    valid_directory="..//dataset//weakly-digit-detector//val"):

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                       rotation_range=20.0,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.4,
                                       zoom_range=0.2)
    valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(directory=train_directory,
                                                        target_size=(224, 224),
                                                        batch_size=8)
    valid_generator = valid_datagen.flow_from_directory(directory=valid_directory,
                                                        target_size=(224, 224),
                                                        batch_size=8)
    return train_generator, valid_generator


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

if __name__ == "__main__":
    model = resnet50_cam()
    optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
    
    train_generator, valid_generator = build_generator()
    mobilenet.fit_generator(train_generator,
                            steps_per_epoch = len(train_generator),
                            validation_data = valid_generator,
                            validation_steps = len(valid_generator),
                            callbacks        = create_callbacks(),
                            epochs=20)


