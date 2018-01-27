#-*- coding: utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator

def create_callbacks():
    from keras.callbacks import ModelCheckpoint
    checkpoint = ModelCheckpoint("weights.{epoch:02d}-{loss:.2f}.h5", 
                                 monitor='loss', 
                                 verbose=1, 
                                 save_best_only=False,
                                 mode='min', 
                                 period=1)
    return [checkpoint]


def build_generator(directory, preprocess_input, augment=False):
    if augment == True:
        data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,
                                           rotation_range=20.0,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.4,
                                           zoom_range=0.2)
    else:
        data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

    generator = data_generator.flow_from_directory(directory=directory,
                                                        target_size=(224, 224),
                                                        batch_size=8)
    return generator
