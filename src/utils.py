#-*- coding: utf-8 -*-
import os
import cv2

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input

def build_generator(train_directory="dataset",
                    valid_directory="..//dataset//weakly-digit-detector//val",
                    preprocess_func=preprocess_input):

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_func,
                                       rotation_range=20.0,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.4,
                                       zoom_range=0.2)
    valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)

    train_generator = train_datagen.flow_from_directory(directory=train_directory,
                                                        target_size=(224, 224),
                                                        batch_size=12)
    if valid_directory is None:
        valid_generator = None
    else:
        valid_generator = valid_datagen.flow_from_directory(directory=valid_directory,
                                                            target_size=(224, 224),
                                                            batch_size=12)
    return train_generator, valid_generator

def create_callbacks(weight_file=None):
    from keras.callbacks import ModelCheckpoint
    if weight_file is None:
        weight_file = "weights.{epoch:02d}-{val_loss:.2f}.h5"
    checkpoint = ModelCheckpoint(weight_file, 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_weights_only=True,
                                 save_best_only=False,
                                 mode='min', 
                                 period=1)
    return [checkpoint]


def download(url="http://ufldl.stanford.edu/housenumbers/train_32x32.mat", fname="train_32x32.mat"):
    import requests
    print("downloading....")
    r = requests.get(url) # create HTTP response object
    with open(fname, 'wb') as f:
        f.write(r.content)    
    print("download is done")

def get_list_images(folder):
    """
    # Args
        folder : str
            folder which has image files
    
    # Returns
        images : list of image array
    """
    def _get_files(folder):
        files = os.listdir(folder)
        files = [os.path.join(folder, fname) for fname in files]
        return files
    
    files = _get_files(folder)
    images = []
    for fname in files:
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    return images





