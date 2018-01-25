#-*- coding: utf-8 -*-
import os
import cv2
import glob
import random
import re

random.seed(111)

class FileSorter:
    def __init__(self):
        pass
    
    def sort(self, list_of_strs):
        list_of_strs.sort(key=self._alphanum_key)

    def _tryint(self, s):
        try:
            return int(s)
        except:
            return s
    
    def _alphanum_key(self, s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [ self._tryint(c) for c in re.split('([0-9]+)', s) ]


def files_to_images(files):
    images = []
    for fname in files:
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    return images


def list_files(directory, pattern="*.*", n_files_to_sample=None, recursive_option=True, random_order=True):
    """list files in a directory matched in defined pattern.

    Parameters
    ----------
    directory : str
        filename of json file

    pattern : str
        regular expression for file matching
    
    n_files_to_sample : int or None
        number of files to sample randomly and return.
        If this parameter is None, function returns every files.
    
    recursive_option : boolean
        option for searching subdirectories. If this option is True, 
        function searches all subdirectories recursively.
        
    Returns
    ----------
    conf : dict
        dictionary containing contents of json file

    Examples
    --------
    """

    if recursive_option == True:
        dirs = [path for path, _, _ in os.walk(directory)]
    else:
        dirs = [directory]
    
    files = []
    for dir_ in dirs:
        for p in glob.glob(os.path.join(dir_, pattern)):
            files.append(p)
    
    FileSorter().sort(files)
        
    if n_files_to_sample is not None:
        if random_order:
            files = random.sample(files, n_files_to_sample)
        else:
            files = files[:n_files_to_sample]
    return files


def create_callbacks(weight_file="mobilenet_cls2.h5"):
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

