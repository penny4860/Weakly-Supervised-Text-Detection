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

def plot_img(image, cam_map, show=True, save_filename=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=3)
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.title("original image")
    plt.imshow(image)
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.title("text activation map")
    plt.imshow(cam_map)
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.imshow(image, alpha=0.7)
    plt.imshow(cam_map, cmap='jet', alpha=0.3)
    if show:
        plt.show()
    if save_filename:
        plt.savefig(save_filename, bbox_inches='tight')
        print("{} is saved".format(save_filename))


