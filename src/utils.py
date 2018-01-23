#-*- coding: utf-8 -*-
import os
import cv2

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

