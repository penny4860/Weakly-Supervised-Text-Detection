
import os
import cv2
import numpy as np
from scipy.io import loadmat

data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "dataset",
                           "weakly-digit-detector",
                           "val",
                           "text")


def matfile_to_images(matfile, dst_directory):
    mat = loadmat(matfile)
    
    # RGB-ordered
    images = mat["X"]
    images = np.rollaxis(images, 3, 0)
    
    for i, img in enumerate(images):
        image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        fname = str(i+1).zfill(5) + ".png"
        path = os.path.join(dst_directory, fname)
        cv2.imwrite(path, image_bgr)
        print(path)

matfile_to_images("test_32x32.mat",
                  os.path.join(data_folder))


