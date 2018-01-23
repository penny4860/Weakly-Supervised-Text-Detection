#-*- coding: utf-8 -*-

import numpy as np
import scipy   
from src.feature import FeatureExtractor

class ClsActWorker(object):
    
    def __init__(self, cls_weights):
        self._fe = FeatureExtractor()
        self._cls_weights = cls_weights
    
    def run(self, images):
        feature_images = self._fe.to_feature_image(images)

        activation_maps = []
        for feature_image in feature_images:
            map_ = activate_label(feature_image,
                                  0,
                                  self._cls_weights,
                                  image_size=(224,224))
            activation_maps.append(map_)
        return np.array(activation_maps)


def activate_label(conv_map, cls_label, final_weight, image_size=(224,224)):
    """
    # Args
        conv_map : (h_conv, w_conv, n_features)
        final_weight : (n_features, n_class_labels)
        cls_label : int
        image_size : (h, w)
    
    # Returns
        activate_map : (input_size[0], input_size[1])
    """
    assert conv_map.shape[-1] == final_weight.shape[0]
    
    n_features = conv_map.shape[-1]
    h, w = image_size
    w_conv, h_conv = conv_map.shape[:2]
    mul_w = int(w / w_conv)
    mul_h = int(h / h_conv)

    # (w, h, n_features)
    conv_map_scaled = scipy.ndimage.zoom(conv_map, (mul_h, mul_w, 1), order=1)
    conv_map_scaled = conv_map_scaled.reshape((h*w, n_features))

    # get AMP layer weights
    feature_to_label = final_weight[:, cls_label] # dim: (2048,) 
    # get class activation map for object class that is predicted to be in the image
    activation_map = np.dot(conv_map_scaled, feature_to_label)
    activation_map = activation_map.reshape(h, w)
    return activation_map
