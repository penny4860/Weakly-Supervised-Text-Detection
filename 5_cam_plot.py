#-*- coding: utf-8 -*-
from src.feature import FeatureExtractor

from keras.models import Model
from keras.layers import Dense
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import numpy as np

def pretrained_path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    x = np.expand_dims(x, axis=0)
    # convert RGB -> BGR, subtract mean ImageNet pixel, and return 4D tensor
    return preprocess_input(x)

def text_activation_map(detector, final_weights, image_path="1.png"):
    # (None, 7, 7, 1024)    
    last_conv_output = detector.predict(pretrained_path_to_tensor(image_path))
    last_conv_output = np.squeeze(last_conv_output) 
    import scipy   
    # bilinear upsampling to resize each filtered image to size of original image 
    mat_for_mult = scipy.ndimage.zoom(last_conv_output, (32, 32, 1), order=1) # dim: 224 x 224 x 2048
    # get AMP layer weights
    text_weights = final_weights[:, 1] # dim: (2048,) 
    # get class activation map for object class that is predicted to be in the image
    text_map = np.dot(mat_for_mult.reshape((224*224, 2048)), text_weights).reshape(224,224) # dim: 224 x 224
    return text_map


from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf

class BinearUpSampling2D(Layer):

    def __init__(self, size=(224,224), **kwargs):
        super(BinearUpSampling2D, self).__init__(**kwargs)
        self._size = size

    def call(self, x):
        return tf.image.resize_images(x, self._size)

    def compute_output_shape(self, input_shape):
        height = self._size[0] if input_shape[1] is not None else None
        width = self._size[1] if input_shape[2] is not None else None
        return (input_shape[0],
                height,
                width,
                input_shape[3])
        
# It takes about 15 minutes on the CPU.
if __name__ == "__main__":
    fe = FeatureExtractor()
    model = fe.get_cls_model()
    model.load_weights("weights.04-0.02.h5")
    final_weights = model.layers[-1].get_weights()[0]
    
    # (None, 7, 7, 2048)
    img_path = "dataset//train//text//200.png"
    last_conv_output = model.layers[-4].output
    img_sized_conv_output = BinearUpSampling2D((224,224))(last_conv_output)
    
    detector = Model(inputs=model.input,
                     outputs=img_sized_conv_output)
    
    output = detector.predict(pretrained_path_to_tensor(img_path))
    print(output.shape, final_weights.shape)
    conv_map = np.dot(output.reshape((224*224, 2048)), final_weights[:,1]).reshape(224,224) # dim: 224 x 224

    
#     detector = Model(inputs=model.input, outputs=model.layers[-4].output)
#     conv_map = text_activation_map(detector, final_weights, img_path)
     
    import cv2
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    conv_map = cv2.resize(conv_map, (img.shape[1], img.shape[0]))
     
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=3)
    plt.subplot(3, 1, 1)
    plt.imshow(img)
    plt.subplot(3, 1, 2)
    plt.imshow(conv_map)
    plt.subplot(3, 1, 3)
    plt.imshow(img, alpha=0.7)
    plt.imshow(conv_map, cmap='jet', alpha=0.3)
    plt.show()
