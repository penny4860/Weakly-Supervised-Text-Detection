
import ast
import cv2
import numpy as np
import os
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model   

dirname = os.path.dirname(__file__)
imgnet_text_file = os.path.join(dirname, "txt", 'imagenet1000_clsid_to_human.txt')

class CustomResNet(object):

    _imgnet_text_file = imgnet_text_file
    
    def __init__(self):
        # define ResNet50 model
        model = ResNet50(weights='imagenet')

        self._final_weights = model.layers[-1].get_weights()[0]
        self._model = Model(inputs=model.input, 
                            outputs=(model.layers[-4].output,
                                     model.layers[-1].output)) 

    def forward(self, image):
        x = self._to_input_tensor(image)
        conv_map, pred_vector = self._model.predict(x)
        conv_map = np.squeeze(conv_map) 
        # get model's prediction (number between 0 and 999, inclusive)
        pred_label = np.argmax(pred_vector)
        return conv_map, pred_label
    
    def get_final_weights(self):
        return self._final_weights

    def _to_input_tensor(self, image):
        # loads RGB image as PIL.Image.Image type
        x = cv2.resize(image, (224, 224)).astype(np.float64)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        x = np.expand_dims(x, axis=0)
        # convert RGB -> BGR, subtract mean ImageNet pixel, and return 4D tensor
        return preprocess_input(x)

    def to_text_label(self, int_label):
        with open(self._imgnet_text_file) as imagenet_classes_file:
            imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())
        return imagenet_classes_dict[int_label]
