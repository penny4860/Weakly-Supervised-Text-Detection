

from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
import cv2     
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input

from keras.preprocessing import image    
from keras.models import Model   
import sys

def mobilenet_binary_classifier():
    # build the VGG16 network
    model = MobileNet(input_shape=(224, 224, 3), weights='imagenet', include_top=True)
    model = Model(inputs=model.input, outputs=model.layers[-6].output)
    x = model.output
    x = Dense(2, activation='softmax', init='uniform')(x)
    model = Model(model.input, x)
    model.summary()
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
    # optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics=['accuracy'])
    return model

def pretrained_path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    x = np.expand_dims(x, axis=0)
    # convert RGB -> BGR, subtract mean ImageNet pixel, and return 4D tensor
    return preprocess_input(x)

def get_detector():
    model = mobilenet_binary_classifier()
    model.load_weights('mobilenet_cls.h5')
    final_weights = model.layers[-1].get_weights()[0]
    final_weights = final_weights.reshape(-1, 2)
    detector = Model(inputs=model.input, outputs=model.layers[-3].output)
    return detector, final_weights

def text_activation_map(image_path="1.png"):

    detector, final_weights = get_detector()
    detector.summary()
    
    # (None, 7, 7, 1024)    
    last_conv_output = detector.predict(pretrained_path_to_tensor(image_path))
    last_conv_output = np.squeeze(last_conv_output) 
    import scipy   
    # bilinear upsampling to resize each filtered image to size of original image 
    mat_for_mult = scipy.ndimage.zoom(last_conv_output, (32, 32, 1), order=1) # dim: 224 x 224 x 2048
    # get AMP layer weights
    text_weights = final_weights[:, 1] # dim: (2048,) 
    # get class activation map for object class that is predicted to be in the image
    text_map = np.dot(mat_for_mult.reshape((224*224, 1024)), text_weights).reshape(224,224) # dim: 224 x 224
    return text_map

img_path = "images//t.jpg"
# load image, convert BGR --> RGB, resize image to 224 x 224,
im = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (224, 224))
# plot image
fig, ax = plt.subplots()
ax.imshow(im, alpha=0.5)
# get class activation map
text_map = text_activation_map(img_path)
# plot class activation map
ax.imshow(text_map, cmap='jet', alpha=0.5)
plt.show()

# preds = model.predict(pretrained_path_to_tensor("dataset//text//00001.png"))
# print(preds)
# preds = model.predict(pretrained_path_to_tensor("dataset//negative//000022.png"))
# print(preds)
# preds = model.predict(pretrained_path_to_tensor("1.png"))
# print(preds)
# preds = model.predict(pretrained_path_to_tensor("7.png"))
# print(preds)











