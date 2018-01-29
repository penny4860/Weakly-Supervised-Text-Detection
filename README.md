
# Weakly Supervised Text Detector

<img src="sample.png" height="300">

I am implementing a detection algorithm with a classification data set that does not have annotation information for the bounding box. I used the class activation mapping proposed in [Learning Deep Features for Discriminative Localization](https://arxiv.org/pdf/1512.04150.pdf).

## Requirements

* python 3.6
* tensorflow 1.2.1
* keras 2.1.1
* opencv 3.3.0

## Usage

The procedure to build detector is as follows:

### 1. Fine Tuning  ([1_train.py](https://github.com/penny4860/Weakly-Supervised-Text-Detection/blob/master/1_train.py))

First build a binary classifier using the pretrained resnet50 structure. As with the structure in the [Learning Deep Features for Discriminative Localization](https://arxiv.org/pdf/1512.04150.pdf), I removed all layers after the activation layer with spatial resoultion of 14x14, and added a convolution layer of 3x3 size instead. The last few layers of the network have the following structure.

```
activation_40 (Activation)      (None, 14, 14, 1024) 0           add_13[0][0]                     
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 14, 14, 1024) 9438208     activation_40[0][0]              
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 14, 14, 1024) 4096        conv2d_1[0][0]                   
__________________________________________________________________________________________________
activation_50 (Activation)      (None, 14, 14, 1024) 0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
cam_average_pooling (AveragePoo (None, 1, 1, 1024)   0           activation_50[0][0]              
__________________________________________________________________________________________________
flatten_2 (Flatten)             (None, 1024)         0           cam_average_pooling[0][0]        
__________________________________________________________________________________________________
cam_cls (Dense)                 (None, 2)            2050        flatten_2[0][0]                  
```

Next, you should train a model that can distinguish between images that contain text and those that do not. The dataset is stored in this repository. To reproduce the result, just run ``1_train.py``.

### 2. Plot Class Actication Map ([2_cam_plot.py](https://github.com/penny4860/Weakly-Supervised-Text-Detection/blob/master/2_cam_plot.py))

The description will be added...

## Results

<img src="svt2.png" height="300">

## References

* [Learning Deep Features for Discriminative Localization](https://arxiv.org/pdf/1512.04150.pdf)

