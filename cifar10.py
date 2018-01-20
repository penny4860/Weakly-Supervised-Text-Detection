
from keras.datasets import cifar10
import cv2
import os

def write_cifar10_file(n_samples, category="train", directory="dataset//train//negative"):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    if category == "train":
        imgs = x_train[:n_samples]
    else:
        imgs = x_test[:n_samples]
    for i, img in enumerate(imgs):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        fname = "cifar_{}.png".format(i+1)
        path = os.path.join(directory, fname)
        print(path)
        cv2.imwrite(path, img)


write_cifar10_file(800, "train", "dataset//train//negative")
write_cifar10_file(100, "test", "dataset//val//negative")


