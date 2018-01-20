
from keras.datasets import cifar10
import cv2
import os

PROJECT_FOLDER = os.path.dirname(__file__)
DATASET_FOLDER = os.path.join(os.path.dirname(PROJECT_FOLDER), "weak_dataset")


def build_dataset_tree(dataset_root=DATASET_FOLDER):
    def _mkdir(dname):
        if not os.path.isdir(dname):
            os.mkdir(dname)
    _mkdir(dataset_root)
    _mkdir(os.path.join(dataset_root, "train"))
    _mkdir(os.path.join(dataset_root, "train", "text"))
    _mkdir(os.path.join(dataset_root, "train", "negative"))
    _mkdir(os.path.join(dataset_root, "val"))
    _mkdir(os.path.join(dataset_root, "val", "text"))
    _mkdir(os.path.join(dataset_root, "val", "negative"))


def to_file(img, directory, fname):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    path = os.path.join(directory, fname)
    print(path)
    cv2.imwrite(path, img)
    

def write_cifar10_file(directory=DATASET_FOLDER):
    (x_train, _), (x_test, _) = cifar10.load_data()
    
    train_dir = os.path.join(directory, "train", "negative")
    for i, img in enumerate(x_train):
        to_file(img, train_dir, fname="{}.png".format(i+1))
    
    valid_dir = os.path.join(directory, "val", "negative")
    for i, img in enumerate(x_test):
        to_file(img, valid_dir, fname="{}.png".format(i+1))


def write_positive(directory=DATASET_FOLDER):
    fname = "train_32x32.mat"
    if os.path.exists(fname):
        print("{} is already exists in project.".format(fname))
    else:
        download(url="http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                 fname=fname)
    mat = loadmat(fname)
    mat = loadmat("positive.mat")
    images = mat["X"]
    images = np.rollaxis(images, 3, 0)
 
    print("===================================================================")    
    print(images.min(), images.max())
    print("===================================================================")    
 
    labels = np.zeros((len(images), 1), dtype=np.int) + 1
    return images[:n_samples], labels[:n_samples]


build_dataset_tree()
write_cifar10_file()


# write_cifar10_file(800, "train", "dataset//train//negative")
# write_cifar10_file(100, "test", "dataset//val//negative")


