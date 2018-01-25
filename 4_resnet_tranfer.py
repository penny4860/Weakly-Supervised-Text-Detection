
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Dropout, Flatten, Dense, Conv2D, AveragePooling2D

from src.resnet import resnet50_cam

if __name__ == "__main__":
    model = resnet50_cam()
    optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
    
    train_generator = build_generator("dataset//train")
    # valid_generator = build_generator("dataset//valid")

    mobilenet.fit_generator(train_generator,
                            steps_per_epoch = len(train_generator),
#                             validation_data = valid_generator,
#                             validation_steps = len(valid_generator),
                            callbacks        = create_callbacks(),
                            epochs=20)


