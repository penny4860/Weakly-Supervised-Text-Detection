
from keras.optimizers import Adam
from keras.applications.resnet50 import preprocess_input
from src.resnet import resnet50_cam
from src.utils import build_generator, create_callbacks

if __name__ == "__main__":
    model = resnet50_cam()
    model.summary()
    optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = optimizer,
                  metrics = ['accuracy'])

    train_generator = build_generator("dataset//train", preprocess_input, augment=True)
    model.fit_generator(train_generator,
                        steps_per_epoch = len(train_generator),
#                             validation_data = valid_generator,
#                             validation_steps = len(valid_generator),
                        callbacks = create_callbacks("resnet.h5"),
                        epochs=20)


