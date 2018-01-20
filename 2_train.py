
from src.net import mobilenet_binary_classifier
from src.utils import build_generator, create_callbacks

mobilenet = mobilenet_binary_classifier()
train_generator, valid_generator = build_generator(train_directory="..//weak_dataset//train",
                                                   valid_directory="..//weak_dataset//val")
mobilenet.fit_generator(train_generator,
                        steps_per_epoch = len(train_generator),
                        validation_data = valid_generator,
                        validation_steps = len(valid_generator),
                        callbacks        = create_callbacks(),
                        epochs=20)


