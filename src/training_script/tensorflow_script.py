from src.config import PATH_PERSO_OUTPUTS

# from src.config import (
#     PATH_PERSO_VALID_SAME_PROPOR,
#     PATH_PERSO_TRAIN_SAME_PROPOR,
#     PATH_PERSO_TEST_SAME_PROPOR,
# )
# from src.config import (
#     PATH_PERSO_VALID_INTRA,
#     PATH_PERSO_TRAIN_INTRA,
#     PATH_PERSO_TEST_INTRA,
# )
from src.config import PATH_PERSO_VALID_ADD, PATH_PERSO_TRAIN_ADD, PATH_PERSO_TEST_ADD
from src.utils.directory_process import create_directory
from src.utils.data_process import get_weight

import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    Dropout,
    MaxPooling2D,
    BatchNormalization,
    Input,
)
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU disponible et activé")
    except RuntimeError as e:
        print(e)
else:
    print("Aucun GPU trouvé, utilisation du CPU")


class CustomModelTrainer(tf.keras.utils.Sequence):
    def __init__(
        self,
        dataset_train,
        validation_dir,
        testing_dir,
        pathout,
        epochs=50,
        resize=(150, 150),
        rescale=1.0 / 255,
        color="rgb",
        mode="binaire",
        save_model=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.dataset_train = dataset_train
        self.validation_dir = validation_dir
        self.testing_dir = testing_dir

        self.rescale = rescale
        self.target_size = resize
        self.color_mode = color
        self.class_mode = mode
        self.epochs = epochs
        self.save_model = save_model

        self.batch_size = 32

        self.train_datagen = ImageDataGenerator(
            rescale=self.rescale,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2,
        )
        self.validation_datagen = ImageDataGenerator(
            rescale=self.rescale, validation_split=0.2
        )
        self.test_datagen = ImageDataGenerator(rescale=self.rescale)

        self.main_model_dir = pathout + r"/models/"
        self.main_log_dir = pathout + r"/logs/"

        self.model_dir = self.main_model_dir + time.strftime("%Y-%m-%d_%H-%M-%S") + "/"
        self.log_dir = self.main_log_dir + time.strftime("%Y-%m-%d_%H-%M-%S")

        create_directory(self.model_dir)
        create_directory(self.log_dir)

    def get_train_generator(self):
        return self.train_datagen.flow_from_directory(
            self.dataset_train,
            target_size=self.target_size,
            color_mode=self.color_mode,
            class_mode=self.class_mode,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def get_validation_generator(self):
        return self.validation_datagen.flow_from_directory(
            self.validation_dir,
            target_size=self.target_size,
            color_mode=self.color_mode,
            class_mode=self.class_mode,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def get_test_generator(self):
        return self.test_datagen.flow_from_directory(
            self.testing_dir,
            color_mode=self.color_mode,
            target_size=self.target_size,
            class_mode=self.class_mode,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def get_callbacks(self):
        checkpoint = ModelCheckpoint(
            self.model_dir + "checkpoint_{epoch:02d}.keras",
            monitor="val_loss",
            verbose=1,
            save_freq="epoch",
        )

        tensorboard = TensorBoard(
            log_dir=self.log_dir,
            update_freq="epoch",
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            mode="max",
            cooldown=2,
            min_lr=1e-8,
            verbose=1,
        )

        return [checkpoint, reduce_lr, tensorboard]

    def build_model(self):
        model = Sequential(
            [
                Input(shape=(150, 150, 3)),
                Conv2D(32, (3, 3), strides=1, padding="same", activation="relu"),
                BatchNormalization(),
                MaxPooling2D((2, 2), strides=2, padding="same"),
                Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"),
                BatchNormalization(),
                MaxPooling2D((2, 2), strides=2, padding="same"),
                Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"),
                BatchNormalization(),
                MaxPooling2D((2, 2), strides=2, padding="same"),
                Flatten(),
                Dense(units=128, activation="relu"),
                Dropout(0.2),
                Dense(units=3, activation="softmax"),
            ]
        )

        optimizer = Adam(learning_rate=0.001)
        loss = "categorical_crossentropy"
        metrics = ["accuracy", "AUC"]

        model.compile(optimizer, loss=loss, metrics=metrics)
        return model

    def train_model(self):
        train_generator = self.get_train_generator()
        validation_generator = self.get_validation_generator()
        callbacks = self.get_callbacks()

        steps_per_epoch = train_generator.samples // train_generator.batch_size
        validation_steps = (
            validation_generator.samples // validation_generator.batch_size
        )

        model = self.build_model()

        model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
            verbose=2,
            callbacks=callbacks,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            class_weight=dict(enumerate(get_weight(train_generator.classes))),
        )
        print("Completed Model Training")

        if self.save_model:
            model.save(self.model_dir + "model.keras")


if __name__ == "__main__":

    # dataset_train = PATH_PERSO_TRAIN_SAME_PROPOR
    # validation_dir = PATH_PERSO_VALID_SAME_PROPOR
    # testing_dir = PATH_PERSO_TEST_SAME_PROPOR

    # dataset_train = PATH_PERSO_TRAIN_INTRA
    # validation_dir = PATH_PERSO_VALID_INTRA
    # testing_dir = PATH_PERSO_TEST_INTRA

    dataset_train = PATH_PERSO_TRAIN_ADD
    validation_dir = PATH_PERSO_VALID_ADD
    testing_dir = PATH_PERSO_TEST_ADD

    pathout = PATH_PERSO_OUTPUTS
    mode = "categorical"
    epoch = 50

    trainer = CustomModelTrainer(
        dataset_train,
        validation_dir,
        testing_dir,
        mode=mode,
        pathout=pathout,
        epochs=epoch,
    )
    trainer.train_model()
