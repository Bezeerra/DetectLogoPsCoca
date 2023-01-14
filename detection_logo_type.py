import tensorflow as tf
from keras.optimizers import Adam

batch_size = 32
img_height = 180
img_width = 180


class DetectionLogoType:
    def __init__(self, path_train: str, path_validate: str):
        self.path_train = path_train
        self.path_validate = path_validate
        self.model = tf.keras.Sequential()
        self.train_ds = None
        self.val_ds = None

    def load_image_data(self):
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.path_train,
            image_size=(img_width, img_height),
            batch_size=batch_size,
            class_names=["coca", "ps"]
        )

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.path_validate,
            image_size=(img_width, img_height),
            batch_size=batch_size,
            class_names=["coca", "ps"]
        )

    def add_sequencial(self, deep: int):
        self.model.add(tf.keras.layers.Rescaling(1./255))
        self.model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_width, img_height, 1)))
        for i in range(deep):
            self.model.add(tf.keras.layers.Conv2D(32, 3, activation='relu'))
            self.model.add(tf.keras.layers.MaxPool2D())
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(128, activation="relu"))
        self.model.add(tf.keras.layers.Dense(2, activation="softmax"))

    def train_model(self, name_model: str, save: bool):

        self.model.compile(
            optimizer=Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=20
        )

        if save:
            self.model.save(f"{name_model}.h5")

    def run_model(self, save: bool, deep: int, name_model: str):
        self.load_image_data()
        self.add_sequencial(deep=deep)

        self.train_model(save=save, name_model=name_model)


if __name__ == "__main__":
    model = DetectionLogoType(
        path_train="/home/bezerra/Desktop/Old_ubuntu/CNPq/DetectionLogotype/train_images",
        path_validate="/home/bezerra/Desktop/Old_ubuntu/CNPq/DetectionLogotype/teste_others"
    )
    model.run_model(save=True, deep=3, name_model="new_model_t")

