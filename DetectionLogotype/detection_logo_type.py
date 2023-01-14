import tensorflow as tf



class DetectionLogoType:
    def __init__(self, path_train: str, path_test: str):
        self.path_train = path_train
        self.path_test = path_test
        self.model = None
        self.train_ds = None
        self.test_ds = None

    def load_image_data(self):
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.train_ds,
            image_size=(180, 180),
            batch_size=35,
            class_names=["coca", "ps"]
        )

        self.test_ds = tf.keras.utils.image_dataset_from_directory(
            self.test_ds,
            image_size=(180,180),
            batch_size=35,
            class_names=["coca", "ps"]
        )

        