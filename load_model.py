import pathlib
import tensorflow as tf
import numpy as np

batch_size = 32
img_height = 180
img_width = 180


class ModelLoad:

    @staticmethod
    def parse_image(path_test: str, name_model: str):
        model = tf.keras.models.load_model(f'{name_model}.h5')

        val_ds = tf.keras.utils.image_dataset_from_directory(
            pathlib.Path(path_test),
            image_size=(img_height, img_width),
            batch_size=batch_size,
        )

        return model.predict(val_ds)


if __name__ == "__main__":
    predicts = ModelLoad.parse_image(
        path_test="/home/bezerra/Desktop/Old_ubuntu/CNPq/DetectionLogotype/niv",
        name_model="/home/bezerra/Desktop/Old_ubuntu/CNPq/DetectionLogotype/new_model_t"
    )
    for predict in np.argmax(predicts, axis=1):
        if predict == 1:
            print("Pepsi")
        else:
            print("Coca")
