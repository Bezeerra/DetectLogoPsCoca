import cv2
from load_model import ModelLoad


def confirm_image(frame):
    cv2.imwrite("/home/bezerra/Desktop/DetectionLogotype/teste_others/connect_webcam/image/teste01.jpg", frame)
    comparison = ModelLoad.parse_image(path="/home/bezerra/Desktop/DetectionLogotype/connect_webcam/image", name_model="modelo")
    print(comparison)


def main():
    webcam = cv2.VideoCapture(0)
    if webcam.isOpened():
        print("Connect to webcam")
        validate, frame = webcam.read()
        while validate:
            validate, frame = webcam.read()
            cv2.imshow("Video Webcam", frame)
            key = cv2.waitKey(5)
            if key == 27:  # on click ESC
                confirm_image(frame)
    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

