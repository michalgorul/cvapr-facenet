import cv2 # opencv
from tensorflow.python.keras.models import load_model

from facenet.facenet import Facenet
from facenet.utils import show_image, show_extracted_face

# TODO TF 2.2, Keras 2.3 or 2.4, Python 3.6.
if __name__ == '__main__':
    # img = cv2.imread('../input/data/train/ben_afflek/httpcsvkmeuaeccjpg.jpg')
    # show_image(img)
    # show_extracted_face('../input/data/train/ben_afflek/httpcsvkmeuaeccjpg.jpg')
    facenet = Facenet()
    facenet.setup()
