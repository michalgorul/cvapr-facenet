import cv2 # opencv

from facenet.facenet import Facenet
from facenet.utils import show_image, show_extracted_face

if __name__ == '__main__':
    img = cv2.imread('../input/data/train/ben_afflek/httpcsvkmeuaeccjpg.jpg')
    show_image(img)
    show_extracted_face('../input/data/train/ben_afflek/httpcsvkmeuaeccjpg.jpg')

    facenet = Facenet()
