import cv2 # opencv

from facenet.utils import show_image

if __name__ == '__main__':
    img = cv2.imread('../input/data/train/ben_afflek/httpcsvkmeuaeccjpg.jpg')
    show_image(img);
