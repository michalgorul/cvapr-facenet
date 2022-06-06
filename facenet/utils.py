import os
from typing import Tuple, List

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from mtcnn import MTCNN
from numpy import ndarray


def show_image(img) -> None:
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.show()
    print(f"img.shape: {img.shape}")


# extract a single face from a given photograph
def extract_face(filename: str, required_size: Tuple = (160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # deal with negative pixel index
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array


def load_faces_from_dir(path_dir: str) -> List[ndarray]:
    faces = list()
    # enumerate files
    for filename in os.listdir(path_dir):
        path = path_dir + filename
        face = extract_face(path)
        faces.append(face)
    return faces


def load_dataset(path_dir: str) -> tuple[ndarray, ndarray]:
    # list for faces and labels
    X, y = list(), list()
    for subdir in os.listdir(path_dir):
        path = path_dir + subdir + '/'
        faces = load_faces_from_dir(path)
        labels = [subdir for i in range(len(faces))]
        print("loaded %d sample for class: %s" % (len(faces),subdir) ) # print progress
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)
