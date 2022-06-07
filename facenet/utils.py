import os
from typing import Tuple, List

import cv2
from PIL import Image
from matplotlib import pyplot
from mtcnn import MTCNN
from numpy import ndarray, expand_dims, asarray


def save_file_to_validation_folder(face_name: str, file_name: str) -> None:
    img = cv2.imread('loadimage.jpeg')
    cv2.imwrite(f'../input/validation/{face_name}/{file_name}', img)


def show_image(img) -> None:
    pyplot.imshow(img, cmap='gray', interpolation='bicubic')
    pyplot.show()
    print(f"img.shape: {img.shape}")


def show_extracted_face(file_path: str) -> None:
    face = extract_face(file_path)
    pyplot.imshow(face)
    pyplot.show()
    print(f"img.shape: {face.shape}")


def get_embedding(model, face):
    # scale pixel values
    face = face.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    # transform face into one sample
    samples = expand_dims(face, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]


# extract a single face from a given photograph
def extract_face(filename: str, required_size: Tuple = (160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


# load images and extract faces for all images in a directory
def load_faces_from_dir(path_dir: str) -> List[ndarray]:
    faces = list()
    # enumerate files
    for filename in os.listdir(path_dir):
        # path
        path = path_dir + filename
        # get face
        face = extract_face(path)
        # store
        faces.append(face)
    return faces
