import os
from typing import Tuple

import numpy as np
from numpy import ndarray
from keras.models import load_model

from facenet.utils import load_faces_from_dir


class Facenet:
    data_train_path = '../input/data/train/'
    data_val_path = '../input/data/val/'
    celebrity_faces_dataset_path = '../resources/5-celebrity-faces-dataset.npz'

    def __int__(self) -> None:
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None
        self.facenet_model = None

    def setup(self):
        # self._load_train_test_datasets()
        # self._load_the_facenet_dataset()
        self._load_facenet_model()

    def _get_embedding(self, model, face):
        # scale pixel values
        face = face.astype('float32')
        # standardization
        mean, std = face.mean(), face.std()
        face = (face - mean) / std
        # transfer face into one sample (3 dimension to 4 dimension)
        sample = np.expand_dims(face, axis=0)
        # make prediction to get embedding
        yhat = model.predict(sample)
        return yhat[0]

    def _load_the_facenet_dataset(self) -> None:
        data = np.load(self.celebrity_faces_dataset_path)
        self.trainX, self.trainY, self.testX, self.testY = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        print('Loaded: ', self.trainX.shape, self.trainY.shape, self.testX.shape, self.testY.shape)

    def _load_facenet_model(self) -> None:
        # load the model
        self.facenet_model = load_model('facenet_keras.h5')
        # summarize input and output shape
        print('Model Loaded')
        print(self.facenet_model.inputs)
        print(self.facenet_model.outputs)

    def _load_train_test_datasets(self) -> None:
        print('\nload train dataset')
        trainX, trainy = self.load_dataset(self.data_train_path)
        print(trainX.shape, trainy.shape)
        print('\nload test dataset')
        testX, testy = self.load_dataset(self.data_val_path)
        print(testX.shape, testy.shape)

        # save and compress the dataset for further use
        np.savez_compressed(self.celebrity_faces_dataset_path, trainX, trainy, testX, testy)

    def load_dataset(self, path_dir: str) -> Tuple[ndarray, ndarray]:
        # list for faces and labels
        X, y = list(), list()
        for subdir in os.listdir(path_dir):
            path = path_dir + subdir + '/'
            faces = load_faces_from_dir(path)
            labels = [subdir for i in range(len(faces))]
            print("loaded %d sample for class: %s" % (len(faces), subdir))  # print progress
            X.extend(faces)
            y.extend(labels)
        return np.asarray(X), np.asarray(y)
