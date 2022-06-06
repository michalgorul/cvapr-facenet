import os

import numpy as np
from numpy import ndarray

from facenet.utils import load_faces_from_dir


class Facenet:
    data_train_path = '../input/data/train/'
    data_val_path = '../input/data/val/'

    def __int__(self):
        self._load_train_test_datasets()

    def _load_train_test_datasets(self):
        print('\nload train dataset')
        trainX, trainy = self.load_dataset(self.data_train_path)
        print(trainX.shape, trainy.shape)
        print('\nload test dataset')
        testX, testy = self.load_dataset(self.data_val_path)
        print(testX.shape, testy.shape)

        # save and compress the dataset for further use
        np.savez_compressed('5-celebrity-faces-dataset.npz', trainX, trainy, testX, testy)

    def load_dataset(self, path_dir: str) -> tuple[ndarray, ndarray]:
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
