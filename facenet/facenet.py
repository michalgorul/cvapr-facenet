import os
from typing import Tuple

import numpy as np
from keras.models import load_model
from numpy import ndarray
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC

from facenet.utils import load_faces_from_dir


class Facenet:
    data_train_path = '../input/data/train/'
    data_val_path = '../input/data/val/'
    celebrity_faces_dataset_path = '../resources/5-celebrity-faces-dataset.npz'
    facenet_keras_path = '../resources/facenet_keras.h5'

    def __int__(self) -> None:
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None
        self.facenet_model = None
        self.emdTrainX = None
        self.emdTestX = None

    def setup(self):
        # self._load_train_test_datasets()
        self._load_the_facenet_dataset()
        self._load_facenet_model()
        self._convert_train_test_faces_into_embedding()

    def calculate_accuracy(self):
        print("Dataset: train=%d, test=%d" % (self.emdTrainX.shape[0], self.emdTestX.shape[0]))
        # normalize input vectors
        in_encoder = Normalizer()
        emdTrainX_norm = in_encoder.transform(self.emdTrainX)
        emdTestX_norm = in_encoder.transform(self.emdTestX)
        # label encode targets
        out_encoder = LabelEncoder()
        out_encoder.fit(self.trainY)
        trainy_enc = out_encoder.transform(self.trainY)
        testy_enc = out_encoder.transform(self.testY)
        # fit model
        model = SVC(kernel='linear', probability=True)
        model.fit(emdTrainX_norm, trainy_enc)
        # predict
        yhat_train = model.predict(emdTrainX_norm)
        yhat_test = model.predict(emdTestX_norm)
        # score
        score_train = accuracy_score(trainy_enc, yhat_train)
        score_test = accuracy_score(testy_enc, yhat_test)
        # summarize
        print('Accuracy: train=%.3f, test=%.3f' % (score_train * 100, score_test * 100))

    def _convert_train_test_faces_into_embedding(self):
        self.emdTrainX = self._convert_faces_into_embedding(self.trainX)
        self.emdTestX = self._convert_faces_into_embedding(self.testX)
        np.savez_compressed('5-celebrity-faces-embeddings.npz', self.emdTrainX, self.trainY,
                            self.emdTestX, self.testY)

    def _convert_faces_into_embedding(self, faces) -> ndarray:
        emd_faces = list()
        for face in faces:
            emd = self._get_embedding(self.facenet_model, face)
            emd_faces.append(emd)

        emd_faces = np.asarray(emd_faces)
        return emd_faces

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

    def _load_facenet_model(self) -> None:
        # load the model
        self.facenet_model = load_model(self.facenet_keras_path)
        # summarize input and output shape
        print('Model Loaded')
        print(self.facenet_model.inputs)
        print(self.facenet_model.outputs)

    def _load_the_facenet_dataset(self) -> None:
        data = np.load(self.celebrity_faces_dataset_path)
        self.trainX, self.trainY, self.testX, self.testY = data['arr_0'], data['arr_1'], data[
            'arr_2'], data['arr_3']
        print('Loaded: trainX.shape: ', self.trainX.shape, ' trainY.shape: ', self.trainY.shape,
              ' testX.shape: ', self.testX.shape, ' testY.shape: ', self.testY.shape)

    def _load_train_test_datasets(self) -> None:
        print('\nload train dataset')
        trainX, trainY = self.load_dataset(self.data_train_path)
        print('trainX.shape: ', trainX.shape, ' trainY.shape: ', trainY.shape)
        print('\nload test dataset')
        testX, testY = self.load_dataset(self.data_val_path)
        print('testX.shape: ', testX.shape, ' testY.shape: ', testY.shape)

        # save and compress the dataset for further use
        np.savez_compressed(self.celebrity_faces_dataset_path, trainX, trainY, testX, testY)

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
