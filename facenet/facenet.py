import os
from os.path import isdir
from random import choice
from typing import Tuple

import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt, pyplot
from numpy import ndarray, load, expand_dims, savez_compressed, asarray
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC

from facenet.utils import load_faces_from_dir, get_embedding


class Facenet:
    data_train_path = '../input/data/train/'
    data_val_path = '../input/data/val/'
    celebrity_faces_dataset_path = '../resources/5-celebrity-faces-dataset.npz'
    celebrity_faces_embeddings_path = '../resources/5-celebrity-faces-embeddings.npz'
    facenet_keras_path = '../resources/facenet_keras.h5'

    def __init__(self) -> None:
        self.model = None
        self._setup()

    def _setup(self):
        # self._load_train_test_datasets()
        self._load_facenet_model()
        self._convert_train_test_faces_into_embedding()

    def make_prediction(self):
        # load faces
        data = load(self.celebrity_faces_dataset_path)
        testX_faces = data['arr_2']
        # load face embeddings
        data = load(self.celebrity_faces_embeddings_path)
        trainX, trainY, testX, testY = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        # normalize input vectors
        in_encoder = Normalizer(norm='l2')
        trainX = in_encoder.transform(trainX)
        testX = in_encoder.transform(testX)
        # label encode targets
        out_encoder = LabelEncoder()
        out_encoder.fit(trainY)
        trainY = out_encoder.transform(trainY)
        testY = out_encoder.transform(testY)
        # fit model
        model = SVC(kernel='linear', probability=True)
        model.fit(trainX, trainY)
        # test model on a random example from the test dataset


        selection = choice([i for i in range(testX.shape[0])])
        random_face_pixels = testX_faces[selection]
        random_face_emb = testX[selection]
        random_face_class = testY[selection]
        random_face_name = out_encoder.inverse_transform([random_face_class])
        # prediction for the face
        samples = expand_dims(random_face_emb, axis=0)
        yhat_class = model.predict(samples)
        yhat_prob = model.predict_proba(samples)
        # get name
        class_index = yhat_class[0]
        class_probability = yhat_prob[0, class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)


        print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
        print('Expected: %s' % random_face_name[0])
        # plot for fun
        pyplot.imshow(random_face_pixels)
        title = '%s (%.3f)' % (predict_names[0], class_probability)
        pyplot.title(title)
        pyplot.show()

    def calculate_accuracy(self):
        # load dataset
        data = load(self.celebrity_faces_embeddings_path)
        trainX, trainY, testX, testY = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        print('Loaded: ', trainX.shape, trainY.shape, testX.shape, testY.shape)
        print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
        # normalize input vectors
        in_encoder = Normalizer(norm='l2')
        print(trainX.shape)
        trainX = in_encoder.transform(trainX)
        testX = in_encoder.transform(testX)
        # label encode targets
        out_encoder = LabelEncoder()
        out_encoder.fit(trainY)
        trainY = out_encoder.transform(trainY)
        testY = out_encoder.transform(testY)
        # fit model
        self.model = SVC(kernel='linear', probability=True)
        self.model.fit(trainX, trainY)
        # predict
        yhat_train = self.model.predict(trainX)
        yhat_test = self.model.predict(testX)
        # score
        score_train = accuracy_score(trainY, yhat_train)
        score_test = accuracy_score(testY, yhat_test)
        # summarize
        print('Accuracy: train=%.3f, test=%.3f' % (score_train * 100, score_test * 100))

    # calculate a face embedding for each face in the dataset using facenet
    def _convert_train_test_faces_into_embedding(self):
        # load the face dataset
        data = load(self.celebrity_faces_dataset_path)
        trainX, trainY, testX, testY = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        print('Loaded: ', trainX.shape, trainY.shape, testX.shape, testY.shape)
        # convert each face in the train set to an embedding
        newTrainX = list()
        for face in trainX:
            embedding = get_embedding(self.model, face)
            newTrainX.append(embedding)
        newTrainX = asarray(newTrainX)
        print(newTrainX.shape)
        # convert each face in the test set to an embedding
        newTestX = list()
        for face in testX:
            embedding = get_embedding(self.model, face)
            newTestX.append(embedding)
        newTestX = asarray(newTestX)
        print(newTestX.shape)
        # save arrays to one file in compressed format
        savez_compressed(self.celebrity_faces_embeddings_path, newTrainX, trainY,
                         newTestX, testY)
        print('Saved embeddings: ', newTrainX.shape, trainY.shape, newTestX.shape, testY.shape)

    def _load_facenet_model(self) -> None:
        # load the model
        self.model = load_model(self.facenet_keras_path)
        # summarize input and output shape
        print('Model Loaded')
        print(self.model.inputs)
        print(self.model.outputs)

    def _load_train_test_datasets(self) -> None:
        # load train dataset
        trainX, trainy = self._load_dataset(self.data_train_path)
        print(trainX.shape, trainy.shape)
        # load test dataset
        testX, testy = self._load_dataset(self.data_val_path)
        # save arrays to one file in compressed format
        savez_compressed(self.celebrity_faces_dataset_path, trainX, trainy, testX, testy)

    # load a dataset that contains one subdir for each class that in turn contains images
    def _load_dataset(self, directory: str):
        X, y = list(), list()
        # enumerate folders, on per class
        for subdir in os.listdir(directory):
            # path
            path = directory + subdir + '/'
            # skip any files that might be in the dir
            if not isdir(path):
                continue
            # load all faces in the subdirectory
            faces = load_faces_from_dir(path)
            # create labels
            labels = [subdir for _ in range(len(faces))]
            # summarize progress
            print('>loaded %d examples for class: %s' % (len(faces), subdir))
            # store
            X.extend(faces)
            y.extend(labels)
        return asarray(X), asarray(y)
