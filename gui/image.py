from os import mkdir
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QFileDialog, QTextEdit
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from distutils.dir_util import copy_tree
from facenet.facenet import Facenet
import shutil
import sys

from facenet.utils import get_index

IMAGE_WIDTH = 720
IMAGE_HEIGHT = 471
TRAIN_DIRECTORY_PATH = r"../input/train/"
VALIDATION_DIRECTORY_PATH = r"../input/validation/"
OUTPUT_FILE_DIRECTORY_PATH = r"../output"


class UI(QMainWindow):
    def __init__(self):
        self.fname = None
        self.outputFileName = None
        super(UI, self).__init__()

        # Load the ui file
        uic.loadUi("../gui/image.ui", self)

        # Define our widgets
        self.button = self.findChild(QPushButton, "pushButton")
        self.button_2 = self.findChild(QPushButton, "pushButton_2")
        self.button_3 = self.findChild(QPushButton, "pushButton_3")
        self.label = self.findChild(QLabel, "label")
        self.label_3 = self.findChild(QLabel, "label_3")
        self.textEdit = self.findChild(QTextEdit, "textEdit")

        # Click The Dropdown Box
        self.button.clicked.connect(self.loadImage)
        self.button_2.clicked.connect(self.processImage)
        self.button_3.clicked.connect(self.loadDirectory)

        # Show The App
        self.show()

    def loadImage(self):
        self.fname = QFileDialog.getOpenFileName(self, "Open File", "c:\\gui\\images",
                                                 "All Files (*);;PNG Files (*.png);;Jpg Files (*.jpg)")
        # Open The Image
        if self.fname:
            self.pixmap = QPixmap(self.fname[0])
            self.pixmap = self.pixmap.scaled(IMAGE_WIDTH, IMAGE_HEIGHT)
            # Add Pic to label
            self.label.setPixmap(self.pixmap)

    def processImage(self):
        index = get_index(self.fname[0])
        print(index)
        facenet = Facenet(True)
        facenet.make_prediction(index)

    def copy_image_to_validation_folder(self):
        shutil.copy(self.fname[0], VALIDATION_DIRECTORY_PATH)

    def loadDirectory(self):
        dir_ = QFileDialog.getExistingDirectory(None, 'Select a folder:', 'C:\\', QFileDialog.ShowDirsOnly)

        print(dir_)
        print(self.textEdit.toPlainText())

        path = TRAIN_DIRECTORY_PATH + self.textEdit.toPlainText()
        print(path)

        mkdir(path)
        copy_tree(dir_, path)
       #Facenet(True)
