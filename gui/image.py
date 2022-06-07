from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QFileDialog
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
import sys

IMAGE_WIDTH = 720
IMAGE_HEIGHT = 471


class UI(QMainWindow):
    def __init__(self):
        self.fname = None
        super(UI, self).__init__()

        # Load the ui file
        uic.loadUi("image.ui", self)

        # Define our widgets
        self.button = self.findChild(QPushButton, "pushButton")
        self.button_2 = self.findChild(QPushButton, "pushButton_2")
        self.label = self.findChild(QLabel, "label")

        # Click The Dropdown Box
        self.button.clicked.connect(self.loadImage)
        self.button_2.clicked.connect(self.processImage)

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
        print(self.fname)
        self.label_2.setText('Ryj kolegi')


# Initialize The App
app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
