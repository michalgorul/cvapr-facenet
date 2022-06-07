import sys

from PyQt5.QtWidgets import QApplication

from facenet.facenet import Facenet
import logging
import tensorflow

from gui.image import UI

tensorflow.get_logger().setLevel(logging.ERROR)

if __name__ == '__main__':
    # Initialize The App
    app = QApplication(sys.argv)
    UIWindow = UI()
    app.exec_()

    facenet = Facenet(True)
    facenet.make_prediction()
