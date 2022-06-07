import sys

from PyQt5.QtWidgets import QApplication

from facenet.facenet import Facenet
import logging
import tensorflow

from gui.image import UI

tensorflow.get_logger().setLevel(logging.ERROR)

if __name__ == '__main__':
    facenet = Facenet(True)
    facenet.calculate_accuracy()
    facenet.make_prediction()
    #
    # # Initialize The App
    # app = QApplication(sys.argv)
    # UIWindow = UI()
    # app.exec_()
