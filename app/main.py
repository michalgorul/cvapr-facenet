from facenet.facenet import Facenet
import logging
import tensorflow

tensorflow.get_logger().setLevel(logging.ERROR)

if __name__ == '__main__':
    facenet = Facenet()
    facenet.calculate_accuracy()
    facenet.make_prediction()
