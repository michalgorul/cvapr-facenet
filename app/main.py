from facenet.facenet import Facenet

if __name__ == '__main__':
    facenet = Facenet()
    facenet.setup()
    facenet.calculate_accuracy()
