import os
from PIL import Image
import numpy as np
import cv2
import pickle
import logging
import coloredlogs


logger = logging.getLogger(__name__)
# cap = None

def initLogger(level='DEBUG'):
    # fmt = '%(asctime)s - %(levelname)s: %(message)s'
    fmt = '%(asctime)s - %(message)s'

    coloredlogs.install(fmt=fmt, datefmt='%d/%m/%Y %H:%M:%S', level=level)
    # logger.info('Logger is active')


def createTrainModel():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, "images")

    faceCascades = []
    for model in ['default', 'alt', 'alt2', 'alt_tree']:
        p = 'cascades/data/haarcascade_frontalface_{model}.xml'
        faceCascades.append(cv2.CascadeClassifier(p.format(model=model)))

    #face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
    face_cascade = cv2.CascadeClassifier()
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    current_id = 0
    label_ids ={}
    y_labels = []
    x_train = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                idx = 0

                for face_cascade in faceCascades:

                    path = os.path.join(root, file)
                    label = os.path.basename(os.path.dirname(path))

                    if not label in label_ids:
                        label_ids[label] = current_id
                        current_id += 1

                    id_ = label_ids[label]
                    #print (label_ids)

                    #y_labels.append(label) #save the labels
                    #x_train.append(path) #save the images

                    # grayscaleImg = Image.open(path).convert("L") #grayscale
                    grayscaleImg = cv2.imread(path, 0)

                    #size = (500, 500)
                    #grayscaleImg = grayscaleImg.resize(size, Image.ANTIALIAS)

                    #image_array = np.array(final_image, "uint8")
                    # image_array = np.array(grayscaleImg, "uint8")
                    #print (image_array)

                    #obtain the region of interest
                    faces = face_cascade.detectMultiScale(grayscaleImg, scaleFactor=1.3, minNeighbors=5)
                    #print (faces)

                    for (x, y, w, h) in faces:
                        # print ("Cara detectada para: " + label[0].upper() + label[1:])
                        croppedROI = grayscaleImg[y:y+h, x:x+w].copy()

                        # cv2.imshow('frame', croppedROI)

                        croppedROI = cv2.resize(croppedROI, (160, 160))
                        
                        x_train.append(croppedROI)
                        y_labels.append(id_)
                    
                    if len(faces) > 0:
                        logger.debug(path.split('\\')[-1] + '\t✓')
                        break
                    else:
                        if idx == len(faceCascades)-1:
                            logger.error(path.split('\\')[-1] + '\t⛌ [{}]'.format(idx))
                        else:
                            logger.warning(path.split('\\')[-1] + '\t⛌ [{}]'.format(idx))
                    
                    idx += 1

    #print (y_labels)
    #print (x_train)

    with open("labels.pickle", 'wb') as f:
        pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("trainer.yml")


if __name__ == "__main__":
    initLogger()
    createTrainModel()

