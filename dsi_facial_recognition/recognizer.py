import numpy as np
import cv2
import pickle
import logging
import coloredlogs
import os
import json
import memcache
from multiprocessing import Process, Queue

logger = logging.getLogger(__name__)
text = ''
colors = { 
    'Ainara': (255, 0, 0),
    'Juan': (0, 255, 0),
    'Mario': (0, 76, 255),
    'Belen': (221, 255, 0),
    'Miguel': (246, 0, 255),
    'Kike': (0, 255, 144)
}

def initLogger(level='DEBUG'):
    # fmt = '%(asctime)s - %(levelname)s: %(message)s'
    fmt = '%(asctime)s - %(message)s'

    coloredlogs.install(fmt=fmt, datefmt='%d/%m/%Y %H:%M:%S', level=level)
    # logger.info('Logger is active')


def recognise(frame, faceCascades, recognizer, labels):
    idx = 0
    recognizedId = None

    for face_cascade in faceCascades:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5)

        #Region detectada
        for (x, y, w, h) in faces:
            # print (x, y, w, h)

            croppedROI = gray[y:y+h, x:x+w].copy()
            croppedROI = cv2.resize(croppedROI, (160, 160))

            # roi_gray = gray[y:y+h, x:x+w] #(ycord_start, y_cord_start + height)
            # roi_color = frame[y:y+h, x:x+w] #(xcord_start, x_cord_start + width)

            id_, conf = recognizer.predict(croppedROI)
            confidence = 100 - (conf*100 / 255)
            if confidence >= 60:  # and conf <= 85:
                recognizedId = id_
                cv2.putText(frame, labels[id_] + ": " + '{:3.2f}'.format(
                    confidence), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[labels[id_]], 1, cv2.LINE_AA)
                # cv2.putText(frame, text, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

            if confidence > 100:
                print('ERROR, cnonf over 100% ({conf})'.format(
                    conf=confidence))

            # img_item = "my-image.png"
            # cv2.imwrite(img_item, roi_gray)

            color = (255, 0, 0)  # BGR
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h

            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), 
                colors[labels[id_]], stroke)

        if len(faces) > 0:
            # logger.debug(path.split('\\')[-1] + '\t✓')
            break
        else:
            if idx == len(faceCascades)-1:
                logger.warning('\t⛌ [{}]'.format(idx))
        idx += 1

    return recognizedId, frame


class Recognizer:

    def __init__(self):
        self.text = 'Placeholder'
        self.speaker = None
        self.client = memcache.Client([('127.0.0.1', 11211)])
        initLogger()
        

    def launch(self):
        basedir = os.path.abspath(os.path.dirname(__file__))        

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(os.path.join(basedir, "trainer.yml"))

        labels = {"person_name": 1}
        with open(os.path.join(basedir, "labels.pickle"), 'rb') as f:
            og_labels = pickle.load(f)
            labels = {v:k for k,v in og_labels.items()}

        cap = cv2.VideoCapture(0)

        faceCascades = []
        for model in ['default', 'alt', 'alt2', 'alt_tree']:
            p = os.path.join(basedir, 'cascades/data/haarcascade_frontalface_{model}.xml')
            faceCascades.append(cv2.CascadeClassifier(p.format(model=model)))

        while(True):

            ret, frame = cap.read()
            id_, frame = recognise(frame, faceCascades, recognizer, labels)

            textColor = (255, 255, 255)
            self.speaker = self.client.get("speaker")
            if self.speaker in colors:
                textColor = colors[self.speaker]

            textLabel = '<...>'
            self.text = self.client.get("speech")
            if self.text is not None:
                # print(self.text)
                textLabel = self.text
                
            w = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]

            offset = int((frame.shape[1] - w[0]) / 2)

            # if w[0] > (frame.shape[1] - 40):
            #     textLabel = textLabel[:20]

            bottomLeft = (offset, frame.shape[0] - 30)
            topRight = (offset + w[0], frame.shape[0] - 30 - w[1])

            cv2.rectangle(frame, (bottomLeft[0] - 10, bottomLeft[1] + 10),
                          (topRight[0] + 10, topRight[1] - 10), 
                          (30, 30, 30), cv2.FILLED)


            cv2.putText(frame, textLabel, (offset,frame.shape[0] - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, textColor, 1, cv2.LINE_AA)

            #Mostramos el frame resultante
            cv2.imshow('frame', frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    initLogger()

    a = Recognizer()
    a.launch()
