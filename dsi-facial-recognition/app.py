import numpy as np
import cv2
import pickle

#face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    #Region detectada
    for (x, y, w, h) in faces:
        # print (x, y, w, h)

        roi_gray = gray[y:y+h, x:x+w] #(ycord_start, y_cord_start + height)
        roi_color = frame[y:y+h, x:x+w] #(xcord_start, x_cord_start + width)

        id_, conf = recognizer.predict(roi_gray) 
        confidence = 100 - (conf*100 / 255)
        if confidence >= 10: #and conf <= 85:
            # print (id_)
            # print (labels[id_])
            cv2.putText(frame, labels[id_] + ": " + '{:3.2f}'.format(confidence), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
        
        if confidence > 100:
            print('ERROR, cnonf over 100% ({conf})'.format(conf=confidence))

        # img_item = "my-image.png"
        # cv2.imwrite(img_item, roi_gray)

        color = (255, 0, 0) #BGR
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h

        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    #Mostramos el frame resultante
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break;

cap.release()
cv2.destroyAllWindows()
