import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

cap = cv2.VideoCapture(0)

while(True):

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    #Region detectada
    for (x, y, w, h) in faces:
        print (x, y, w, h)

        roi_gray = gray[y:y+h, x:x+w] #(ycord_start, y_cord_start + height)
        roi_color = frame[y:y+h, x:x+w] #(xcord_start, x_cord_start + width)

        id_, conf = recognizer.predict(roi_gray) 
        if conf >= 45 and conf <= 85:
            print (id_)

        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)

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
