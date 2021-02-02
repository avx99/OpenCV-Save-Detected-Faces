import cv2
import json
import face_recognition
from datetime import datetime 


def compareImages(img1,img2):    
    height1, width1, _ = img1.shape
    height2, width2, _ = img2.shape
    face_location1 = (0, width1, height1, 0)
    face_location2 = (0, width2, height2, 0)
    encoding1 = face_recognition.face_encodings(img1,known_face_locations=[face_location1])[0]
    encoding2 = face_recognition.face_encodings(img2,known_face_locations=[face_location2])[0]
    
    return face_recognition.compare_faces([encoding1], encoding2)



cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('src/cascades/data/haarcascade_frontalface_alt2.xml')
d = {}
i = 0
while(True):
    ret,frame = cap.read()
    grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayFrame,scaleFactor=1.3,minNeighbors=5)
    for (x,y,w,h) in  faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(10,255,255),3)
        # cv2.putText(frame, 'Machi Oussama 89.482%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10,255,255), 2)
        im = frame[y:y+h,x:x+w]

        print("************************** "+str(i))
        if len(d) == 0: 
            x = {"image" : im,"date" : datetime.now()}
            d[i] = x
            i = i + 1
        if compareImages(d[len(d)-1]["image"],im) == False:
            x = {"image" : im,"date" : datetime.now()}
            d[i] = x
            i = i + 1

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    cv2.imshow('Frame',frame)
cap.release()
cv2.imshow('rr',im)
cv2.waitKey()
cv2.destroyAllWindows()



