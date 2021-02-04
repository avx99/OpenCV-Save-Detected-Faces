import cv2
import json
import face_recognition
from datetime import datetime 
import json
import numpy as np
from json import JSONEncoder
import os
import shutil

Known_distance = 30 #centimeter
Known_width =14.3 
GREEN = (0,255,0)
RED = (0,0,255)
WHITE = (255,255,255)
fonts = cv2.FONT_HERSHEY_COMPLEX

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def FocalLength(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image* measured_distance)/ real_width
    return focal_length


def Distance_finder (Focal_Length, real_face_width, face_width_in_frame):
    distance = (real_face_width * Focal_Length)/face_width_in_frame
    return distance

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
d={}
i = 0
Focal_length_found = FocalLength(Known_distance, Known_width, 5)
shutil.rmtree('images')
os.remove("date.txt")
os.remove("distance.txt")
os.mkdir("images")
while(True):
    ret,frame = cap.read()
    grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayFrame,scaleFactor=1.3,minNeighbors=5)
    for (x,y,w,h) in  faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(10,255,255),3)
        im = frame[y:y+h,x:x+w]
        cv2.imwrite('images/'+str(i)+'.png',im)
        print("************************** "+str(i))     
        distance = Distance_finder(Focal_length_found, Known_width,w)
        cv2.putText(frame, f"Distance = {distance}", (50,50), fonts,1, (WHITE),2)
        x = {"date" : datetime.now()}
        d[i] = x
        i = i + 1
        f = open("date.txt", "a")
        f.write(str(datetime.now())+"\n")
        f.close()
        f = open("distance.txt", "a")
        f.write(str(distance)+"\n")
        f.close()
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    cv2.imshow('Frame',frame)
cap.release()
cv2.imshow('last frame',im)
cv2.waitKey()
cv2.destroyAllWindows()



# with open("data.json", "w") as fp:
#     json.dump(d, fp, indent=4)
