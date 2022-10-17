import cv2
import numpy as np
def resize(img,new_width=500):
    height,width,_ = img.shape
    ratio = height/width
    new_height = int(ratio*new_width)
    return cv2.resize(img,(new_width,new_height))

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

cap = cv2.VideoCapture(0)

while True:
    _,frame = cap.read()
    frame = resize(frame)
    detections = face_cascade.detectMultiScale(frame,scaleFactor=1.9,minNeighbors=6)

    for face in detections:
        x,y,w,h = face

        frame[y:y+h,x:x+w] = cv2.GaussianBlur(frame[y:y+h,x:x+w],(25,17),cv2.BORDER_DEFAULT)
        
        #comment below line if coloured border is not needed
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow("output",frame)

        if cv2.waitKey(1) & 0xFF == ord('q') == 27:
            break

cap.release()
cv2.destroyAllWindows()