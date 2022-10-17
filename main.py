import cv2
cap=cv2.VideoCapture(0)
model=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
def face_blur(photo):
    face=model.detectMultiScale(photo)

    if len(face) != 0:

        for nface in face:

            crop_photo= cv2.blur(photo[nface[1]:nface[1]+nface[3],nface[0]:nface[0]+nface[2]], (20,20))

            photo[nface[1]:nface[1]+nface[3],nface[0]:nface[0]+nface[2]]= crop_photo

        return photo

    else: 
        return photo


while True:
    ret, photo = cap.read()
    cv2.imshow("hello", face_blur(photo)) 
    if cv2.waitKey(10) == 13:
        break

cv2.destroyAllWindows()
cap.release()