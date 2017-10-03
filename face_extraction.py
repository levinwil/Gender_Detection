import numpy as np
import cv2
from gender_detect_MLP import *

def model_predict(model, img):
    prediction = model.predict()
    if prediction == 1:
        return "MALE"
    else:
        return "FEMALE"


    return str(prediction)

def extract_faces():
    model = Image_MLP(model_path = './saved_models/gender.h5')
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_detected = False
    while(True):
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            image = frame[max(0, y - 40): min(frame.shape[0], y+h + 40), max(0, x - 40):min(frame.shape[1], x+w + 40)]
            cv2.imwrite('data/test/test/face.png', image)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            face_detected = True
        if face_detected:
            gender = model_predict(model, image)
            cv2.putText(image, gender, (image.shape[1]/5, image.shape[0]/5), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.imshow('img', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    extract_faces()
