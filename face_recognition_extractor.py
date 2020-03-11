# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 09:36:21 2019

@author: Harry180
"""

# Importing Libraries
import cv2
import numpy as np

# Importing a frontal face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to extract and return cropped faces
def face_extractor(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    # conversion to gray
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)     # Returns 'value' for faces??

    if faces is():                                           # IF NO FACE DETECTED
        return None

    for (x,y,w,h) in faces:                                  # Cropping of image
        cropped_face = img[y:y+h,x:x+w]

    return cropped_face

cap = cv2.VideoCapture(0)
count = 0               # count for no. of cropped faces

# Loop to create and save samples of cropped faces
while True:
    ret, frame = cap.read()

    if face_extractor is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200,200))    # Getting cropped face of 'frame' from extractor and resizing it

        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        file_name_path = 'D:/FACE_RECOG/Samples/user' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        cv2.putText(face, str(count),(50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper',face)

    else:
        print("Face Not Found :(")
        pass

    if cv2.waitKey(1) == 13 or count == 150:        # Program breaks on pressing "ENTER" key
        break

cap.release()
cv2.destroyAllWindows()
print('Collecting Samples Complete')
