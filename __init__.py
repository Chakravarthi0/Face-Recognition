#import opencv and numpy modules
import cv2
import numpy as np

#load cascade classifier training file for haarcascade
detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#capturing video from the camera
cap = cv2.VideoCapture(0)

while True:
    #reading the video frame by frame
    res,img = cap.read()
    
    #convert captured image into gray image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #detecting faces in the captured image
    faces = detect.detectMultiScale(gray,1.3,5)
    
    #draw a rectangle around the detected face(s)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
    
    #display the output    
    cv2.imshow("img",img)
    
    #loop stops when the user presses esc key
    k = cv2.waitKey(3) & 0xff
    if k == 27:
        break

#close the camera and destroy the windows
cv2.release()
cv2.destroyAllWindows()
