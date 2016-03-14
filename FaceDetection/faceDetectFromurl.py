import numpy as np
import cv2
import os
import sys
from sys import stdin
import os.path
import re
import urllib2
import base64
import math
import copy

sys.path.append('classes')
from myFaceBdd import myFaceBdd 
from webcamGrabber import webcamGrabber 


bddName = 'myFaceDb'

face_cascade = cv2.CascadeClassifier('data\haarcascades\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data\haarcascades\haarcascade_eye.xml')
fullbody = cv2.CascadeClassifier('data\haarcascades\haarcascade_fullbody.xml')
pedestrian = cv2.CascadeClassifier('data\hogcacades\hogcascade_pedestrian.xml')
car_cascade = cv2.CascadeClassifier('data\haarcascades\cars3.xml')


myBdd = myFaceBdd( "./data/MyFaceBdd", bddName)
##### Coef
coef = 2
font = cv2.FONT_HERSHEY_SIMPLEX

# mjpg-streamer URL
user= "user"
pwd = "password"
url = "http://<ip>:<port>/?action=stream"

min_area = 500
blurCoef = 3
rotation = -90
font = cv2.FONT_HERSHEY_SIMPLEX

def detectAndDraw(cascade, image, color):
    img = image
    newx,newy = img.shape[1]/coef,img.shape[0]/coef
    
    newimage = cv2.resize(img,(newx,newy))
    
    gray = cv2.cvtColor(newimage, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(coef*x,coef*y),(coef*(x+w),coef*(y+h)),color,2)
#         imgsized = cv2.resize(img[coef*y:coef*(y+h), coef*x:coef*(x+w)],(300,300))
#         libelle, conf = myBdd.whoisIt(imgsized)
#         cv2.imshow('mini',imgsized)
#         cv2.putText(img,'['+libelle+'] ('+str(conf)+')',(coef*x,coef*y), font, 1,(255,255,255),2)


    return img


def diffAndDraw(lastFrame, currentFrame, color):
    frame = copy.copy(currentFrame)
    firstFrame = cv2.cvtColor(lastFrame, cv2.COLOR_BGR2GRAY)
    firstFrame = cv2.GaussianBlur(firstFrame, (blurCoef, blurCoef), 0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blurCoef, blurCoef), 0)

    
    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    ret, thresh = cv2.threshold(frameDelta, 3, 255, cv2.THRESH_BINARY)
    
    cv2.imshow('Frame Delta',frameDelta)
    
    cv2.imshow('Threshol 1',thresh)

    
    
    
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
 
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
            continue
 
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"
    
    return frame


def rotate_about_center(src, angle, scale=1.):
    if (angle == 0):
        return src
    
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)


cam = webcamGrabber(url,user, pwd)
cam.connect()

#cam = cv2.VideoCapture(0)



prev = ''

while True:
    
    #ret, img = cam.read()
    img = cam.read()
    img2 = rotate_about_center(img, rotation)

    if (prev == ''):
        prev = img2
    
    
    img_out3 = diffAndDraw(prev, img2, (255,255, 0))
    prev = img2
    
    
    img2 = detectAndDraw(face_cascade, img_out3, (0,0, 255))
    #img2 = detectAndDraw(car_cascade, img2, (0,0, 255))
    #cv2.imshow('r',img_out1)
    #cv2.imshow('w',img_out2)
    cv2.imshow('x',img_out3)
    if cv2.waitKey(1) ==27:
        break 
            
            
#cam.release()
myBdd.saveMyFaceBdd()
cv2.destroyAllWindows()


