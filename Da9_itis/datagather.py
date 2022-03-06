from turtle import width
from pynput.mouse import Listener
import logging
import sys
import time
import cv2 as cv
import os
import numpy as np
import dlib


def image_resize(image, width = None, height = None, inter = cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def on_click(X, Y, button, pressed):
    if pressed:
        # logging.info('Mouse clicked at ({0}, {1}) with {2}'.format(x, y, button))
        # print('Mouse clicked at ({0}, {1}) with {2}'.format(x, y, button))
        cam_port = 0
        cam = cv.VideoCapture(cam_port)

        j=0
        # capital x and y are click position
        while(j<2):
            result, image = cam.read()
            # image = cv.resize(image,(240,320))
        	# saving image in local storage
            img = np.zeros((480,640,3), np.uint8)
            boxes = cascade.detectMultiScale(image, 1.3, 10)
            if len(boxes)==2:
                for box in boxes:
                    x, y, w, h = box
                    img[y:y + h, x:x + w] = image[y:y + h, x:x + w]
                dets = detector(image, 1)
                for k, d in enumerate(dets):
                    shape=predictor(image, d)
                    if not shape.part(1):
                        print('bad bud')
                        pass
                    else:
                        for i in range(68):
                            img=cv.circle(img, (shape.part(i).x, shape.part(i).y), 1, (255, 255, 255), thickness=5)
                now = time.time()
                img = image_resize(img,width=400)
                img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
                cv.imwrite(os.path.join('/home/amir/Desktop/Webcam-Eyetracking/Da9_itis/data' , "{},{},{}.jpg".format(X,Y,now)), img)
                print(X,Y,img.shape)
                j+=1
            else:
                pass
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                sys.exit()
    # pass


cascade = cv.CascadeClassifier("Da9_itis/haarcascade_eye.xml")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('Da9_itis/shape_predictor_68_face_landmarks.dat')

with Listener(on_click=on_click) as listener:
    listener.join()