from pynput.mouse import Listener
import sys
import time
import cv2 as cv
import os
import numpy as np

cascade = cv.CascadeClassifier("/home/amir/Desktop/Webcam-Eyetracking/Final/haarcascade_eye.xml")

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


def process(image):
    image = image_resize(image,width=400)
    image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    return image

def get_eye(image):
    image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    boxes = cascade.detectMultiScale(image, 1.3, 10)
    if len(boxes) == 2:
        if boxes[0][1] > boxes[1][1]:
            right_eye = image[boxes[0][1]:boxes[0][1] + boxes[0][3], boxes[0][0]:boxes[0][0] + boxes[0][2]]
            left_eye = image[boxes[1][1]:boxes[1][1] + boxes[1][3], boxes[1][0]:boxes[1][0] + boxes[1][2]]
        else:
            left_eye = image[boxes[0][1]:boxes[0][1] + boxes[0][3], boxes[0][0]:boxes[0][0] + boxes[0][2]]
            right_eye = image[boxes[1][1]:boxes[1][1] + boxes[1][3], boxes[1][0]:boxes[1][0] + boxes[1][2]]
        
        right_eye = cv.resize(right_eye,(60,30))
        left_eye = cv.resize(left_eye,(60,30))
        return right_eye,left_eye
    else:
        print(len(boxes))
        return None,None

def on_click(X, Y, button, pressed):
    if pressed:
        # logging.info('Mouse clicked at ({0}, {1}) with {2}'.format(x, y, button))
        # print('Mouse clicked at ({0}, {1}) with {2}'.format(x, y, button))
        cam_port = 0
        cam = cv.VideoCapture(cam_port)

        j=0
        # capital x and y are click position
        while(j<2):
            _, image = cam.read()
            now = time.time()
            img = process(image)
            RI,LI = get_eye(image)
            if (RI is not None) and (LI is not None):
                cv.imwrite(os.path.join('/home/amir/Desktop/Webcam-Eyetracking/Final/newdata/' , "Face,{},{},{}.jpg".format(X,Y,now)), img)
                cv.imwrite(os.path.join('/home/amir/Desktop/Webcam-Eyetracking/Final/newdata/' , "RI,{},{},{}.jpg".format(X,Y,now)), RI)
                cv.imwrite(os.path.join('/home/amir/Desktop/Webcam-Eyetracking/Final/newdata/' , "LI,{},{},{}.jpg".format(X,Y,now)), LI)
                print(X,Y,img.shape,RI.shape,LI.shape)
            
            
            
            j+=1
            if cv.waitKey(1) & 0xFF == ord('q'):
                sys.exit()
    # pass
    
with Listener(on_click=on_click) as listener:
    listener.join()



# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
