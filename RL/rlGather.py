from pynput.mouse import Listener
import sys
import time
import cv2 as cv
import os
import numpy as np

cascade = cv.CascadeClassifier("Final/haarcascade_eye.xml")

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



cam_port = 0
cam = cv.VideoCapture(cam_port)

j=0
# capital x and y are click position
while(j<800):
    _, image = cam.read()
    img = image_resize(image,width=400)
    cv.imwrite("./data/"+str(j)+".jpg",img)
    print(j)
    j += 1


# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
