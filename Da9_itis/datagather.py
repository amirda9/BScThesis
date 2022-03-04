from pynput.mouse import Listener
import logging
import sys
import time
import cv2 as cv
import os

def on_click(x, y, button, pressed):
    if pressed:
        # logging.info('Mouse clicked at ({0}, {1}) with {2}'.format(x, y, button))
        # print('Mouse clicked at ({0}, {1}) with {2}'.format(x, y, button))
        cam_port = 0
        cam = cv.VideoCapture(cam_port)

        i=0
        while(i<3):
            result, img = cam.read()
        	# saving image in local storage
            now = time.time()
            # img = cv.resize(image, (188,256))
            # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # print(img)
            cv.imwrite(os.path.join('/home/amir/Desktop/Webcam-Eyetracking/Da9_itis/data' , "{},{},{}.jpg".format(x,y,now)), img)
            print(x,y,img.shape)
            i+=1
        if cv.waitKey(1) & 0xFF == ord('q'):
            sys.exit()

with Listener(on_click=on_click) as listener:
    listener.join()