# import cv2 as cv
# import time

# # cv.NamedWindow("camera", 1)

# # capture = cv.CaptureFromCAM(0)

# i = 0
# while True:
#     img = cv.QueryFrame(capture)
#     cv.ShowImage("camera", img)
#     cv.SaveImage('pic{:>05}.jpg'.format(i), img)
#     if cv.WaitKey(10) == 27:
#         break
#     i += 1



# cam = VideoCapture(0)
# result, image = cam.read()

from pynput.mouse import Listener
import logging

logging.basicConfig(filename="mouse_log.txt", level=logging.DEBUG, format='%(asctime)s: %(message)s')

def on_move(x, y):
    logging.info("Mouse moved to ({0}, {1})".format(x, y))

def on_click(x, y, button, pressed):
    if pressed:
        logging.info('Mouse clicked at ({0}, {1}) with {2}'.format(x, y, button))

def on_scroll(x, y, dx, dy):
    logging.info('Mouse scrolled at ({0}, {1})({2}, {3})'.format(x, y, dx, dy))

with Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll) as listener:
    listener.join()