from turtle import width
from pynput.mouse import Listener
import logging
import sys
import time
import cv2 as cv
import os
import numpy as np
import pickle

Data = np.zeros((1920,1080))


def on_click(X, Y, button, pressed):
    if pressed:
        print(Data[X,Y])
        pickle.dump( Data, open( "data.p", "wb" ) )
        # pass
    
    
def on_move(X, Y):
    print('working')
    Data[X,Y] += 10
    # pass


with Listener(on_click=on_click , on_move=on_move , on_scroll=on_move) as listener:
    listener.join()