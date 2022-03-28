import cv2 
import numpy as np
import os
import random


filepaths = os.listdir("./VAL/")
valF,valR,valL, valY = [], [],[],[]
i =0 
for filepath in filepaths:
    if i >9:
        break
    kind,x, y, _ = filepath.split(',')
    if kind == 'Face':
        a = cv2.imread("./VAL/" + filepath, cv2.IMREAD_GRAYSCALE)
        a = cv2.resize(a, (200,150), interpolation = cv2.INTER_AREA)
        try:
            valF.append(a)
            i += 1
        except:
            pass
    else:
        pass
valF = np.array(valF)

print(valF.shape)



i = 0
while i<10:
    for _ in range(0,100):
        pixel_x = random.randint(0,149)
        pixel_y = random.randint(0,199)
        valF[i][pixel_x,pixel_y] = 255
    for __ in range(0,100):    
        pixel_x = random.randint(0,149)
        pixel_y = random.randint(0,199)
        valF[i][pixel_x,pixel_y] = 0
    cv2.imwrite("noisy/" + str(i) + ".jpg", valF[i])
    i += 1