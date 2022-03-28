import cv2 
import numpy as np
import os
import tensorflow as tf


filepaths = os.listdir("./TEST/")
valF,valR,valL, valY = [], [],[],[]
for filepath in filepaths:
    kind,x, y, _ = filepath.split(',')
    if kind == 'Face':
        a = cv2.imread("./TEST/" + filepath, cv2.IMREAD_GRAYSCALE)
        a = cv2.resize(a, (200,150), interpolation = cv2.INTER_AREA)
        a = np.expand_dims(a, axis=2)
        b_path = filepath.replace('Face', 'RI')
        b = cv2.imread("./TEST/" + b_path, cv2.IMREAD_GRAYSCALE)
        c_path = filepath.replace('Face', 'LI')
        c = cv2.imread("./TEST/" + c_path, cv2.IMREAD_GRAYSCALE)
        try:
            b = np.expand_dims(b, axis=2)
            c = np.expand_dims(c, axis=2)
            valF.append(a)
            valR.append(b)
            valL.append(c)
            x = float(x) / 1920
            y = float(y) / 1080
            valY.append([x, y])
        except:
            try:
                print(b.shape)
            except:
                pass
    else:
        pass
valF = np.array(valF)
valR = np.array(valR)
valL = np.array(valL)
valY = np.array(valY)
print(valF.shape,valR.shape,valL.shape, valY.shape)


model = tf.keras.models.load_model('./eyemodel.h5')

model.summary()

model.evaluate([valR, valL,valF], valY)

pred = model.predict([valR, valL,valF])
# print(pred)


# a = cv2.cvtColor(valF[20], cv2.COLOR_GRAY2BGR)
# a = cv2.circle(a, (int(valY[0][0]*150), int(valY[0][1]*200)), 5, (255), -1)

# while(1):
#     cv2.imshow('img',a)
#     k = cv2.waitKey(33)
#     if k==27:    # Esc key to stop
#         break
#     elif k==-1:  # normally -1 returned,so don't print it
#         continue
#     else:
#         print (k) # else print its value
# cv2.waitKey(0)
    
for i in range(valF.shape[0]):
    # print(valF[i].shape)
    a = cv2.flip(valF[i], 1)
    a = np.expand_dims(a, axis=2)
    # print(a.shape)
    # ground truth is white
    a = cv2.circle(a, (int(valY[i][0]*150), int(valY[i][1]*200)), 5, (255,255,255), -1)
    # pred is black
    a = cv2.circle(a, (int(pred[i][0]*150), int(pred[i][1]*200)), 5, (0,0,0), -1)
    cv2.imwrite('./TEST_marked/' + str(i) + '.jpg', a)