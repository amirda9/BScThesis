from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import os
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from matplotlib import colors
import pickle



# _______________________ DATASET _______________________

filepaths = os.listdir("./data/")
dataF,dataR,dataL, Y = [], [],[],[]
for filepath in filepaths:
    kind,x, y, _ = filepath.split(',')
    if kind == 'Face':
        a = cv2.imread("./data/" + filepath, cv2.IMREAD_GRAYSCALE)
        a = cv2.resize(a, (200,150), interpolation = cv2.INTER_AREA)
        a = np.expand_dims(a, axis=2)
        b_path = filepath.replace('Face', 'RI')
        b = cv2.imread("./data/" + b_path, cv2.IMREAD_GRAYSCALE)
        c_path = filepath.replace('Face', 'LI')
        c = cv2.imread("./data/" + c_path, cv2.IMREAD_GRAYSCALE)
        try:
            b = np.expand_dims(b, axis=2)
            c = np.expand_dims(c, axis=2)
            dataF.append(a)
            dataR.append(b)
            dataL.append(c)
            x = float(x) / 1920
            y = float(y) / 1080
            Y.append([x, y])
        except:
            try:
                print(b.shape)
            except:
                pass
    else:
        pass
dataF = np.array(dataF)
dataR = np.array(dataR)
dataL = np.array(dataL)
Y = np.array(Y)
print(dataF.shape,dataR.shape,dataL.shape, Y.shape)



filepaths = os.listdir("./VAL/")
valF,valR,valL, valY = [], [],[],[]
for filepath in filepaths:
    kind,x, y, _ = filepath.split(',')
    if kind == 'Face':
        a = cv2.imread("./VAL/" + filepath, cv2.IMREAD_GRAYSCALE)
        a = cv2.resize(a, (200,150), interpolation = cv2.INTER_AREA)
        a = np.expand_dims(a, axis=2)
        b_path = filepath.replace('Face', 'RI')
        b = cv2.imread("./VAL/" + b_path, cv2.IMREAD_GRAYSCALE)
        c_path = filepath.replace('Face', 'LI')
        c = cv2.imread("./VAL/" + c_path, cv2.IMREAD_GRAYSCALE)
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



#Implementation of matplotlib function
# plt.hist2d(Y[:, 0], 
#            Y[:, 1],bins=100)
# plt.show()
    
    
if __name__ == '__main__':
    

    
    
    # ________________________ MODEL ________________________
    
    eyeInputR = Input((30,60,1))
    eyeInputL = Input((30,60,1))
    faceInput = Input((150,200,1))
    
    
    
    # Right eye
    RE = Conv2D(96, (6, 3), activation='relu')(eyeInputR)
    RE = (AveragePooling2D(pool_size=(2, 2)))(RE)
    RE = (Conv2D(384,(3, 3), activation='relu'))(RE)
    RE = (Conv2D(256,(3, 3), activation='relu'))(RE)
    RE = (Flatten())(RE)
    RE = (Dense(8, activation='linear'))(RE)
    
    
    
    # Left eye
    LE = Conv2D(96, (6, 3), activation='relu')(eyeInputL)
    LE = (AveragePooling2D(pool_size=(2, 2)))(LE)
    LE = (Conv2D(384,(3, 3), activation='relu'))(LE)
    LE = (Conv2D(256,(3, 3), activation='relu'))(LE)
    LE = (Flatten())(LE)
    LE = (Dense(8, activation='linear'))(LE)
    
    
    # face
    Face = (Conv2D(96, (9, 12), activation='relu', input_shape=(150, 200, 1)))(faceInput)
    Face = (AveragePooling2D(pool_size=(3, 3)))(Face)
    Face = (Conv2D(256, (5, 5), activation='relu'))(Face)
    Face = (AveragePooling2D(pool_size=(3, 3))) (Face)
    Face = (Conv2D(384, (3, 3), activation='relu'))(Face)
    Face = (Conv2D(256, (3, 3), activation='relu'))(Face)
    Face = (Conv2D(128, (3, 3), activation='relu') )(Face)
    Face = (Flatten())(Face)
    Face = (Dense(16, activation='linear') )(Face)
    
    res = Concatenate()([RE, LE, Face])
    res = Dense(2, activation='linear')(res)
    model = Model(inputs=[eyeInputR, eyeInputL, faceInput], outputs=res)
    print(model.summary())
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])
    history = model.fit(x=[dataR, dataL, dataF], y=Y, epochs=100, batch_size=8 , validation_data = ( [valR, valL, valF], valY))
    pickle.dump( history, open( "data.p", "wb" ) )

    model.save('./model.h5')
