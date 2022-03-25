from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import os
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from matplotlib import colors
import pickle
from sklearn.model_selection import train_test_split



# _______________________ DATASET _______________________

filepaths = os.listdir("./data/")
dataF,dataR,dataL, Y = [], [],[],[]
for filepath in filepaths:
    kind,x, y, _ = filepath.split(',')
    if kind == 'Face':
        a = cv2.imread("./data/" + filepath, cv2.IMREAD_GRAYSCALE)
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

#Implementation of matplotlib function
# plt.hist2d(Y[:, 0], 
#            Y[:, 1],bins=100)
# plt.show()
    
    
if __name__ == '__main__':
    

    
    X,Y,X_val,Y_val = train_test_split([dataR, dataL, dataF],Y, test_size=0.2, random_state=30)
    print(X.shape, Y.shape, X_val.shape, Y_val.shape)
    
    # ________________________ MODEL ________________________
    
    eyeInputR = Input((30,60,1))
    eyeInputL = Input((30,60,1))
    faceInput = Input((300,400,1))
    
    
    
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
    Face = (Conv2D(96, (9, 12), activation='relu', input_shape=(300, 400, 1)))(faceInput)
    Face = (AveragePooling2D(pool_size=(3, 3)))(Face)
    Face = (Conv2D(256, (5, 5), activation='relu'))(Face)
    Face = (AveragePooling2D(pool_size=(3, 3))) (Face)
    Face = (Conv2D(128, (3, 3), activation='relu'))(Face)
    Face = (Conv2D(128, (3, 3), activation='relu'))(Face)
    Face = (Conv2D(64, (3, 3), activation='relu') )(Face)
    Face = (Flatten())(Face)
    Face = (Dense(8, activation='linear') )(Face)
    
    res = Concatenate()([RE, LE, Face])
    res = Dense(2, activation='linear')(res)
    model = Model(inputs=[eyeInputR, eyeInputL, faceInput], outputs=res)
    print(model.summary())
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    history = model.fit(x=[dataR, dataL, dataF], y=Y, epochs=10, batch_size=8)
    pickle.dump( history, open( "data.p", "wb" ) )
