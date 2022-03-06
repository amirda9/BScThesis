from statistics import mode
import numpy as np
import os
import cv2
import time
import pyautogui
from sklearn import metrics
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from sklearn.model_selection import train_test_split
import dlib
from matplotlib import pyplot as plt

# cascade = cv2.CascadeClassifier("Da9_itis/haarcascade_eye.xml")
video_capture = cv2.VideoCapture(0)


cascade = cv2.CascadeClassifier("Da9_itis/haarcascade_eye.xml")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    'Da9_itis/shape_predictor_68_face_landmarks.dat')


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
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
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


width, height = 1920, 1080

filepaths = os.listdir("Da9_itis/data")
X, Y = [], []
for filepath in filepaths:
    x, y, _ = filepath.split(',')
    # x = float(x) / width
    # y = float(y) / height
    x = float(x)
    y = float(y)
    # print(filepath)
#   print(cv2.imread("Da9_itis/data/" + filepath).shape)
    a = cv2.imread("Da9_itis/data/" + filepath, cv2.IMREAD_GRAYSCALE)
    a = np.expand_dims(a, axis=2)
    # a = cv2.imread("Da9_itis/data/" + filepath, cv2.IMREAD_
    X.append(a)
    Y.append([x, y])
X = np.array(X) 
Y = np.array(Y)
print(X.shape, Y.shape)


epochs = 30
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=30)
print('\n\n')
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape )
print('\n\n')




# model = Sequential()
# model.add(Conv2D(32, 3, 2, activation = 'relu', input_shape = [150,200,1]))
# model.add(Conv2D(64, 2, 2, activation = 'relu'))
# model.add(Flatten())
# model.add(Dense(32, activation = 'relu'))
# model.add(Dense(2, activation = 'sigmoid'))
# model.compile(optimizer = "adam", loss = "mean_squared_error",metrics=['mse','mae'])
# model.summary()

# model = Sequential()

# # Conv1 32 32 (32)
# model.add(Conv2D(32, 3,2, input_shape=[300, 400, 1]))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Conv2D(16, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# # Conv2 16 16 (64)
# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# # Conv2 8 8 (128)
# model.add(Conv2D(128, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(128, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# # FC
# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(32))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2))
# model.add(Activation('linear'))
# model.summary()
# model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error', 'mean_absolute_error'])

model = load_model('my_model.h5')
model.summary()
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=32,
          validation_data=(X_test, Y_test))


model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'


print(history.history.keys())
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()





# while True:
#     # Capture frame-by-frame
#     ret, frame = video_capture.read()
    
    
    
#     # if frame is read correctly ret is True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
      
      
      
#     else:
#       img = np.zeros_like(frame)
#       boxes = cascade.detectMultiScale(frame, 1.3, 10)
#       if len(boxes)==2:
#         for box in boxes:
#             x, y, w, h = box
#             img[y:y + h, x:x + w] = frame[y:y + h, x:x + w]
#         dets = detector(frame, 1)
#         for k, d in enumerate(dets):
#             shape = predictor(frame, d)
#             if not shape.part(1):
#               print('bad picture')
#               pass
#             else:
#               for i in range(68):
#                   img = cv2.circle(img, (shape.part(i).x, shape.part(
#                       i).y), 1, (255, 255, 255), thickness=5)
#         img = image_resize(img,width=200)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img = np.expand_dims(img, axis=2)
#         cv2.imshow("img", img)
#         img = np.expand_dims(img, axis = 0)
#         print(img.shape,X_train[1].shape)
#         x, y = model.predict(img)[0]
#         print(x, y)
#         pyautogui.moveTo(x , y)
        
#       else:
#         pass
#       if cv2.waitKey(1) == ord('q'):
#           break
      
#       #     # # pyautogui.FAILSAFE = False
#       #     # pyautogui.moveTo(x*width , y*height)
#       #     if cv2.waitKey(1) & 0xFF == ord('q'):
#       #         break
