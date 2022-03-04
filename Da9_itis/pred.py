import numpy as np
import os
import cv2
import pyautogui
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from sklearn.model_selection import train_test_split

cascade = cv2.CascadeClassifier("Da9_itis/haarcascade_eye.xml")
video_capture = cv2.VideoCapture(0)

def normalize(x):
  minn, maxx = x.min(), x.max()
  return (x - minn) / (maxx - minn)
  
def scan(image_size=(32, 32)):
  _, frame = video_capture.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  boxes = cascade.detectMultiScale(gray, 1.3, 10)
  if len(boxes) == 2:
    eyes = []
    for box in boxes:
      x, y, w, h = box
      eye = frame[y:y + h, x:x + w]
      eye = cv2.resize(eye, image_size)
      eye = normalize(eye)
      eye = eye[10:-10, 5:-5]
      eyes.append(eye)
    return (np.hstack(eyes) * 255).astype(np.uint8)
  else:
    return None
    
# Note that there are actually 2560x1440 pixels on my screen
# I am simply recording one less, so that when we divide by these
# numbers, we will normalize between 0 and 1. Note that mouse
# coordinates are reported starting at (0, 0), not (1, 1)
width, height = 1920, 1080

filepaths = os.listdir("Da9_itis/v1")
X, Y = [], []
for filepath in filepaths:
  x, y, _ = filepath.split(' ')
  x = float(x) / width
  y = float(y) / height
  X.append(cv2.imread("Da9_itis/v1/" + filepath))
  Y.append([x, y])
  print([x,y],filepath)
X = np.array(X) / 255.0
Y = np.array(Y)
print (X.shape, Y.shape)

model = Sequential()
model.add(Conv2D(32, 3, 2, activation = 'relu', input_shape = (12, 44, 3)))
model.add(Conv2D(64, 2, 2, activation = 'relu'))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(2, activation = 'sigmoid'))
model.compile(optimizer = "adam", loss = "mean_squared_error")
model.summary()

epochs = 200
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
for epoch in range(epochs):
  model.fit(X_train, Y_train, batch_size = 32,validation_data=(X_test,Y_test))
  
while True:
  eyes = scan()
  if not eyes is None:
    eyes = np.expand_dims(eyes / 255.0, axis = 0)
    x, y = model.predict(eyes)[0]
    print(x,y)
    pyautogui.moveTo(x * width, y * height)
    if cv2.waitKey(1) & 0xFF == ord('q'):
                break