import model
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import sys
import numpy as np
import cv2

if __name__ == '__main__':
    
    
    filepaths = os.listdir("/home/amir/Desktop/Webcam-Eyetracking/Final/data")
    X,Y = [],[]
    for filepath in filepaths:
        x, y, _ = filepath.split(',')
        x = float(x)
        y = float(y)
        a = cv2.imread("/home/amir/Desktop/Webcam-Eyetracking/Final/data/" + filepath)
        X.append(a)
        Y.append([x, y])
    X = np.array(X) 
    Y = np.array(Y)
    print(X.shape, Y.shape)

    print('hi')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # model
    net = model.model()
    net.train()
    net.to(device)
    
    # params
    loss_op = getattr(nn, 'L1Loss')().cuda()
    base_lr = 0.00001
    
    
    # optimizer
    optimizer = optim.Adam(net.parameters(),lr=base_lr, betas=(0.9,0.95))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)
    
    cur = 0
    timebegin = time.time()
    length = X.shape[0]
    total = length*20
    with open(os.path.join('./', "train_log"), 'w') as outfile:
        for epoch in range(1, 20):
            for i in range(length):
                
                # Acquire data
                # X[i] = X[i].to(device)
                # Y[i] = Y[i].to(device)
        
                # forward
                gaze = net(torch.from_numpy(X[i]).type(torch.FloatTensor))

                # loss calculation
                loss = loss_op(gaze, Y[i])
                print(loss)
                optimizer.zero_grad()

                # backward
                loss.backward()
                optimizer.step()
                scheduler.step()
                cur += 1

                # print logs
                if i % 20 == 0:
                    timeend = time.time()
                    resttime = (timeend - timebegin)/cur * (total-cur)/3600
                    log = f"[{epoch}/20]: [{i}/{length}] loss:{loss} lr:{base_lr}, rest time:{resttime:.2f}h"
                    print(log)
                    outfile.write(log + "\n")
                    sys.stdout.flush()   
                    outfile.flush()

                if epoch % 5 == 0:
                    torch.save(net.state_dict(), os.path.join(savepath, f"Iter_{epoch}_{modelname}.pt"))