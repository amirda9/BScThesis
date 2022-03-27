from msilib.schema import tables
import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import scipy.ndimage.filters as filters
import plot
import pickle




def myplot(data, title, save_path):
    colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0.75), (0, 1, 0), (0.75, 1, 0),
              (1, 1, 0), (1, 0.8, 0), (1, 0.7, 0), (1, 0, 0)]

    cm = LinearSegmentedColormap.from_list('sample', colors)

    plt.imshow(data, cmap=cm)
    plt.colorbar()
    plt.title(title)
    # plt.show()
    plt.savefig(save_path)
    plt.close()

def fit(centers,X,Y):
    d = np.ones((X.shape[0],centers.shape[0]))
    for i in range(X.shape[0]):
        for j in range(centers.shape[0]):
            d[i][j]= np.linalg.norm([X[i],Y[i]]-centers[j])
      
    labels=np.ones(X.shape[0])
    for i in range(X.shape[0]):
        labels[i]=np.argmin(d[i])
        # idx = np.argmin(d[i])
        # if d[i][idx]>100:
        #     labels[i] = -1
        # else:
        #     labels[i]=idx
    return labels


df = pickle.load( open( "data.p", "rb" ) )


# _______________________ DATASET _______________________
print(df.shape)


df_copy = (df.copy()).astype(int)
points = []
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        for k in range(df_copy[i,j]//10):
            points.append([i,j])


# print(points)
points = np.array(points)
# x = df.iloc[:,3]
# y = df.iloc[:,4]
# Study = [x,y]
# x = np.array(x)
# y = np.array(y)

# print(np.max(x),np.max(y))

# for i in range(len(x)):
#     x[i] = int(x[i] * 1920)//1
#     y[i] = int(y[i] * 1080)//1 - 250
#     # print(x[i],y[i])

# x = x.astype(int)
# y = y.astype(int)

# # table = np.concatenate((x,y), axis=1)
# # print(table.shape)
# Data = np.zeros((1920,1080))
# for i in range(len(x)):
#     # print(x[i],y[i])
#     try:
#         Data[x[i],y[i]] += 10,
#     except:
#         pass

# Data = Data.T
# # Smooth it to create a "blobby" look
# Data = filters.gaussian_filter(Data,sigma=30)

# myplot(Data, 'Sample plot', 'sample.jpg')


center =[]
for i in range(95):
    col = (50+(int(i/15)*70)) 
    row = (50+(int(i%15)*70)) 
    center.append([row,col])

center = np.array(center)
print(center)


labels = fit(center,points[:,0],points[:,1])

scores = 0
score_vec = np.ones(95)
labels = labels.astype(int)
for i in range(points.shape[0]):
    if labels[i] == -1:
        pass
    else:
        print(labels[i],i)
        score_vec[labels[i]] += 1
        scores += 1

# print(scores)
# print(0.71*1920,0.585*1080)
# i = 66
# y = ((50+(int(i/10)*70)) )
# x = ((50+(int(i%10)*125)) )
# print(x,y)
print(points.shape)
score_vec = score_vec /scores
print(score_vec)
print(np.argmax(score_vec))

scaled = 1 / (1 + np.exp(-10*score_vec))
print(scaled)
pickle.dump(scaled, open( "scaled.p", "wb" ) )
