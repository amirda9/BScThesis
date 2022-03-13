import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pickle

import scipy.ndimage.filters as filters


def plot(data, title, save_path):
    colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0.75), (0, 1, 0), (0.75, 1, 0),
              (1, 1, 0), (1, 0.8, 0), (1, 0.7, 0), (1, 0, 0)]

    cm = LinearSegmentedColormap.from_list('sample', colors)

    plt.imshow(data, cmap=cm)
    plt.colorbar()
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    
    
    data = pickle.load( open( "data.p", "rb" ) )
    data = data.T
    # Smooth it to create a "blobby" look
    data = filters.gaussian_filter(data,sigma=15)

    plot(data, 'Sample plot', 'sample.jpg')
