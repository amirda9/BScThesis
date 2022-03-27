import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

image1 = Image.open('./test.png')
image2 = plt.imread('./Heatmap.jpg')

print(image1.size)
f, axarr = plt.subplots(2,1)
axarr[0].imshow(image1)
axarr[1].imshow(image2)

plt.show()