import cv2 as cv
import os

def image_resize(image, width = None, height = None, inter = cv.INTER_AREA):
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
    resized = cv.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


# cv.imwrite('raw_resized.jpg', image_resize(cv.imread('raw.jpg'), width=480))
# cv.imwrite('retouch_resized.jpg', image_resize(cv.imread('retouched.jpg'), width=480))


files = os.listdir('../../Raw')
files = sorted(files)
for filename in files:
    print(filename)
    img = cv.imread('../../Raw/{}'.format(filename))
    img2 = cv.imread('../../C/{}'.format(filename))
    
    if (img is not None and img2 is not None):
        resized = image_resize(img,width=480)
        cv.imwrite('../../Raw/{}'.format(filename),resized)

        resized2 = image_resize(img2,width=480)
        cv.imwrite('../../C/{}'.format(filename),resized2)
    # train_pairs.append(['../../Raw/{}'.format(filename),'../../C/{}'.format(filename)])