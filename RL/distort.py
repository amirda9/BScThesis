import os
import numpy as np
from PIL import Image,ImageEnhance
import random



def _transit( a,image):
    s = image
    if a == 0:
        return s
    elif a == 1:
        return np.asarray(adjust_contrast(Image.fromarray(s), 0.95))
    elif a == 2:
        return np.asarray(adjust_contrast(Image.fromarray(s), 1.05))
    elif a == 3:
        return np.asarray(adjust_saturation(Image.fromarray(s), 0.95))
    elif a == 4:
        return np.asarray(adjust_saturation(Image.fromarray(s), 1.05))
    elif a == 5:
        return np.asarray(adjust_brightness(Image.fromarray(s), 0.95))
    elif a == 6:
        return np.asarray(adjust_brightness(Image.fromarray(s), 1.05))
    elif a == 7:
        return adjust_channels(s, 0.95, [0, 1])
    elif a == 8:
        return adjust_channels(s, 1.05, [0, 1])
    elif a == 9:
        return adjust_channels(s, 0.95, [2, 1])
    elif a == 10:
        return adjust_channels(s, 1.05, [2, 1])
    elif a == 11:
        return adjust_channels(s, 0.95, [0, 2])
    elif a == 12:
        return adjust_channels(s, 1.05, [0, 2])
    else:
        raise NotImplementedError



def adjust_contrast(image_rgb, contrast_factor):
    """Adjust contrast"""
    enhancer = ImageEnhance.Contrast(image_rgb)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(image_rgb, saturation_factor):
    """Adjust saturation"""
    enhancer = ImageEnhance.Color(image_rgb)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_brightness(image_rgb, brightness_factor):
    """Adjust brightness"""
    enhancer = ImageEnhance.Brightness(image_rgb)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_channels(array, factor, channels):
    """Adjust channel pixel value"""
    temp = array.copy()
    for c in channels:
        temp[:, :, c] = (array[:, :, c] * factor).clip(0, 255).astype('uint8')
    return temp


for i in range(0,4999):
    im = Image.open('./data/{}.jpg'.format(i))
    im_arr = np.array(im)
    sets = random.randint(1,20)
    for _ in range(0,sets):
        action = random.randint(0,12)
        im_arr = _transit(action,im_arr)
        im = Image.fromarray(im_arr)
        im.save('./distortion/'+str(i)+'.jpg')