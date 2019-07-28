import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


# Read image from system path as RGB type.
def read_image(path):
    img = cv2.imread(path)
    # opencv default color space is BGR, change it to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def show_image(img, figsize=(10, 10), gray=False):
    plt.figure(figsize=figsize)
    if gray:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()


def get_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def rotate(img, du):
    image = Image.fromarray(np.uint8(img))
    image = image.rotate(du)
    image = np.asarray(image)
    return image
