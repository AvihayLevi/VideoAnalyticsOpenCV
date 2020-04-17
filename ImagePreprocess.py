import cv2
import numpy as np
from skimage.filters import unsharp_mask, wiener


def test(frame):
    # Sanity check, do not use
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def unsharp(frame):
    # Apply gaussian blur to an image to make it sharper
    gaussian = cv2.GaussianBlur(frame, (9, 9), 10.0)
    unsharp_image = cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0, frame)
    return unsharp_image

"""
All functions:
input: image
optional: params, if any
output: processed image
"""

def convert2gray(frame):
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame

# Filter 2D
def filter2d(frame):
    # simple 2d sharpenning
    kernel = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]])
    sharppened = cv2.filter2D(frame, -1, kernel)
    return sharppened

# Unsharp
def unsharp_opencv(frame):
    # Apply gaussian blur to an image to make it
    frame = convert2gray(frame)
    gaussian = cv2.GaussianBlur(frame, (9, 9), 10.0)
    unsharp_image = cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0, frame)
    return unsharp_image

def unsharp_skimage(frame, amount=1):
    # Apply gaussian blur to an image to make it
    # stronger unsharp => use amount = 2
    frame = convert2gray(frame)
    unsharp_image = unsharp_mask(frame, radius=20, amount=amount)
    return (unsharp_image*255).astype('uint8')

# Wiener filter
def filt_func(r, c, sigma = 0.5):
    return np.exp(-np.hypot(r, c)/sigma)

def wiener_deblur(frame):
    return wiener(frame,filt_func)

# measure blur
def measure_blur(image):
    """
     compute the Laplacian of the image and then return the focus
	 measure, which is simply the variance of the Laplacian
     results should be examined with images. current levels are 120-150
    :param image:
    :return:
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()
