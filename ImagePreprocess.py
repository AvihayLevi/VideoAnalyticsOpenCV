import cv2


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