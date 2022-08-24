# Get required libraries
import cv2
import numpy as np
import pytesseract
import imutils

# Define Functions
def ocr_core(img):
    text = pytesseract.image_to_string(img, config='--psm 6')
    return text

def get_grey_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def rescale(img, scale_percent=100):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation =cv2.INTER_AREA)

def thresholding(img):
    blur = cv2.GaussianBlur(img, (5,5), 0)
    output = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return output

def dilating(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    output = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return output

# Main steps

image = cv2.imread('Image Test/t02.png')
image = imutils.resize(image, height=244, width=244)
cv2.imshow('Original', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

image = get_grey_scale(image)
image = rescale(image, scale_percent=50)
image = thresholding(image)
image = dilating(image)
cv2.imshow('Preprocessing', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('OCR output: ', ocr_core(image))