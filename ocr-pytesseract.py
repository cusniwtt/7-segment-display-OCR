# Get required libraries
import cv2
import numpy as np
import pytesseract

# Define Functions
def ocr_core(img):
    text = pytesseract.image_to_string(img)
    return text

def get_grey_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def remove_noise(img):
    return cv2.medianBlur(img, 5)

def thresholding(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Main steps

image = cv2.imread('Image Test/t01.png')
cv2.imshow('Original', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

image = get_grey_scale(image)
image = thresholding(image)
image = remove_noise(image)
cv2.imshow('Preprocessing', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

text = ocr_core(image)
print(text)