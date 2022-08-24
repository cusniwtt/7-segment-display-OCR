# import the necessary packages
import cv2 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
# define the dictionary of digit segments so we can identify
# each digit on the thermostat
DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 0, 1): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}

#Read the image
img = cv2.imread('Image Test/t01.png', 0)
img = imutils.resize(img, height=244, width=244)
print(img.shape)
cv2.imshow('Original', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Apply contrast stretching
alpha = 2
beta = -1
img_con = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
cv2.imshow('Contrast', img_con)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Set Threshold
###Gausian blur + Osthu threshold
blur = cv2.GaussianBlur(img_con, (5,5), 0)
img_gthr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cv2.imshow('Gaussian Threshold', img_gthr)
cv2.waitKey(0)
cv2.destroyAllWindows()
img_thr = img_gthr

#Set Opening
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
img_opn = cv2.morphologyEx(img_thr, cv2.MORPH_OPEN, kernel)
cv2.imshow('Opening', img_opn)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Image to plot
img_plot = img_thr

#Plot Histogram of image
#plt.hist(img_plot.ravel(), 256, [0,256], color='crimson')
#plt.ylabel("Number Of Pixels", color='crimson')
#plt.xlabel("Pixel Intensity- From 0-255", color='crimson')
#plt.title("Histogram Showing Pixel Intensity And Corresponding Number Of Pixels", color='crimson')
#plt.show()

# find contours in the thresholded image, then initialize the
# digit contours lists
cnts = cv2.findContours(img_plot.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
digitCnts = []
# loop over the digit area candidates
for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)
	# if the contour is sufficiently large, it must be a digit
    if w >= 30 and (h >=80  and h <= 115):
        rec = cv2.rectangle(img_plot, (x, y), (x + w, y + h), (255, 255, 255), 2)
        digitCnts.append(c)
cv2.imshow('Contours 1', rec)
cv2.waitKey(0)
cv2.destroyAllWindows()

# sort the contours from left-to-right, then initialize the
# actual digits themselves
digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
digits = []

# loop over each of the digits
for c in digitCnts:
	# extract the digit ROI
	(x, y, w, h) = cv2.boundingRect(c)
	roi = img_plot[y:y + h, x:x + w]
	# compute the width and height of each of the 7 segments
	# we are going to examine
	(roiH, roiW) = roi.shape
	(dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
	dHC = int(roiH * 0.05)
	# define the set of 7 segments
	segments = [
		((0, 0), (w, dH)),	# top
		((0, 0), (dW, h // 2)),	# top-left
		((w - dW, 0), (w, h // 2)),	# top-right
		((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
		((0, h // 2), (dW, h)),	# bottom-left
		((w - dW, h // 2), (w, h)),	# bottom-right
		((0, h - dH), (w, h))	# bottom
	]
	on = [0] * len(segments)

    # loop over the segments
	for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
		# extract the segment ROI, count the total number of
		# thresholded pixels in the segment, and then compute
		# the area of the segment
		segROI = roi[yA:yB, xA:xB]
		total = cv2.countNonZero(segROI)
		area = (xB - xA) * (yB - yA)
		# if the total number of non-zero pixels is greater than
		# 50% of the area, mark the segment as "on"
		if total / float(area) > 0.5:
			on[i]= 1
	# lookup the digit and draw it on the image
	digit = DIGITS_LOOKUP[tuple(on)]
	digits.append(digit)
	cv2.rectangle(img_plot, (x, y), (x + w, y + h), (0, 255, 0), 1)
	cv2.putText(img_plot, str(digit), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

# display the digits
print(u"{}{}.{}".format(*digits))
cv2.imshow("Input", img)
cv2.imshow("Output", img_plot)
cv2.waitKey(0)