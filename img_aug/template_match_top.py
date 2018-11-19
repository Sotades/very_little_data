import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('data/images/flt_bottle.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('data/images/label_top_left.jpg', cv2.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]


# Apply template Matching
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.7
loc = np.where( res >= threshold)

pts = zip(*loc[::-1])
pointlist = list(pts)
ul = pointlist[1]
ul_x = ul[0]



for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

cv2.imshow('Detected', img_rgb)
cv2.waitKey(50)
