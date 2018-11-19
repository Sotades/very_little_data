import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('../data/new_images/NOK/OK.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('data/images/match_template.jpg', cv2.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]


# Apply template Matching
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

cv2.imshow('Detected', img_rgb)
cv2.waitKey(50)
