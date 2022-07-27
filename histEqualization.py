import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read image in grayscale
path = 'lena.jpg'
img = cv2.imread(path,0)
hist, bins = np.histogram(img.flatten(), 256, [0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max())/cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(), 256, [0, 256], color = 'r')
plt.xlim([0,256])
plt.legend(('cfd', 'histogram'), loc = 'upper left')
plt.show()

cdf_masked = np.ma.masked_equal(cdf,0)
cdf_masked = (cdf_masked - cdf_masked.min())*255/(cdf_masked.max()-cdf_masked.min())
cdf = np.ma.filled(cdf_masked,0).astype('uint8')
img2 = cdf[img]
hist1, bins = np.histogram(img2.flatten(), 256, [0,256])
cdf = hist1.cumsum()
cdf_normalized1 = cdf * float(hist1.max())/cdf.max()
plt.plot(cdf_normalized1, color = 'b')
plt.hist(img2.flatten(), 256, [0, 256], color = 'r')
plt.xlim([0,256])
plt.legend(('cfd', 'histogram'), loc = 'upper left')
plt.show()

# HISTOGRAM EQUALIZATION IN OPENCV
# creating a Histograms Equalization
equ = cv2.equalizeHist(img)
# Stacking image side-by-side
res = np.hstack((img, equ))
# Show image input vs output images
cv2.imshow('Simple Histogram Equalization', res)
cv2.waitKey(0)

hist2, bins = np.histogram(equ.flatten(), 256, [0,256])
cdf = hist2.cumsum()
cdf_normalized = cdf * float(hist2.max())/cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(equ.flatten(), 256, [0, 256], color = 'r')
plt.xlim([0,256])
plt.legend(('cfd', 'histogram'), loc = 'upper left')
plt.show()


# CLAHE - Contrast Limited Adaptive Histogram Equalization
clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
cl1 = clahe.apply(img)
res1 = np.hstack((img, cl1))
cv2.imshow('Contrast Limited Adaptive Histogram Equalization', res1)
cv2.waitKey(0)

hist3, bins = np.histogram(cl1.flatten(), 256, [0,256])
cdf = hist3.cumsum()
cdf_normalized = cdf * float(hist3.max())/cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(cl1.flatten(), 256, [0, 256], color = 'r')
plt.xlim([0,256])
plt.legend(('cfd', 'histogram'), loc = 'upper left')
plt.show()

image = cv2.imread(path)


# COLORED IMAGE
# ---------------------------------------------------------
# convert image from RGB to HSV
img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
# Histogram equalisation on the V-channel
img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
# convert image back from HSV to RGB
image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
cv2.imshow("equalizeHist", image)
cv2.waitKey(0)
# ---------------------------------------------------------
img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
cv2.imshow("equalizeHist", img)
cv2.waitKey(0)