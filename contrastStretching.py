# Contras Streching
# https://theailearner.com/tag/percentile-stretching/
# https://notebook.community/OpenGenus/cosmos/code/artificial_intelligence/src/image_processing/contrast_enhancement/Min-Max-Contrast-Stretching

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg

# CONTRAST STRETCHING OF BLACK AND WHITE IMAGE
# ----------------------------------------------------------------------------------------------------------------------
# Reading the original image, here 0 implies that image is read as grayscale
path = 'lena.jpg'
img = cv2.imread(path,0)

# Generating the histogram of the original image
hist, bins = np.histogram(img.flatten(), 256, [0,256])

# Generating the cumulative distribution function of the original image
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

# Create zeros array to store the stretched image
img_cs = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')

# Loop over the image and apply Min-Max Contrasting formula
min = np.min(img)
max = np.max(img)

for i in range (img.shape[0]):
    for j in range (img.shape[1]):
        img_cs[i,j] = 255 * (img[i,j] - min)/(max-min)

# Generating the histogram of the image after applying Min-Max Contrast Stretchong
hist_cs, bins_cs = np.histogram(img_cs.flatten(),256,[0,256])

# Generating the cumulative distribution function of the original image
cdf_cs = hist_cs.cumsum()
cdf_cs_normalized = cdf_cs * hist_cs.max() / cdf_cs.max()

# Plotting the original and histogram equalized image, histogram, and CDF
fig, axs = plt.subplots(2,2)

axs[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0,0].axis('off')
axs[0,0].set_title('Original Image')

axs[0,1].imshow(cv2.cvtColor(img_cs, cv2.COLOR_BGR2RGB))
axs[0,1].axis('off')
axs[0,1].set_title('Contrast Stretched Image')

axs[1,0].plot(cdf_normalized, color='b')
axs[1,0].hist(img.flatten(),256,[0,256],color='r')
axs[1,0].legend(('cdf','histogram'), loc = 'upper left')

axs[1,1].plot(cdf_cs_normalized, color = 'b')
axs[1,1].hist(img_cs.flatten(),256,[0,256], color = 'r')
axs[1,1].legend(('cdf_cs','histogram'), loc = 'upper left')

# Hide x labels and tick labels for top plots and y ticks for right plots
for ax in axs.flat:
    ax.label_outer()

plt.show()

# CONTRAST STRETCHING OF COLOR IMAGES
# ----------------------------------------------------------------------------------------------------------------------

# Reading the original image, here 1 implies that image is read as color
imgc = cv2.imread('lena.jpg', 1)

# Generating the histogram of the original image
hist_c, bins_c = np.histogram(imgc.flatten(),256,[0,256])

# Generating the cumulative distribution function of the original image
cdf_c = hist_c.cumsum()
cdf_c_normalized = cdf_c * hist_c.max() / cdf_c.max()

# Converting the image to YCrCb
img_yuv = cv2.cvtColor(imgc, cv2.COLOR_BGR2YUV)

# Loop over the Y channel and apply Min-Max Contrasting
min = np.min(img_yuv[:,:,0])
max = np.max(img_yuv[:,:,0])

for i in range(imgc.shape[0]):
    for j in range(imgc.shape[1]):
        img_yuv[:,:,0][i,j] = 255 * (img_yuv[:,:,0][i,j]-min)/(max-min)

# Convert the YUV image back to RGB format
img_c_cs = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# Generating the histogram of the image after applying contrast stretching
hist_c_cs, bins_c_cs = np.histogram(img_c_cs.flatten(), 256, [0,256])

# Generating the cumulative distribution function of the original image
cdf_c_cs = hist_c_cs.cumsum()
cdf_c_cs_normalized = cdf_c_cs * hist_c_cs.max() / cdf_c_cs.max()

# Plotting the original and histogram equalized image, histogram, and CDF
fig, axs = plt.subplots(2,2)

axs[0,0].imshow(cv2.cvtColor(imgc, cv2.COLOR_BGR2RGB))
axs[0,0].axis('off')
axs[0,0].set_title('Original Image')

axs[0,1].imshow(cv2.cvtColor(img_c_cs, cv2.COLOR_BGR2RGB))
axs[0,1].axis('off')
axs[0,1].set_title('Contrast Stretched Image')

axs[1,0].plot(cdf_c_normalized, color='b')
axs[1,0].hist(imgc.flatten(),256,[0,256],color='r')
axs[1,0].legend(('cdf','histogram'), loc = 'upper left')

axs[1,1].plot(cdf_c_cs_normalized, color = 'b')
axs[1,1].hist(img_c_cs.flatten(),256,[0,256], color = 'r')
axs[1,1].legend(('cdf_cs','histogram'), loc = 'upper left')

# Hide x labels and tick labels for top plots and y ticks for right plots
for ax in axs.flat:
    ax.label_outer()

plt.show()