import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('input.jpeg', cv2.IMREAD_GRAYSCALE) #read input image

M, N = image.shape

padded_median = np.pad(image, pad_width=2, mode='reflect') #5x5 median filter
median_filter = np.zeros_like(image)

#compute the median for the 5x5 median filter by looping through and computing each piuxel
for m in range(M):
    for n in range(N):
        region = padded_median[m : m+5, n : n+5]
        median_filter[m, n] = np.median(region)

padded_knn = np.pad(image, pad_width=1, mode='reflect') #3x3 KNN mean filter
knn_filtered = np.zeros_like(image)


for m in range(M):        #compute the KNN mean by looping through and computing each pixel      
    for n in range(N):
        region = padded_knn[m : m+3, n : n+3]
        center_value = padded_knn[m+1, n+1]
        differences = np.abs(region - center_value)
        flat_region = region.flatten()
        flat_diffs = differences.flatten()
        sorted_indices = np.argsort(flat_diffs)
        selected_pixels = flat_region[sorted_indices[:4]]
        knn_filtered[m, n] = np.mean(selected_pixels)

plt.figure(figsize=(15, 5)) #Display the three images

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(median_filter, cmap='gray')
plt.title('5x5 Median Filter')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(knn_filtered, cmap='gray')
plt.title('3x3 KNN Mean Filter (k=4)')
plt.axis('off')

plt.tight_layout()
plt.show()
