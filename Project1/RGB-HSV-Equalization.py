import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt

img_rgb = io.imread('Low-light-img1-256.jpg') #read the lowlight image
if img_rgb.dtype == np.uint8:
    img_rgb = img_rgb / 255.0

img_hsv = color.rgb2hsv(img_rgb) #converts rgb to hsv

intensity = img_hsv[:, :, 2] #get the value channel

rows, cols = intensity.shape
intensity_uint8 = np.zeros((rows, cols), dtype=np.uint8)
for i in range(rows):
    for j in range(cols):
        # Multiply by 255 and round to the nearest integer
        intensity_uint8[i, j] = int(round(intensity[i, j] * 255)) #converts the intensity to be from 0-255 instead of float 0-1

# ----- Step 4. Compute histogram and cumulative distribution (explicitly) -----
# Initialize histogram array with 256 bins (for 0-255 levels)
hist = np.zeros(256, dtype=int)
for i in range(rows):
    for j in range(cols):
        pixel_val = intensity_uint8[i, j]
        hist[pixel_val] += 1

cdf = np.zeros(256, dtype=float) #computes the cdf
cdf[0] = hist[0]
for i in range(1, 256):
    cdf[i] = cdf[i - 1] + hist[i]

# Gets the minimum nonzero cdf value
cdf_min = 0
for i in range(256):
    if hist[i] != 0:
        cdf_min = cdf[i]
        break

total_pixels = rows * cols #computes the total amount of pixerls

mapping = np.zeros(256, dtype=np.uint8) #Compute the histogram specification mapping
for i in range(256):
    mapped_val = (cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255
    mapping[i] = int(round(mapped_val))
    if mapping[i] < 0:
        mapping[i] = 0
    if mapping[i] > 255:
        mapping[i] = 255

new_intensity_uint8 = np.zeros((rows, cols), dtype=np.uint8) #get K(x,y)
for i in range(rows):
    for j in range(cols):
        orig_val = intensity_uint8[i, j]
        new_intensity_uint8[i, j] = mapping[orig_val]

new_intensity = np.zeros((rows, cols), dtype=float) #convert values back to fp
for i in range(rows):
    for j in range(cols):
        new_intensity[i, j] = new_intensity_uint8[i, j] / 255.0

new_hsv = np.copy(img_hsv) #creates the new hsv and converts it back to RGB
new_hsv[:, :, 2] = new_intensity

new_rgb = color.hsv2rgb(new_hsv)

#display input and output image
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(new_rgb)
plt.title('Output Image')
plt.axis('off')

plt.show()

#io.imsave('output_image.jpg', (new_rgb * 255).astype(np.uint8))
