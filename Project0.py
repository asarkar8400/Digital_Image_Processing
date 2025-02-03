import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt

# Read in the RGB image
A = io.imread('Pikachu.jpg') # This is a 512 x 512 Image

# Extract color channels.
RC = A[:,:,0] # Red channel
GC = A[:,:,1] # Green channel
BC = A[:,:,2] # Blue channel

#Display each channel
plt.imshow(RC) 
plt.show()
plt.imshow(GC)
plt.show()
plt.imshow(BC)
plt.show()

# Convert each channel to unsigned 16-bit integers to handle overflow
# and take the average of the channels to convert image to grayscale
AG = (RC.astype(np.uint16) + GC.astype(np.uint16) + BC.astype(np.uint16)) / 3
# Convert back to unsigned 8-bit integer
AG = AG.astype(np.uint8)  

plt.imshow(AG, cmap=plt.cm.gray)  # Map image to grayscale and display
plt.show()

# Computes histogram for each image
hist_RC, _ = np.histogram(RC, bins=256, range=(0, 255)) 
hist_GC, _ = np.histogram(GC, bins=256, range=(0, 255))
hist_BC, _ = np.histogram(BC, bins=256, range=(0, 255))
hist_AG, _ = np.histogram(AG, bins=256, range=(0, 255))

# Plot histograms
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.bar(range(256), hist_RC, color='red', alpha=0.6)
plt.title("Histogram of Red Channel")

plt.subplot(2, 2, 2)
plt.bar(range(256), hist_GC, color='green', alpha=0.6)
plt.title("Histogram of Green Channel")

plt.subplot(2, 2, 3)
plt.bar(range(256), hist_BC, color='blue', alpha=0.6)
plt.title("Histogram of Blue Channel")

plt.subplot(2, 2, 4)
plt.bar(range(256), hist_AG, color='black', alpha=0.6)
plt.title("Histogram of Grayscale Image")

plt.tight_layout()
plt.show()

# Binarization
# Input a brightness threshold value (TB)
while True:
    try:
        TB = int(input("Enter threshold brightness (0-255): "))
        if 0 <= TB <= 255:
            break  # Valid input, exit the loop
        else:
            print("Invalid input! Please enter a number between 0 and 255.")
    except ValueError:
        print("Invalid input! Please enter a valid integer between 0 and 255.")

# Create binary image using threshold
AB = np.where(AG < TB, 0, 255).astype(np.uint8)

# Display binary image
plt.imshow(AB, cmap='gray')
plt.title(f"Binary Image (Threshold Brightness = {TB})")
plt.show()

# Edge Detection
# Input an edge detection threshold value (TE)
while True:
    try:
        TE = int(input("Enter edge detection threshold (TE): "))
        if TE >= 0:
            break  # Valid input, exit loop
        else:
            print("Invalid input! Please enter a non-negative integer.")
    except ValueError:
        print("Invalid input! Please enter a valid integer.")

# Compute Gx (Gradient along rows)
Gx = np.zeros_like(AG, dtype=np.float32)  # Initialize Gx with all 0s
Gx[:, :-1] = AG[:, 1:] - AG[:, :-1]       # Gx(m,n) = AG(m,n+1)-AG(m,n) 
Gx[:, -1] = 0                             # Set last column to 0

# Compute Gy (Gradient along columns)
Gy = np.zeros_like(AG, dtype=np.float32) # Initialize Gy with all 0s
Gy[:-1, :] = AG[1:, :] - AG[:-1, :]   	 # Gy(m,n) = AG(m+1,n)-AG(m,n) 
Gy[-1, :] = 0                            # Set last row to 0

# Compute gradient magnitude GM
GM = np.sqrt(Gx**2 + Gy**2)              # GM(m,n)= √(Gx(m,n)^2 + Gy(m,n)^2)  

# AE(m,n)=255 if GM(m,n) > TE; else AE(m,n)=0.
AE = np.where(GM > TE, 255, 0).astype(np.uint8)

# Display edge detection result
plt.imshow(AE, cmap='gray')
# Turns completely black at TE > 360
plt.title(f"Edge Detection Image (Threshold = {TE})") 
plt.axis('off')
plt.show()

# Image Pyramid 
# Downsample by a factor of 2 using the average of 2x2 blocks.
def downsampleBy2(image):
    m, n = image.shape
    # Convert to unsigned 16-bit integer to avoid overflow
    # then average and convert back to uint8.
    return ((image[0::2, 0::2].astype(np.uint16) + image[0::2, 1::2].astype(np.uint16) +
             image[1::2, 0::2].astype(np.uint16) + image[1::2, 1::2].astype(np.uint16)) / 4).astype(np.uint8)

# Compute each pyramid level
AG2 = downsampleBy2(AG)   # 1/2 the size of AG
AG4 = downsampleBy2(AG2)  # 1/4 the size of AG
AG8 = downsampleBy2(AG4)  # 1/8 the size of AG

# Display the pyramid images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(AG2, cmap='gray')
plt.title("Image Pyramid: AG2")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(AG4, cmap='gray')
plt.title("Image Pyramid: AG4")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(AG8, cmap='gray')
plt.title("Image Pyramid: AG8")
plt.axis('off')

plt.tight_layout()
plt.show()
