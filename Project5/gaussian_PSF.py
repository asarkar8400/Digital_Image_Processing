import numpy as np
import matplotlib.pyplot as plt

image_path = 'dance2gray(1).jpg' #Read the image in grayscal
img = plt.imread(image_path)
if len(img.shape) == 3:  #if the image is RGB, convert to grayscale
    img = np.mean(img, axis=2)
img = img.astype(np.float32)

sigma = 3.0 #initializing parameters
M, N = img.shape
kernel_size = 21
Q = P = 10  

#create a 1D gaussian kernel
n_vals = np.arange(-10, 11)
g = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-n_vals**2 / (2 * sigma**2))
g_sum = g.sum()
g_normalized = g / g_sum

output_h = np.zeros_like(img)   #compute along columns
for m in range(M):
    for n in range(N):
        sum_val = 0.0
        for q in range(-Q, Q + 1):
            col_idx = n - q
            if col_idx < 0:
                l = abs(col_idx)
            elif col_idx >= N:
                l = 2 * (N - 1) - col_idx
            else:
                l = col_idx
            kernel_idx = q + Q
            sum_val += g_normalized[kernel_idx] * img[m, l]
        output_h[m, n] = sum_val

output_v = np.zeros_like(output_h) #compute along rows
for m in range(M):
    for n in range(N):
        sum_val = 0.0
        for p in range(-P, P + 1):
            row_idx = m - p
            if row_idx < 0:
                k = abs(row_idx)
            elif row_idx >= M:
                k = 2 * (M - 1) - row_idx
            else:
                k = row_idx
            kernel_idx = p + P
            sum_val += g_normalized[kernel_idx] * output_h[k, n]
        output_v[m, n] = sum_val

output_v = np.clip(output_v, 0, 255)    #clip the output to only have valid pixel values

#Plot the input and output images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(output_v, cmap='gray')
plt.title('Gaussian Blurred Image')
plt.axis('off')

plt.show()
