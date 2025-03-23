import numpy as np
import matplotlib.pyplot as plt

image_path = 'dance2gray(1).jpg' #Read the image in grayscal
img = plt.imread(image_path)
if len(img.shape) == 3:  #if the image is RGB, convert to grayscale
    img = np.mean(img, axis=2)
img = img.astype(np.float32)

radius = 3.0 #initializing parameters
M, N = img.shape
kernel_size = 11
P = Q = 5 

h = np.zeros((kernel_size, kernel_size)) #cylindrical kernal
center = kernel_size // 2
for m in range(kernel_size):
    for n in range(kernel_size):
        x = m - center
        y = n - center
        if x**2 + y**2 <= radius**2:
            h[m, n] = 1.0 / (np.pi * radius**2)


output = np.zeros_like(img) #compute convolution with reflection
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
            for q in range(-Q, Q + 1):
                col_idx = n - q
                if col_idx < 0:
                    l = abs(col_idx)
                elif col_idx >= N:
                    l = 2 * (N - 1) - col_idx
                else:
                    l = col_idx
                kernel_p = p + P
                kernel_q = q + Q
                sum_val += h[kernel_p, kernel_q] * img[k, l]
        output[m, n] = sum_val


output = np.clip(output, 0, 255) #clip the output to only have valid pixel values

# Plot the input and output images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(output, cmap='gray')
plt.title('Cylindrical Blurred Image')
plt.axis('off')

plt.show()
