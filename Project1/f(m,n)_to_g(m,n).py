import numpy as np
from skimage import io
import matplotlib.pyplot as plt

input_img = io.imread('Low-light-img1-256.jpg', as_gray=True) #reads image and converts to grayscale

if input_img.dtype != np.uint8: #if the image isnt a 8-bit unsigned int then convert it
    input_img = (input_img * 255).astype(np.uint8)

M, N = input_img.shape

hg = np.ones(256, dtype=int) #using a simple uniform jistrogram hg(k)
hg = hg * (M * N // 256)
hg[-1] += (M * N - np.sum(hg)) 

hf = np.zeros(256, dtype=int) #compute histogram of input image
for i in range(M):
    for j in range(N):
        intensity = input_img[i, j]
        hf[intensity] += 1

F = np.zeros(256, dtype=int) 
F[0] = hf[0]
for k in range(1, 256):
    F[k] = F[k - 1] + hf[k]

G = np.zeros(256, dtype=int)
G[0] = hg[0]
for k in range(1, 256):
    G[k] = G[k - 1] + hg[k]

mapping = np.zeros(256, dtype=np.uint8) #map input gray-levels to the output's
for k in range(256):
    min_diff = abs(F[k] - G[0])
    mapping_val = 0
    for j in range(1, 256):
        diff = abs(F[k] - G[j])
        if diff < min_diff:
            min_diff = diff
            mapping_val = j
    mapping[k] = mapping_val

#monotonicity 
for k in range(1, 256):
    if mapping[k] < mapping[k - 1]:
        mapping[k] = mapping[k - 1]

output_img = np.zeros((M, N), dtype=np.uint8) #creates the output image
for i in range(M):
    for j in range(N):
        p = input_img[i, j]
        output_img[i, j] = mapping[p]

#plot images and their histograms
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 6))
axes1[0].imshow(input_img, cmap='gray')
axes1[0].set_title('Input Image')
axes1[0].axis('off')
axes1[1].imshow(output_img, cmap='gray')
axes1[1].set_title('Output Image')
axes1[1].axis('off')
plt.tight_layout()
plt.show()

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))
axes2[0].hist(input_img.ravel(), bins=256, range=[0, 256], color='gray')
axes2[0].set_title('Histogram of Input Image')
axes2[1].hist(output_img.ravel(), bins=256, range=[0, 256], color='gray')
axes2[1].set_title('Histogram of Output Image')
plt.tight_layout()
plt.show()
