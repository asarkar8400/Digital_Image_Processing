import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open("dance2gray256.jpg").convert("L") #load image and convert to fp
f = np.array(image, dtype=np.float64)
M, N = f.shape  #256x256 image

print("Loaded image with dimensions:", M, "x", N)

#create index arrays
m = np.arange(M)
u = np.arange(M)
n = np.arange(N)
v = np.arange(N)

P = (1 / M) * np.exp(-1j * 2 * np.pi * np.outer(u, m) / M) #create MxM matrix
Q = (1 / N) * np.exp(-1j * 2 * np.pi * np.outer(n, v) / N) #create NxN matrix

# Compute F = P * f * Q (using two matrix multiplications)
F = P.dot(f).dot(Q)
print("DFT computed. F shape:", F.shape)

#Frequency-Domain Filtering
H = np.ones((M, N), dtype=np.float64)

# Create coordinate grids for frequency indices (low frequencies are at the corners)
U, V = np.meshgrid(np.arange(M), np.arange(N), indexing='ij')

H[(U < 10) & (V < 10)] = 0.5 #attenuation
H[(U >= (M - 10)) & (V < 10)] = 0.5
H[(U < 10) & (V >= (N - 10))] = 0.5
H[(U >= (M - 10)) & (V >= (N - 10))] = 0.5

#multiply the DFT by the filter
G = F * H

P_inv = np.exp(1j * 2 * np.pi * np.outer(u, m) / M)
Q_inv = np.exp(1j * 2 * np.pi * np.outer(n, v) / N)

#compute the idft
g_complex = P_inv.dot(G).dot(Q_inv)
g = np.real(g_complex)

g_clipped = np.clip(g, 0, 255)
g_uint8 = g_clipped.astype(np.uint8)

#plot everything
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(f, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(g_uint8, cmap='gray')
plt.title("Filtered Image")
plt.axis("off")

plt.show()
