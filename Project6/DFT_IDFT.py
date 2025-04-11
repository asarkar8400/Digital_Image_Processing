import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open("dance2gray256.jpg").convert("L") #read image and convert to fp
f = np.array(image, dtype=np.float64)
M, N = f.shape  # M and N should be 256

#create index arrays
m = np.arange(M)
u = np.arange(M)
n = np.arange(N)
v = np.arange(N)

#P[u][m] = (1/M)*exp(-j*2π*u*m/M)
P = (1.0 / M) * np.exp(-1j * 2 * np.pi * np.outer(u, m) / M)

#Q[n][v] = (1/N)*exp(-j*2π*v*n/N)
Q = (1.0 / N) * np.exp(-1j * 2 * np.pi * np.outer(n, v) / N)

#compute the DFT
F = P.dot(f).dot(Q)

#P'[u][m] = exp(j*2π*u*m/M)
P_inv = np.exp(1j * 2 * np.pi * np.outer(u, m) / M)

#Q'[n][v] = exp(j*2π*v*n/N)
Q_inv = np.exp(1j * 2 * np.pi * np.outer(n, v) / N)

#f' = P' * F * Q'
f_complex = P_inv.dot(F).dot(Q_inv)
f_recovered = np.real(f_complex)  

f_recovered = np.clip(f_recovered, 0, 255)
f_recovered_uint8 = f_recovered.astype(np.uint8)

#plot everyhting
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(f, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(f_recovered_uint8, cmap='gray')
plt.title("Recovered Image (IDFT)")
plt.axis("off")

plt.show()

