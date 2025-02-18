import numpy as np
from skimage import io
import matplotlib.pyplot as plt

K = 256  # of gray levels

use_medium = True

if use_medium:
    A = 'Medium-light-img3-256.jpg'  
    medium = io.imread(A, as_gray=True)  #read the medium light image
    if medium.dtype != np.uint8:
        medium = (medium * 255).astype(np.uint8)
    M_medium, N_medium = medium.shape
    hg = np.zeros(K, dtype=int)  #histogram computed from medium-light image
    for m in range(M_medium):
        for n in range(N_medium):
            value = medium[m, n]
            hg[value] += 1
else:                                   #ONLY FOR TESTING
    A = 'Low-light-img1-256.jpg'
    temp = io.imread(A, as_gray=True)  
    if temp.dtype != np.uint8:
        temp = (temp * 255).astype(np.uint8)
    M, N = temp.shape  
    hg = np.ones(K, dtype=int) * (M * N // K)  
    remainder = (M * N) - np.sum(hg)
    hg[-1] += remainder  


A = 'Low-light-img1-256.jpg'  #read first low-light image
f1 = io.imread(A, as_gray=True)
if f1.dtype != np.uint8:
    f1 = (f1 * 255).astype(np.uint8)
M, N = f1.shape  

hf1 = np.zeros(K, dtype=int)  #compute histogram 
for m in range(M):
    for n in range(N):
        k_val = f1[m, n]
        hf1[k_val] += 1

cf1 = np.zeros(K, dtype=int)  #compute the cumulative histogram 
cf1[0] = hf1[0]
for k in range(1, K):
    cf1[k] = cf1[k - 1] + hf1[k]

cg = np.zeros(K, dtype=int)  #histogram for hg 
cg[0] = hg[0]
for k in range(1, K):
    cg[k] = cg[k - 1] + hg[k]

qf = np.zeros(K, dtype=int)  
qg = np.zeros(K, dtype=int)
for k in range(K):
    qf[k] = cf1[k] - hf1[k]
    qg[k] = cg[k] - hg[k]

P = np.zeros((M * N, 5), dtype=int)  #pixel array for dirst image
qf_copy = qf.copy()
for m in range(M):
    for n in range(N):
        v = f1[m, n]
        u = qf_copy[v]
        P[u, 0] = m
        P[u, 1] = n
        P[u, 2] = v
        qf_copy[v] += 1

g1 = np.zeros((M, N), dtype=np.uint8)  #monotonicity constraint
for k in range(K):
    start = qg[k]
    end = qg[k] + hg[k]
    for u in range(start, end):
        m_idx = P[u, 0]
        n_idx = P[u, 1]
        g1[m_idx, n_idx] = k
        P[u, 3] = k


B = 'Low-light-img2-256.jpg'  #second image
f2 = io.imread(B, as_gray=True)
if f2.dtype != np.uint8:
    f2 = (f2 * 255).astype(np.uint8)
M2, N2 = f2.shape  

hf2 = np.zeros(K, dtype=int)  
for m in range(M2):
    for n in range(N2):
        k_val = f2[m, n]
        hf2[k_val] += 1

cf2 = np.zeros(K, dtype=int)  
cf2[0] = hf2[0]
for k in range(1, K):
    cf2[k] = cf2[k - 1] + hf2[k]

qf2 = np.zeros(K, dtype=int) 
qg2 = np.zeros(K, dtype=int)
for k in range(K):
    qf2[k] = cf2[k] - hf2[k]
    qg2[k] = cg[k] - hg[k]  

P2 = np.zeros((M2 * N2, 5), dtype=int)  
qf2_copy = qf2.copy()
for m in range(M2):
    for n in range(N2):
        v = f2[m, n]
        u = qf2_copy[v]
        P2[u, 0] = m
        P2[u, 1] = n
        P2[u, 2] = v
        qf2_copy[v] += 1

g2 = np.zeros((M2, N2), dtype=np.uint8) #image 2 output
for k in range(K):
    start = qg2[k]
    end = qg2[k] + hg[k]
    for u in range(start, end):
        m_idx = P2[u, 0]
        n_idx = P2[u, 1]
        g2[m_idx, n_idx] = k
        P2[u, 3] = k

#display each image input and output seperately 

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(f1, cmap='gray')
plt.title("Input Image 1")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(g1, cmap='gray')
plt.title("Output Image 1")
plt.axis("off")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(f2, cmap='gray')
plt.title("Input Image 2")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(g2, cmap='gray')
plt.title("Output Image 2")
plt.axis("off")
plt.tight_layout()
plt.show()
