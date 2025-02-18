import numpy as np
from skimage import io
import matplotlib.pyplot as plt

K = 256  # of gray levels

B = 'Medium-light-img3-256.jpg'  
medium = io.imread(B, as_gray=True) #read the medium light image
if medium.dtype != np.uint8:
    medium = (medium * 255).astype(np.uint8)
M_med, N_med = medium.shape

hg = np.zeros(K, dtype=int)         #histogram computed from medium-light image
for m in range(M_med):
    for n in range(N_med):
        value = medium[m, n]
        hg[value] += 1


cg = np.zeros(K, dtype=int)         #histogram for hg
cg[0] = hg[0]
for k in range(1, K):
    cg[k] = cg[k - 1] + hg[k]


A = 'Low-light-img1-256.jpg'        #read first low-light image
f1 = io.imread(A, as_gray=True)
if f1.dtype != np.uint8:
    f1 = (f1 * 255).astype(np.uint8)
M, N = f1.shape  

hf1 = np.zeros(K, dtype=int)       #compute histogram
for m in range(M):
    for n in range(N):
        k_val = f1[m, n]
        hf1[k_val] += 1

cf1 = np.zeros(K, dtype=int)      #compute the cumulative histogram 
cf1[0] = hf1[0]
for k in range(1, K):
    cf1[k] = cf1[k - 1] + hf1[k]

qf1 = np.zeros(K, dtype=int)  
qg1 = np.zeros(K, dtype=int)
for k in range(K):
    qf1[k] = cf1[k] - hf1[k]
    qg1[k] = cg[k] - hg[k]

P1 = np.zeros((M * N, 5), dtype=int)    #pixel array for dirst image
qf1_copy = qf1.copy()
for m in range(M):
    for n in range(N):
        v = f1[m, n]
        u = qf1_copy[v]
        P1[u, 0] = m
        P1[u, 1] = n
        P1[u, 2] = v
        qf1_copy[v] += 1

g1_1 = np.zeros((M, N), dtype=np.uint8)  #monotonicity constraint
for k in range(K):
    start = qg1[k]
    end = qg1[k] + hg[k]
    for u in range(start, end):
        m_idx = P1[u, 0]
        n_idx = P1[u, 1]
        g1_1[m_idx, n_idx] = k
        P1[u, 3] = k

g2_1 = np.zeros((M, N), dtype=np.uint8)  #monotonicity and equality constraints 

qf_eq1 = np.zeros(K, dtype=int)
qg_eq1 = np.zeros(K, dtype=int)
for k in range(K):
    qf_eq1[k] = cf1[k] - hf1[k]
    qg_eq1[k] = cg[k] - hg[k]

for k in range(K):
    sum_val = 0
    for u in range(qf_eq1[k], qf_eq1[k] + hf1[k]):
        sum_val += P1[u, 3]
    if hf1[k] > 0:
        avg_val = int(round(sum_val / hf1[k]))
    else:
        avg_val = 0
    for u in range(qf_eq1[k], qf_eq1[k] + hf1[k]):
        m_idx = P1[u, 0]
        n_idx = P1[u, 1]
        g2_1[m_idx, n_idx] = avg_val
        P1[u, 4] = avg_val  


B = 'Low-light-img2-256.jpg'            #second image
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

g1_2 = np.zeros((M2, N2), dtype=np.uint8)  
for k in range(K):
    start = qg2[k]
    end = qg2[k] + hg[k]
    for u in range(start, end):
        m_idx = P2[u, 0]
        n_idx = P2[u, 1]
        g1_2[m_idx, n_idx] = k
        P2[u, 3] = k

g2_2 = np.zeros((M2, N2), dtype=np.uint8)  #output for imagfe 2

qf_eq2 = np.zeros(K, dtype=int)
qg_eq2 = np.zeros(K, dtype=int)
for k in range(K):
    qf_eq2[k] = cf2[k] - hf2[k]
    qg_eq2[k] = cg[k] - hg[k]

for k in range(K):
    sum_val = 0
    for u in range(qf_eq2[k], qf_eq2[k] + hf2[k]):
        sum_val += P2[u, 3]
    if hf2[k] > 0:
        avg_val = int(round(sum_val / hf2[k]))
    else:
        avg_val = 0
    for u in range(qf_eq2[k], qf_eq2[k] + hf2[k]):
        m_idx = P2[u, 0]
        n_idx = P2[u, 1]
        g2_2[m_idx, n_idx] = avg_val
        P2[u, 4] = avg_val  

#display each image input and output seperately 

fig1, ax1 = plt.subplots(1, 2, figsize=(12, 6))
ax1[0].imshow(f1, cmap='gray')
ax1[0].set_title("Input Image 1")
ax1[0].axis("off")
ax1[1].imshow(g2_1, cmap='gray')
ax1[1].set_title("Output Image 1")
ax1[1].axis("off")
plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6))
ax2[0].imshow(f2, cmap='gray')
ax2[0].set_title("Input Image 2")
ax2[0].axis("off")
ax2[1].imshow(g2_2, cmap='gray')
ax2[1].set_title("Output Image 2")
ax2[1].axis("off")
plt.tight_layout()
plt.show()
