import numpy as np
from skimage import io
import matplotlib.pyplot as plt

K = 256  # Number of gray levels (0 to 255)

A = 'Low-light-img1-256.jpg'  #read image
f = io.imread(A, as_gray = True)

if f.dtype != np.uint8:
    f = (f * 255).astype(np.uint8)
M, N = f.shape  

hg = np.ones(K, dtype=int) * (M * N // K) #desired histogram
remainder = (M * N) - np.sum(hg)
hg[-1] += remainder  

hf = np.zeros(K, dtype=int) #compute the input image histogram
for m in range(M):
    for n in range(N):
        k_val = f[m, n]
        hf[k_val] += 1

cf = np.zeros(K, dtype=int) #compute the cumlative hist
cf[0] = hf[0]
for k in range(1, K):
    cf[k] = cf[k - 1] + hf[k]

cg = np.zeros(K, dtype=int) #cum hist for hg
cg[0] = hg[0]
for k in range(1, K):
    cg[k] = cg[k - 1] + hg[k]

qf = np.zeros(K, dtype=int) #index arrays
qg = np.zeros(K, dtype=int)
for k in range(K):
    qf[k] = cf[k] - hf[k]
    qg[k] = cg[k] - hg[k]


P = np.zeros((M * N, 5), dtype=int) #pixel array P
qf_copy = qf.copy()
for m in range(M):
    for n in range(N):
        v = f[m, n]
        u = qf_copy[v]
        P[u, 0] = m
        P[u, 1] = n
        P[u, 2] = v
        qf_copy[v] += 1

g1 = np.zeros((M, N), dtype=np.uint8) #monotonicity constraint
for k in range(K):
    start = qg[k]
    end = qg[k] + hg[k]
    for u in range(start, end):
        m_idx = P[u, 0]
        n_idx = P[u, 1]
        g1[m_idx, n_idx] = k
        P[u, 3] = k


g2 = np.zeros((M, N), dtype=np.uint8) #computes monotonicity and equality constraints

qf_eq = np.zeros(K, dtype=int)
qg_eq = np.zeros(K, dtype=int)
for k in range(K):
    qf_eq[k] = cf[k] - hf[k]
    qg_eq[k] = cg[k] - hg[k]


for k in range(K):
    sum_val = 0
    for u in range(qf_eq[k], qf_eq[k] + hf[k]):
        sum_val += P[u, 3]
    if hf[k] > 0:
        avg_val = int(round(sum_val / hf[k]))
    else:
        avg_val = 0
    for u in range(qf_eq[k], qf_eq[k] + hf[k]):
        m_idx = P[u, 0]
        n_idx = P[u, 1]
        g2[m_idx, n_idx] = avg_val
        P[u, 4] = avg_val  

fig1, ax1 = plt.subplots(1, 2, figsize=(12, 6))

ax1[0].imshow(f, cmap='gray')
ax1[0].set_title("Input Image")
ax1[0].axis("off")

ax1[1].imshow(g2, cmap='gray')
ax1[1].set_title("Output Image \n(Monotonicity and Equality Constraint)")
ax1[1].axis("off")

plt.tight_layout()
plt.show()
