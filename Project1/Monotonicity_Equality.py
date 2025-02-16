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

qf_eq = np.zeros(K, dtype=int) #monotonicity and equality constraints
qg_eq = np.zeros(K, dtype=int)
for k in range(K):
    qf_eq[k] = cf[k] - hf[k]
    qg_eq[k] = cg[k] - hg[k]

g2 = np.zeros((M, N), dtype=np.uint8)
for k in range(K):
    s = 0
    if hf[k] > 0:
        for u in range(qf_eq[k], qf_eq[k] + hf[k]):
            s += P[u, 3]
        v = int(round(s / hf[k]))
    else:
        v = 0
    for u in range(qf_eq[k], qf_eq[k] + hf[k]):
        m_idx = P[u, 0]
        n_idx = P[u, 1]
        g2[m_idx, n_idx] = v
        P[u, 4] = v

#display everything

fig1, ax1 = plt.subplots(1, 3, figsize=(18, 6))

ax1[0].imshow(f, cmap='gray')
ax1[0].set_title("Input Image f(m,n)")
ax1[0].axis("off")

ax1[1].imshow(g1, cmap='gray')
ax1[1].set_title("Output Image g1\n(Monotonicity Constraint)")
ax1[1].axis("off")

ax1[2].imshow(g2, cmap='gray')
ax1[2].set_title("Output Image g2\n(Monotonicity + Equality Constraint)")
ax1[2].axis("off")

plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(1, 3, figsize=(18, 6))

ax2[0].hist(f.ravel(), bins=256, range=[0, 256], color='gray')
ax2[0].set_title("Histogram of Input Image f")

ax2[1].hist(g1.ravel(), bins=256, range=[0, 256], color='gray')
ax2[1].set_title("Histogram of Output Image g1")

ax2[2].hist(g2.ravel(), bins=256, range=[0, 256], color='gray')
ax2[2].set_title("Histogram of Output Image g2")

plt.tight_layout()
plt.show()
