import cv2
import numpy as np
import matplotlib.pyplot as plt


I1 = cv2.imread('food1.jpg').astype(np.float32) / 255.0 #read image
I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)  #convert to RGB
M, N, C = I1.shape

s = float(input("Enter scale value (0.1 to 3.0): ")) #read s the scaling factor
A = np.array([[s, 0.0], [0.0, s]])  #scale the transformation matrix
T = np.array([0, 0])  

A_inv = np.linalg.inv(A)

corners = np.array([[0, 0, 1], [0, N, 1], [M, 0, 1], [M, N, 1]]).T
new_corners = A @ corners[:2, :] + T[:, np.newaxis]

xmin, ymin = np.floor(new_corners.min(axis=1)).astype(int)
xmax, ymax = np.ceil(new_corners.max(axis=1)).astype(int)
Mp, Np = xmax - xmin + 1, ymax - ymin + 1

I_output = np.zeros((Mp, Np, C), dtype=np.float32)

for i in range(xmin, xmax):
    for j in range(ymin, ymax):
        p = A_inv @ ([i, j] - T)
        x0, y0 = p[0], p[1]
        
        if 0 <= x0 < M and 0 <= y0 < N:
            x_shifted, y_shifted = i - xmin, j - ymin
            x0_int, y0_int = int(np.floor(x0)), int(np.floor(y0))
            x1_int, y1_int = min(x0_int + 1, M - 1), min(y0_int + 1, N - 1)
            
            dx, dy = x0 - x0_int, y0 - y0_int
            
            for c in range(C):
                s1 = I1[x0_int, y0_int, c] * (1 - dx) * (1 - dy)
                s2 = I1[x0_int, y1_int, c] * (1 - dx) * dy
                s3 = I1[x1_int, y0_int, c] * dx * (1 - dy)
                s4 = I1[x1_int, y1_int, c] * dx * dy
                
                I_output[x_shifted, y_shifted, c] = s1 + s2 + s3 + s4

plt.figure(figsize=(10,5)) #display original image
plt.subplot(1,2,1)
plt.imshow(I1)
plt.title('Original Image')

plt.subplot(1,2,2)
plt.imshow(np.clip(I_output, 0, 1))
plt.title(f'Scaled Image (s = {s})') #display new image
plt.show()
