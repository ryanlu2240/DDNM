import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.io
from functions.conv_util import *
# # Load the HQ image
# hq_image = cv2.imread('/eva_data2/shlu2240/DDNM/exp/datasets/old_celeba_hq/face/00258.png')

# # Define the downsampling factor (you can adjust this)
# downsampling_factor = 4  # Adjust as needed

# # Generate a random Gaussian kernel
# kernel_size = 13  # Adjust the kernel size as needed
# std_deviation = random.uniform(1, 2)  # Random standard deviation within a range
# random_kernel = cv2.getGaussianKernel(kernel_size, std_deviation)
# random_kernel = random_kernel.dot(random_kernel.T)

# # Normalize the kernel to make it sum to 1
# random_kernel /= np.sum(random_kernel)

# # Convolve the image with the random kernel to downsample it
# lq_image = cv2.filter2D(hq_image, -1, random_kernel)

# # Save the LQ image and the kernel
# cv2.imwrite('LQ.png', lq_image)
# # np.savetxt('random_kernel.txt', random_kernel, delimiter=' ')
# # Display the LQ image and the random kernel
# plt.imshow(random_kernel, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.title('Gaussian Kernel Heatmap')
# plt.savefig('random_kernel_heatmap.png')
# mat_data = scipy.io.loadmat('/eva_data2/shlu2240/FKP/data/datasets/Test/DIPFKP_gt_k_x4/102061.mat')
# print(mat_data['Kernel'].shape)

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torch

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torch

# Define the kernel size and standard deviations
kernel_size = 19
std_deviation_x = 3.0
std_deviation_y = 1.7  # Different standard deviation for Y
tilt_angle = np.random.uniform(0, 180)  # Random tilt angle in degrees

# Create a 2D Gaussian kernel with different std deviations and tilt
x = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
y = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
x, y = np.meshgrid(x, y)

# Apply the tilt to the kernel
rot_matrix = np.array([[np.cos(np.radians(tilt_angle)), -np.sin(np.radians(tilt_angle))],
                       [np.sin(np.radians(tilt_angle)), np.cos(np.radians(tilt_angle))]])

x_rotated = x * rot_matrix[0, 0] + y * rot_matrix[0, 1]
y_rotated = x * rot_matrix[1, 0] + y * rot_matrix[1, 1]

kernel = np.exp(-0.5 * ((x_rotated / std_deviation_x)**2 + (y_rotated / std_deviation_y)**2))
kernel /= kernel.sum()

# Save the kernel as a heatmap
plt.imshow(kernel, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Gaussian Kernel Heatmap')
plt.savefig('gaussian_kernel_heatmap.png')

# Save the kernel in a .mat file
scipy.io.savemat('kernel.mat', {'Kernel': kernel})

from PIL import Image
import cv2
downsample_factor = 4
# Load the HQ image
hq_image = Image.open('/eva_data2/shlu2240/DDNM/exp/datasets/old_celeba_hq/face/00258.png')
hq_image = hq_image.resize((256, 256))
hq_image.save('HQ.png')

# Convert the HQ image to a NumPy array and a PyTorch tensor
hq_image_np = np.array(hq_image)
hq_image = torch.from_numpy(hq_image_np).permute(2, 0, 1).float()

# Load the kernel from the .mat file
mat_data = scipy.io.loadmat('/eva_data2/shlu2240/DDNM/kernel.mat')
kernel_size = mat_data['Kernel'].shape[0]
kernel = torch.from_numpy(mat_data['Kernel']).float()

# Perform the degradation using PyTorch Conv2D
LQ = convolution2d(hq_image.unsqueeze(0), kernel, stride=downsample_factor, padding=kernel_size // 2).squeeze()
print(LQ.shape)
# Convert the LQ image back to a NumPy array for saving
lq_image_np = LQ.permute(1, 2, 0).numpy().astype(np.uint8)

# Save the LQ image using Pillow (PIL)
lq_image_pil = Image.fromarray(lq_image_np)
lq_image_pil.save('LQ.png')