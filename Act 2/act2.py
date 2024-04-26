import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from matplotlib import cm
import cv2
from skimage.color import rgb2hsv
from skimage.io import imread, imshow

img = cv2.imread('bags.png')
plt.imshow(img)
plt.show()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

r, g, b = cv2.split(img)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection = "3d")
pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
norm = colors.Normalize(vmin = -1., vmax = 1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors = pixel_colors, marker = ".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()

hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_img)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection = "3d")
axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors = pixel_colors, marker = ".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()


#AMARILLO
light_yellow = (255, 254, 30)
dark_yellow = (166, 130, 1)

from matplotlib.colors import hsv_to_rgb
lo_square = np. full((10, 10, 3), light_yellow, dtype = np.uint8) / 255.0
do_square = np. full((10, 10, 3), dark_yellow, dtype = np.uint8) / 255.0
plt.subplot(1, 2, 1)
plt.imshow(lo_square)
plt.subplot(1, 2, 2)
plt.imshow(do_square)
plt.show()

mask = cv2.inRange(img, dark_yellow, light_yellow)
result_amarillo = cv2.bitwise_and(img, img, mask = mask)
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap = "gray")
plt.subplot(1, 2, 2)
plt.imshow(result_amarillo)
plt.show()

#AZUL
light_azul = (60, 108, 218)
dark_azul = (13, 24, 97)

from matplotlib.colors import hsv_to_rgb
lo_square = np. full((10, 10, 3), light_azul, dtype = np.uint8) / 255.0
do_square = np. full((10, 10, 3), dark_azul, dtype = np.uint8) / 255.0
plt.subplot(1, 2, 1)
plt.imshow(lo_square)
plt.subplot(1, 2, 2)
plt.imshow(do_square)
plt.show()

mask = cv2.inRange(img, dark_azul, light_azul)
result_azul = cv2.bitwise_and(img, img, mask = mask)
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap = "gray")
plt.subplot(1, 2, 2)
plt.imshow(result_azul)
plt.show()

#VERDE
light_verde = (40, 165, 144)
dark_verde = (11, 47, 28)

from matplotlib.colors import hsv_to_rgb
lo_square = np. full((10, 10, 3), light_verde, dtype = np.uint8) / 255.0
do_square = np. full((10, 10, 3), dark_verde, dtype = np.uint8) / 255.0
plt.subplot(1, 2, 1)
plt.imshow(lo_square)
plt.subplot(1, 2, 2)
plt.imshow(do_square)
plt.show()

mask = cv2.inRange(img, dark_verde, light_verde)
result_verde = cv2.bitwise_and(img, img, mask = mask)
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap = "gray")
plt.subplot(1, 2, 2)
plt.imshow(result_verde)
plt.show()

#NARANJA
light_naranja = (254, 138, 81)
dark_naranja = (172, 54, 19)

from matplotlib.colors import hsv_to_rgb
lo_square = np. full((10, 10, 3), light_naranja, dtype = np.uint8) / 255.0
do_square = np. full((10, 10, 3), dark_naranja, dtype = np.uint8) / 255.0
plt.subplot(1, 2, 1)
plt.imshow(lo_square)
plt.subplot(1, 2, 2)
plt.imshow(do_square)
plt.show()

mask = cv2.inRange(img, dark_naranja, light_naranja)
result_naranja = cv2.bitwise_and(img, img, mask = mask)
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap = "gray")
plt.subplot(1, 2, 2)
plt.imshow(result_naranja)
plt.show()

#TODOS
todos = result_naranja + result_amarillo + result_azul + result_verde
plt.imshow(todos)

# ---------------------------------------------------------------------------------------------------------------
h, s, v = cv2.split(hsv_img)
fig, ax = plt.subplots(1, 3, figsize = (15, 5))
ax[0].imshow(h, cmap = 'hsv')
ax[0].set_title('hue')
ax[1].imshow(s, cmap = 'hsv')
ax[1].set_title('transparency')
ax[2].imshow(v, cmap = 'hsv')
ax[2].set_title('value')
fig.colorbar(imshow(hsv_img[:, :, 2], cmap = 'hsv'))
#fig.tight_layout()
plt.show()

#VERDE
hsv_img = rgb2hsv(img)
lower_mask = hsv_img[:, :, 0] > 0.2
upper_mask = hsv_img[:, :, 0] < 0.5
saturation_mask = hsv_img[:, :, 1] > 0.4

mask2 = upper_mask * lower_mask * saturation_mask
red = img[:, :, 0] * mask2
green = img[:, :, 1] * mask2
blue = img[:, :, 2] * mask2
img_masked = np.dstack((red, green, blue))
plt.imshow(img_masked)
plt.show()

#AZUL
hsv_img = rgb2hsv(img)
lower_mask = hsv_img[:, :, 0] > 0.5
upper_mask = hsv_img[:, :, 0] < 0.7
saturation_mask = hsv_img[:, :, 1] > 0.4

mask2 = upper_mask * lower_mask * saturation_mask
red = img[:, :, 0] * mask2
green = img[:, :, 1] * mask2
blue = img[:, :, 2] * mask2
img_masked = np.dstack((red, green, blue))
plt.imshow(img_masked)
plt.show()

#NARANJA
hsv_img = rgb2hsv(img)
lower_mask = hsv_img[:, :, 0] > 0.0
upper_mask = hsv_img[:, :, 0] < 0.08
saturation_mask = hsv_img[:, :, 1] > 0.56

mask2 = upper_mask * lower_mask * saturation_mask
red = img[:, :, 0] * mask2
green = img[:, :, 1] * mask2
blue = img[:, :, 2] * mask2
img_masked = np.dstack((red, green, blue))
plt.imshow(img_masked)
plt.show()

#AMARILLO
hsv_img = rgb2hsv(img)
lower_mask = hsv_img[:, :, 0] > 0.09
upper_mask = hsv_img[:, :, 0] < 0.18
saturation_mask = hsv_img[:, :, 1] > 0.7

mask2 = upper_mask * lower_mask * saturation_mask
red = img[:, :, 0] * mask2
green = img[:, :, 1] * mask2
blue = img[:, :, 2] * mask2
img_masked = np.dstack((red, green, blue))
plt.imshow(img_masked)
plt.show()

#TODOS
hsv_img = rgb2hsv(img)
lower_mask = hsv_img[:,:,0] > 0.0
upper_mask = hsv_img[:,:,0] < 1
saturation_mask = hsv_img[:,:,1] > 0.5

mask2 = upper_mask*lower_mask*saturation_mask
red = img[:,:,0]*mask2
green = img[:,:,1]*mask2
blue = img[:,:,2]*mask2
img_masked = np.dstack((red, green, blue))
plt.imshow(img_masked)
plt.show()

