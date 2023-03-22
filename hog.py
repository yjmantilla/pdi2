import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure,io,color
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import area_opening
from skimage.measure import label
from skimage.transform import resize
import os
import numpy as np

from PIL import Image

path = 'dataset/ACU185.jpg'
image= io.imread(path)
image = color.rgb2gray(image)


def imadjust(x,a,b,c,d,gamma=1):
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y

def stretchlim(img,method='adaptive'):

    # Contrast stretching
    if method == 'percentile':
        p2 = np.percentile(img, 2)
        p98 = np.percentile(img, 98)
        img2 = exposure.rescale_intensity(img, in_range=(p2, p98))
    elif method == 'equalization':
        # Equalization
        img2 = exposure.equalize_hist(img)
    elif method == 'adaptive':
        # Adaptive Equalization
        img2 =exposure.equalize_adapthist(img, clip_limit=0.03)
    return img2

image=stretchlim(image)

image=1-image
l=threshold_otsu(image)
def binarize(x,l):
    return (x > l)*1
image = binarize(image,l)

image = clear_border(image)

image = area_opening(image)[30:130,:]

plt.imshow(image)
plt.show()

#labeling
labeled_image = label(image,connectivity=2)
labels,counts = np.unique(labeled_image,return_counts=True)

# keep first 7 (background+6 letters)

labels = labels[:7]
counts = counts[:7]

# for each label crop
la = 1
idxs=np.where(labeled_image==la)
#identify border x,y
minx,maxx = min(idxs[0]),max(idxs[0])
miny,maxy = min(idxs[1]),max(idxs[1])

# Assume label 0 is background
standard_size =(100, 50)
letter = resize(labeled_image[minx:maxx,miny:maxy]>0,standard_size ,anti_aliasing=False)
plt.imshow(letter)
plt.show()


image=letter
"""
# Calculating the SVD
u, s, v = np.linalg.svd(labeled_image[minx:maxx,miny:maxy], full_matrices=False)
  
# inspect shapes of the matrices
print(f'u.shape:{u.shape},s.shape:{s.shape},v.shape:{v.shape}')


comps = [u.shape[0], u.shape[0]//2, u.shape[0]//4,u.shape[0]//8]
plt.figure(figsize=(12, 6))
  
for i in range(len(comps)):
    low_rank = u[:, :comps[i]] @ np.diag(s[:comps[i]]) @ v[:comps[i], :]
      
    if(i == 0):
        plt.subplot(2, 3, i+1),
        plt.imshow(low_rank, cmap='gray'),
        plt.title(f'Actual Image with n_components = {comps[i]}')
      
    else:
        plt.subplot(2, 3, i+1),
        plt.imshow(low_rank, cmap='gray'),
        plt.title(f'n_components = {comps[i]}')
"""
fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()