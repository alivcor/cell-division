

import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle
from PIL import Image
from skimage.segmentation import slic
import cv2
from skimage.data import astronaut
from skimage.viewer import ImageViewer

from pystruct import learners
import pystruct.models as crfs
from pystruct.utils import SaveLogger
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt


from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage import color
from skimage.util import img_as_float
from skimage import io as skimageIO
from skimage import img_as_ubyte

cell_pixel_labels = [np.array([0.,  0.,  0.]), np.array([1.,  1.,  1.])]

orig_image_path = "cell_data/I0.jpg"
mask_image_path = "cell_data/M0.png"

# load img and img mask
orig_img = cv2.imread(orig_image_path)
orig_img_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
mask_img = cv2.imread(mask_image_path)
mask_img_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
print "orig_img_gray.shape", orig_img_gray.shape
print "mask_img_gray.shape",mask_img_gray.shape

# Extract edges using slic
segments = slic(orig_img, n_segments=200, compactness=10, sigma = 1.0)
orig_segments = img_as_ubyte(mark_boundaries(orig_img, segments))
avg_orig = color.label2rgb(segments, orig_img, kind='avg')

mask_segments = img_as_ubyte(mark_boundaries(mask_img, segments))
avg_mask = color.label2rgb(segments, mask_img, kind='avg')




#visualize Segments
print("Shape of the variable segments:",np.array(segments).shape)
print("mark_boundaries Returns : ", mark_boundaries(img_as_float(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)), segments))
fig = plt.figure("Superpixels")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)), segments))
plt.axis("off")
plt.show()

# Create graph of superpixels and compute their centers
vertices, edges = pulzuhr_core.make_graph(segments)
gridx, gridy = np.mgrid[:segments.shape[0], :segments.shape[1]]
centers = dict()
for v in vertices:
    centers[v] = [gridy[segments == v].mean(), gridx[segments == v].mean()]

# Compute features
xInstance = []
yInstance = []
n_features = []
n_labels = []
edge_features = []
