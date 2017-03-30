

import numpy as np
from pulzuhr_core import constructCRFGraph
from pulzuhr_core import generateTrainData
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
crf_graph = constructCRFGraph(segments)
print "Pulzuhr Graph :"
print crf_graph.vertices, crf_graph.edges


formattedData = generateTrainData(orig_img=orig_img, mask_img=mask_img, graph=crf_graph, segments=segments, avg_orig= avg_orig, avg_mask=avg_mask, cell_pixel_labels=cell_pixel_labels)



# Create superpixeled image

labeledSuperPixeledRGB = np.zeros(mask_img.shape)
labeledSuperPixeled = np.zeros(mask_img.shape)
# print labeledSuperPixeled.shape
# print labelImageRGB.shape
for i in range(0, labeledSuperPixeledRGB.shape[0]):
    for j in range(0, labeledSuperPixeledRGB.shape[1]):
        labeledSuperPixeledRGB[i][j] = cell_pixel_labels[n_labels[segments[i][j]]]
        labeledSuperPixeled[i][j] = n_labels[segments[i][j]]

# Predict with CRF and build image label
y_pred = crfmodel.predict(np.array(data_train_x))
labeledPredictionRGB = np.zeros(image.shape)
labeledPrediction = np.zeros((image.shape[0], image.shape[1]))
# print labeledPrediction.shape
for i in range(0, labeledPredictionRGB.shape[0]):
    for j in range(0, labeledPredictionRGB.shape[1]):
        labeledPredictionRGB[i][j] = pixelClasses[y_pred[0][segments[i][j]]]
        labeledPrediction[i][j] = y_pred[0][segments[i][j]]

# Print performance
if labelfile is not None:
    pixelwise_accuracy = accuracy_score(labelImage.flatten().flatten(), labeledPrediction.flatten().flatten())
    pixelwise_precision = precision_score(labelImage.flatten().flatten(), labeledPrediction.flatten().flatten())
    pixelwise_recall = recall_score(labelImage.flatten().flatten(), labeledPrediction.flatten().flatten())
    pixelwise_f1 = f1_score(labelImage.flatten().flatten(), labeledPrediction.flatten().flatten())
    pixelwise_so = foregroundQualityScore(labelImage.flatten().flatten(), labeledPrediction.flatten().flatten())

    # comment on f1 score
    if pixelwise_f1 >= 0.9:
        comment = ' <------------ HIGH!'
    elif pixelwise_f1 <= 0.8:
        comment = ' <------------ LOW!'
    else:
        comment = ''

    print ''
    print 'Segmentation completed in ' + str(time.time() - start_time) + ' seconds.'
    print 'Total Pixels: ' + str(labelImage.flatten().flatten().shape[0])
    print 'SLIC Pixelwise Accuracy: ' + str(
        accuracy_score(labelImage.flatten().flatten(), labeledSuperPixeled.flatten().flatten()))
    print ''
    print 'Pixelwise Accuracy: ' + str(pixelwise_accuracy)
    print 'Pixelwise Precision: ' + str(pixelwise_precision)
    print 'Pixelwise Recall: ' + str(pixelwise_recall)
    print 'Pixelwise F1: ' + str(pixelwise_f1) + comment
    print 'Pixelwise S0: ' + str(pixelwise_so)
else:
    'There is no label image hence no performance stats...'

# Show the Images
if visualizeSegmentation:
    fig, ax = plt.subplots(2, 3)
    fig.canvas.set_window_title('Image Segmentation')
    ax[0, 0].imshow(image)
    ax[0, 0].set_title("Original Image")

    ax[0, 1].imshow(rgb_segments)
    ax[0, 1].set_title("Super Pixels")

    if labelfile is not None:
        ax[0, 2].imshow(label_segments)
        ax[1, 0].imshow(labelImageRGB)
        ax[1, 1].imshow(labeledSuperPixeledRGB)

    ax[0, 2].set_title("Segmented Ground Truth")
    ax[1, 0].set_title("Ground truth")
    ax[1, 1].set_title("Labeled Super Pixels")

    ax[1, 2].imshow(labeledPredictionRGB)
    ax[1, 2].set_title("Prediction")

    for a in ax.ravel():
        a.set_xticks(())
        a.set_yticks(())
    plt.show()