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

#Extract edges using slic
img = cv2.imread("SAMPLE.jpg")
segments = slic(img, n_segments=92, compactness=10)
print(np.array(segments).shape)
fig = plt.figure("Superpixels")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), segments))
plt.axis("off")
plt.show()


#data_train is a dict : please download the pre-processed data from: http://www.ais.uni-bonn.de/deep_learning/downloads.html
# this is the only available information about what library expects from us as input.
# I have done some analysis to find out and have written some code to help us understand it.
# what each variable means has been written alongside in the comments
data_train = pickle.load(open("data_train.pickle", 'rb'))
# data_train is a dict having 4 things: X, Y, Superpixels and filenames

print(type(data_train))
print(len(data_train))
print("-------------\n\n")
for key in data_train:
    print key
    val = data_train[key]
    print "Length of val : ", len(val)
    print "Shape of val : ", (np.array(val)).shape
    print "Shape of each element in val : ", (np.array(val[0:1])).shape
    print "-------------------"

print(np.array(data_train['Y'][0]))
print(np.array(data_train['X']).shape)
print(np.array(data_train['X'])[0][0].shape) # 21 features for each of these 92 superpixels
print(np.array(data_train['X'])[0][1].shape) # Pairwise Edges between superpixels
print(np.array(data_train['X'])[0][2].shape) # Three Edge Features for each edge between superpixels.


#The code from here is from an example in the library over here : http://pystruct.github.io/auto_examples/image_segmentation.html
#From what I observe, only 'X' is used in this code.

C = 0.01

n_states = 21

print("number of samples: %s" % len(data_train['X']))
class_weights = 1. / np.bincount(np.hstack(data_train['Y']))
class_weights *= 21. / np.sum(class_weights)
print(class_weights)

model = crfs.EdgeFeatureGraphCRF(inference_method='max-product',
                                 class_weight=class_weights,
                                 symmetric_edge_features=[0, 1],
                                 antisymmetric_edge_features=[2])

experiment_name = "edge_features_one_slack_trainval_%f" % C

ssvm = learners.NSlackSSVM(
    model, verbose=2, C=C, max_iter=100000, n_jobs=-1,
    tol=0.0001, show_loss_every=5,
    logger=SaveLogger(experiment_name + ".pickle", save_every=100),
    inactive_threshold=1e-3, inactive_window=10, batch_size=100)
ssvm.fit(data_train['X'], data_train['Y'])

data_val = pickle.load(open("data_val_dict.pickle"))
y_pred = ssvm.predict(data_val['X'])

# we throw away void superpixels and flatten everything
y_pred, y_true = np.hstack(y_pred), np.hstack(data_val['Y'])
y_pred = y_pred[y_true != 255]
y_true = y_true[y_true != 255]

print("Score on validation set: %f" % np.mean(y_true == y_pred))
