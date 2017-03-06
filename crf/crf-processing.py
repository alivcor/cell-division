import sys

path = "/home/dpakhom1/dense_crf_python/"
sys.path.append(path)

import pydensecrf.densecrf as dcrf

from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary, unary_from_softmax

import numpy as np

from matplotlib import pyplot as plt


import h5py

import scipy.io as sio


heat_data = sio.loadmat('heatmap.mat')

#teX =  dataset['teX'][()]
#trY, valY, teY = dataset['trY'][()], dataset['valY'][()], dataset['teY'][()]

classmap_answer = heat_data['classmap_answer']


dataset_large = h5py.File('dataset_large_224.mat')

teX =  dataset_large['teX'][()]
trY, valY, teY = dataset_large['trY'][()], dataset_large['valY'][()], dataset_large['teY'][()]




image = teX


image = np.transpose(image,[3,2,1,0])

image = image[1,:,:,:].squeeze()

classmap_answer = classmap_answer[1,:,:]
#softmax = clasmap_answer.squeeze()
classmap_answer = np.tile(classmap_answer, [3,1,1])

classmap_answer = np.transpose(classmap_answer,(1,2,0))



softmax = classmap_answer.transpose((2, 0, 1))

# The input should be the negative of the logarithm of probability values
# Look up the definition of the softmax_to_unary for more information
unary = unary_from_softmax(softmax)

# The inputs should be C-continious -- we are using Cython wrapper
unary = np.ascontiguousarray(unary) * 100

d = dcrf.DenseCRF(image.shape[0] * image.shape[1], 2)

d.setUnaryEnergy(unary)

# This potential penalizes small pieces of segmentation that are
# spatially isolated -- enforces more spatially consistent segmentations
feats = create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])

d.addPairwiseEnergy(feats*0.01, compat=3,
                    kernel=dcrf.DIAG_KERNEL,
                    normalization=dcrf.NORMALIZE_SYMMETRIC)

# This creates the color-dependent features --
# because the segmentation that we get from CNN are too coarse
# and we can use local color features to refine them
feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20), # sdims was 50,50
                                   img=image, chdim=2)

d.addPairwiseEnergy(feats*0.01, compat=10,
                     kernel=dcrf.DIAG_KERNEL,
                     normalization=dcrf.NORMALIZE_SYMMETRIC)
Q = d.inference(5) # was 5

res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))

cmap = plt.get_cmap('bwr')

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.imshow(res, vmax=1.5, vmin=-0.4, cmap=cmap)
ax1.set_title('Segmentation with CRF post-processing')
#probability_graph = ax2.imshow(np.dstack((train_annotation,)*3)*100)
ax2.set_title('Ground-Truth Annotation')
plt.show()