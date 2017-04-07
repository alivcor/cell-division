import math
import time
from os import path
from PIL import Image
from skimage.segmentation import slic
import cv2
from os import walk
import numpy as np
from scipy.io import loadmat
from skimage.data import astronaut
from skimage.viewer import ImageViewer
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import color
from skimage.util import img_as_float
from skimage import io as skimageIO
from pystruct.utils import make_grid_edges, SaveLogger
from pystruct.models import LatentGridCRF, GridCRF, LatentGraphCRF, GraphCRF, EdgeFeatureGraphCRF
from pystruct.learners import LatentSSVM, OneSlackSSVM, SubgradientSSVM, FrankWolfeSSVM
from pystruct.utils import make_grid_edges, SaveLogger
from skimage import img_as_ubyte
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.transform import resize
import skimage.feature as skf
from skimage.filters import threshold_otsu

try:
    import cPickle as pickle
except ImportError:
    import pickle

# superpixel color intensity, histogram, and the HoG.
# color intensity difference, histogram difference, and texture similarity for edge features.

pixel_class_arr = [np.array([0., 0., 0.]), np.array([1., 1., 1.])]

# Load trained Model
trained_model = SaveLogger('save/cells-hog.model', save_every=1)
cellsCRF = trained_model.load()
segments = None
patchSize = 96
patchHalfSize = patchSize / 2


def create_graph(grid):
    # get unique labels
    vertices = np.unique(grid)
    # map unique labels to [1,...,num_labels]
    reverse_dict = dict(zip(vertices, np.arange(len(vertices))))
    grid = np.array([reverse_dict[x] for x in grid.flat]).reshape(grid.shape)

    # create edges
    down = np.c_[grid[:-1, :].ravel(), grid[1:, :].ravel()]
    right = np.c_[grid[:, :-1].ravel(), grid[:, 1:].ravel()]
    all_edges = np.vstack([right, down])
    all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1], :]
    all_edges = np.sort(all_edges, axis=1)
    num_vertices = len(vertices)
    edge_hash = all_edges[:, 0] + num_vertices * all_edges[:, 1]
    # find unique connections
    edges = np.unique(edge_hash)
    # undo hashing
    edges = [[vertices[x % num_vertices],
              vertices[x / num_vertices]] for x in edges]

    return vertices, edges


def getTextureSimilarity(p1, p2):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    filt = np.logical_and(p1 != 0, p2 != 0)
    return np.sum(p1[filt] * np.log2(p1[filt] / p2[filt]))


def get_chi2_distance(histA, histB, eps=1e-10):
    return 0.5 * np.sum([(math.pow((x - y), 2)) / (x + y + eps) for (x, y) in zip(histA, histB)])


def getHistogramFeatures(bgrImage, centerY, centerX, forUnaryFeature=False):
    patch = np.zeros((patchSize, patchSize, 3), dtype=np.float32)

    # construct patch
    for i in range(centerY - patchHalfSize, centerY + patchHalfSize):
        for j in range(centerX - patchHalfSize, centerX + patchHalfSize):
            if i >= 0 and j >= 0 and i < bgrImage.shape[0] and j < bgrImage.shape[1]:
                patch[i - (centerY - patchHalfSize)][j - (centerX - patchHalfSize)] = bgrImage[i][j]

    # Histogram of intensity values
    hist = cv2.calcHist(patch, [0, 1, 2], None, [12, 12, 12], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, cv2.NORM_L2).flatten()

    grayPatch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    if forUnaryFeature:
        # Histogram of Oriented Gradients
        hog = skf.hog(grayPatch, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), visualise=False,
                      transform_sqrt=True)
        return hist, hog
    else:
        lbp = skf.local_binary_pattern(grayPatch, 24, 3, method='nri_uniform')
        n_bins = lbp.max() + 1
        lbphist, _ = np.histogram(lbp, normed=True, bins=n_bins.astype(np.int),
                                  range=(0, n_bins.astype(np.int)))
        return hist, lbphist


def preproc_data(basedir='dataset/cells/training/images/', labeldir='dataset/cells/training/masks/'):
    global pixel_class_arr
    dataSetX = []
    dataSetY = []

    for (dirpath, dirnames, filenames) in walk(basedir):
        n = 0
        for imageFilename in filenames:
            print imageFilename, n
            n = n + 1

            # Read RGB and label image
            image = img_as_float(skimageIO.imread(basedir + imageFilename))
            bgrImage = cv2.imread(basedir + imageFilename, cv2.IMREAD_COLOR)
            labelImage = img_as_float(
                skimageIO.imread(labeldir + imageFilename.replace('image', 'mask').replace('jpg', 'png')))
            # print "labelimage shape", labelImage.shape
            # plt.imshow(labelImage)
            if len(image.shape) == 2:
                image = color.gray2rgb(image)

            # Scan label image for additional classes
            if len(labelImage.shape) == 2:
                labelImageRGB = color.gray2rgb(labelImage)
            else:
                labelImageRGB = labelImage
            # Derive superpixels and get their average RGB component
            segments = slic(image, n_segments=500, sigma=1.0)
            rgb_segments = img_as_ubyte(mark_boundaries(image, segments))

            label_segments = img_as_ubyte(mark_boundaries(labelImageRGB, segments))

            avg_rgb = color.label2rgb(segments, image, kind='avg')
            avg_label = color.label2rgb(segments, labelImageRGB, kind='avg')
            # Create graph of superpixels and compute their centers
            vertices, edges = create_graph(segments)
            gridx, gridy = np.mgrid[:segments.shape[0], :segments.shape[1]]
            centers = dict()
            for v in vertices:
                centers[v] = [gridy[segments == v].mean(), gridx[segments == v].mean()]
            # print vertices
            # print edges
            # print centers

            # Build training instances
            n_features = []
            edge_features = []
            n_labels = []

            # Compute image centers
            centerX = labelImageRGB.shape[1] / 2.0
            centerY = labelImageRGB.shape[0] / 2.0

            for v in vertices:
                # unary features layer 1 - average rgb of superpixel, histogram of patch surrounding center and CNN features
                avg_rgb2 = avg_rgb[int(centers[v][1])][int(centers[v][0])]
                hist, hogFeatures = getHistogramFeatures(bgrImage, int(centers[v][1]), int(centers[v][0]),
                                                         forUnaryFeature=True)

                node_feature = np.concatenate([avg_rgb2, hist, hogFeatures])
                n_features.append(node_feature)

                # label
                minEuclideanDistance = np.inf
                pixelClass = -1
                for i in range(0, len(pixel_class_arr)):
                    # set the label of the superpixel to the pixelClass with minimum euclidean distance
                    dist = np.linalg.norm(avg_label[int(centers[v][1])][int(centers[v][0])] - pixel_class_arr[i])
                    if dist < minEuclideanDistance:
                        pixelClass = i
                        minEuclideanDistance = dist
                n_labels.append(pixelClass)

            hist_dict = {}
            for e in edges:
                dist = np.linalg.norm(
                    avg_rgb[int(centers[e[0]][1])][int(centers[e[0]][0])] - avg_rgb[int(centers[e[1]][1])][
                        int(centers[e[1]][0])])

                if e[0] not in hist_dict:
                    hist1, lbphist1 = getHistogramFeatures(bgrImage, int(centers[e[0]][1]), int(centers[e[0]][0]))
                    hist_dict[e[0]] = {'hist': hist1, 'lbphist': lbphist1}
                else:
                    hist1 = hist_dict[e[0]]['hist']
                    lbphist1 = hist_dict[e[0]]['lbphist']
                if e[1] not in hist_dict:
                    hist2, lbphist2 = getHistogramFeatures(bgrImage, int(centers[e[1]][1]), int(centers[e[1]][0]))
                    hist_dict[e[1]] = {'hist': hist2, 'lbphist': lbphist2}
                else:
                    hist2 = hist_dict[e[1]]['hist']
                    lbphist2 = hist_dict[e[1]]['lbphist']

                histogramDist = cv2.compareHist(hist1, hist2, 3)  # Bhattacharyya distance
                textureSimilarity = getTextureSimilarity(lbphist1, lbphist2)  # KL divergence

                pairwise_feature = np.array([dist, histogramDist, textureSimilarity])
                edge_features.append(pairwise_feature)

            # Add to dataset
            dataSetX.append((np.array(n_features), np.array(edges), np.array(edge_features)))
            dataSetY.append(np.array(n_labels))

    return dataSetX, dataSetY


def evaluatePerformance(clf, datasetX, datasetY):
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0

    y_pred = clf.predict(datasetX)
    for i in range(0, len(y_pred)):
        accuracy += accuracy_score(y_pred[i], datasetY[i])
        precision += precision_score(y_pred[i], datasetY[i])
        recall += recall_score(y_pred[i], datasetY[i])
        f1 += f1_score(y_pred[i], datasetY[i])

    print 'Superpixelwise Accuracy: ' + str(accuracy / len(y_pred))
    print 'Superpixelwise Precision: ' + str(precision / len(y_pred))
    print 'Superpixelwise Recall: ' + str(recall / len(y_pred))
    print 'Superpixelwise F1: ' + str(f1 / len(y_pred))

    return y_pred


def _train_crf(trainSetX, trainSetY, testSetX, testSetY):
    modelLogger = SaveLogger('save/cells-hog_2000.model', save_every=1)

    print 'Training CRF...'
    start_time = time.time()
    crf = EdgeFeatureGraphCRF()
    clf = FrankWolfeSSVM(model=crf, C=10., tol=.1, verbose=3, show_loss_every=1, logger=modelLogger,
                         max_iter=1000)  # #max_iter=50
    clf.fit(np.array(trainSetX), np.array(trainSetY))
    print 'Training CRF took ' + str(time.time() - start_time) + ' seconds'

    print 'SUPERPIXELWISE ACCURACY'
    print '-----------------------------------------------------------------------'
    print ''
    print 'TRAINING SET RESULTS'
    train_ypred = evaluatePerformance(clf, np.array(trainSetX), np.array(trainSetY))
    print ''
    print 'TEST SET RESULTS'
    evaluatePerformance(clf, np.array(testSetX), np.array(testSetY))
    print '-----------------------------------------------------------------------'


def getCellAccuracy(a, b):
    TP = TN = FP = FN = 0.0
    for i in xrange(0, len(a)):
        if a[i] == b[i]:
            if a[i] == 0:
                TN += 1
            else:
                TP += 1
        else:
            if a[i] == 0:
                FP += 1
            else:
                FN += 1
    return (TP / (TP + FP + FN))


def segment_image(orig_file, mask_file, pixelClasses=pixel_class_arr, crfmodel=cellsCRF):
    start_time = time.time()

    image = img_as_float(skimageIO.imread(orig_file))
    rgb_img = cv2.imread(orig_file, cv2.IMREAD_COLOR)

    if len(image.shape) == 2:
        image = color.gray2rgb(image)

    # get superpixels using slic tool
    segments = slic(image, n_segments=500, sigma=1.0)
    rgb_segments = img_as_ubyte(mark_boundaries(image, segments))
    avg_rgb = color.label2rgb(segments, image, kind='avg')

    labelImage = img_as_float(skimageIO.imread(mask_file))

    if len(labelImage.shape) == 2:
        labelImageRGB = color.gray2rgb(labelImage)
    else:
        labelImageRGB = labelImage
    label_segments = img_as_ubyte(mark_boundaries(labelImageRGB, segments))
    avg_label = color.label2rgb(segments, labelImageRGB, kind='avg')

    # superpixel graph
    vertices, edges = create_graph(segments)
    gridx, gridy = np.mgrid[:segments.shape[0], :segments.shape[1]]
    centers = dict()
    for v in vertices:
        centers[v] = [gridy[segments == v].mean(), gridx[segments == v].mean()]

    train_x = []
    train_y = []
    n_features = []
    n_labels = []
    edge_features = []

    for v in vertices:
        # average intensity
        avg_rgb2 = avg_rgb[int(centers[v][1])][int(centers[v][0])]
        hist, hogFeatures = getHistogramFeatures(rgb_img, int(centers[v][1]), int(centers[v][0]), forUnaryFeature=True)
        node_feature = np.concatenate([avg_rgb2, hist, hogFeatures])
        n_features.append(node_feature)

        minEuclideanDistance = np.inf
        pixelClass = -1
        for i in range(0, len(pixelClasses)):
            # set the label of the superpixel to the pixelClass with minimum euclidean distance
            dist = np.linalg.norm(avg_label[int(centers[v][1])][int(centers[v][0])] - pixelClasses[i])
            if dist < minEuclideanDistance:
                pixelClass = i
                minEuclideanDistance = dist
        n_labels.append(pixelClass)

    hist_dict = {}
    for e in edges:
        # pairwise feature - euclidean distance of pairwise superpixels
        dist = np.linalg.norm(avg_rgb[int(centers[e[0]][1])][int(centers[e[0]][0])] - avg_rgb[int(centers[e[1]][1])][
            int(centers[e[1]][0])])

        if e[0] not in hist_dict:
            hist1, lbphist1 = getHistogramFeatures(rgb_img, int(centers[e[0]][1]), int(centers[e[0]][0]))
            hist_dict[e[0]] = {'hist': hist1, 'lbphist': lbphist1}
        else:
            hist1 = hist_dict[e[0]]['hist']
            lbphist1 = hist_dict[e[0]]['lbphist']
        if e[1] not in hist_dict:
            hist2, lbphist2 = getHistogramFeatures(rgb_img, int(centers[e[1]][1]), int(centers[e[1]][0]))
            hist_dict[e[1]] = {'hist': hist2, 'lbphist': lbphist2}
        else:
            hist2 = hist_dict[e[1]]['hist']
            lbphist2 = hist_dict[e[1]]['lbphist']

        histogramDist = cv2.compareHist(hist1, hist2, 3)
        textureSimilarity = getTextureSimilarity(lbphist1, lbphist2)

        pairwise_feature = np.array([dist, histogramDist, textureSimilarity])
        edge_features.append(pairwise_feature)

    train_x.append((np.array(n_features), np.array(edges), np.array(edge_features)))
    train_y.append(np.array(n_labels))

    orig_masked_sp = np.zeros(labelImageRGB.shape)
    masked_pred_sp = np.zeros(labelImage.shape)
    # print orig_masked_sp.shape
    # print masked_pred_sp.shape
    for i in range(0, orig_masked_sp.shape[0]):
        for j in range(0, orig_masked_sp.shape[1]):
            orig_masked_sp[i][j] = pixelClasses[n_labels[segments[i][j]]]
            masked_pred_sp[i][j] = n_labels[segments[i][j]]

    # Predict with CRF and build image mask
    y_pred = crfmodel.predict(np.array(train_x))
    orig_masked = np.zeros(image.shape)
    masked_pred = np.zeros((image.shape[0], image.shape[1]))

    for i in range(0, orig_masked.shape[0]):
        for j in range(0, orig_masked.shape[1]):
            orig_masked[i][j] = pixelClasses[y_pred[0][segments[i][j]]]
            masked_pred[i][j] = y_pred[0][segments[i][j]]

    pw_recall = recall_score(labelImage.flatten().flatten(), masked_pred.flatten().flatten())
    pw_f1 = f1_score(labelImage.flatten().flatten(), masked_pred.flatten().flatten())
    pw_so = getCellAccuracy(labelImage.flatten().flatten(), masked_pred.flatten().flatten())
    pw_accuracy = accuracy_score(labelImage.flatten().flatten(), masked_pred.flatten().flatten())
    pw_precision = precision_score(labelImage.flatten().flatten(), masked_pred.flatten().flatten())

    print '\nSegmentation completed in ' + str(time.time() - start_time) + ' seconds.'
    print 'Total Pixels: ' + str(labelImage.flatten().flatten().shape[0])
    print 'SLIC Pixelwise Accuracy: ' + str(
        accuracy_score(labelImage.flatten().flatten(), masked_pred_sp.flatten().flatten()))
    print ''
    print 'Pixelwise Accuracy: ' + str(pw_accuracy)
    print 'Pixelwise Precision: ' + str(pw_precision)
    print 'Pixelwise Recall: ' + str(pw_recall)
    print 'Pixelwise F1: ' + str(pw_f1)
    print 'Pixelwise S0: ' + str(pw_so)

    fig, ax = plt.subplots(2, 3)
    fig.canvas.set_window_title('Image Segmentation')
    ax[0, 0].imshow(image)
    ax[0, 0].set_title("Original Image")

    ax[0, 1].imshow(rgb_segments)
    ax[0, 1].set_title("Super Pixels")

    if mask_file is not None:
        ax[0, 2].imshow(label_segments)
        ax[1, 0].imshow(labelImageRGB)
        ax[1, 1].imshow(orig_masked_sp)

    ax[0, 2].set_title("Segmented Ground Truth")
    ax[1, 0].set_title("Ground truth")
    ax[1, 1].set_title("Labeled Super Pixels")

    ax[1, 2].imshow(orig_masked)
    ax[1, 2].set_title("Prediction")

    for a in ax.ravel():
        a.set_xticks(())
        a.set_yticks(())
    plt.show()

    # Return metrics
    if mask_file is not None:
        return pw_accuracy, pw_precision, pw_recall, pw_f1, pw_so
    else:
        return


def trainCRF():
    print 'Generating training set'
    train_X, train_Y = preproc_data(basedir='../exports224/train/x/',
                                    labeldir='../exports224/train/y/')
    print 'Generating test set'
    test_X, test_Y = preproc_data(basedir='../exports224/test/x/', labeldir='../exports224/test/y/')
    print 'Training the CRF'
    _train_crf(train_X, train_Y, test_X, test_Y)


if __name__ == "__main__":
    # orig_file = "dataset/cells/test/images/image-12.jpg"
    # mask_file = orig_file.replace('image', 'mask').replace('jpg', 'png')
    #
    # segment_image(orig_file=orig_file, mask_file=mask_file)
    trainCRF()
