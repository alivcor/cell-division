import time
from imp_crf_core import preproc_data
from imp_crf_core import create_graph
from imp_crf_core import getHistogramFeatures
from imp_crf_core import train_crf, evaluatePerformance, kullback_leibler_divergence, chi2_distance
from os import path
from PIL import Image
from skimage.segmentation import slic
import cv2
import numpy as np
from skimage.data import astronaut
from skimage.viewer import ImageViewer
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import color
from skimage.util import img_as_float
from skimage import io as skimageIO
from pystruct.utils import make_grid_edges, SaveLogger
from skimage import img_as_ubyte
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.transform import resize
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

def foregroundQualityScore(a , b) :
    TP = TN = FP = FN = 0.0
    for i in range(0, len(a)) :
        if a[i] == b[i] :
            if a[i] == 0 :
                TN += 1
            else :
                TP += 1
        else :
            if a[i] == 0 :
                FP += 1
            else :
                FN += 1
    
    #print 'accuracy:' + str(((TP+TN) / (TP+FP+FN+TN)))
    #print 'precision:' + str((TP / (TP+FP)))
    #print 'recall:' + str((TP / (TP+FN)))
    #print 'so:' + str(TP / (TP+FP+FN))
    
    return (TP / (TP+FP+FN))


def segment_image(rgbfile, labelfile, pixelClasses = pixel_class_arr, crfmodel = cellsCRF, visualizeSegmentation=True) :
    start_time = time.time()
    
    # Read RGB and label image
    image = img_as_float(skimageIO.imread(rgbfile))
    bgrImage = cv2.imread(rgbfile,cv2.IMREAD_COLOR)
    #bgrImage = cv2.fastNlMeansDenoisingColored(bgrImage)
    #bgrImage = exposure.adjust_sigmoid(bgrImage)
    if len(image.shape) == 2 : 
                image = color.gray2rgb(image)
                
    # Resize
    #image = resize(image,(120,120), preserve_range=True )
    #bgrImage = resize(bgrImage,(120,120), preserve_range=True)
    
                
    # Derive superpixels and get their average RGB component
    segments = slic(image, n_segments =500, sigma = 1.0)
    rgb_segments = img_as_ubyte(mark_boundaries(image, segments))
    avg_rgb = color.label2rgb(segments, image, kind='avg') 
                
    if labelfile is not None :            
        labelImage = img_as_float(skimageIO.imread(labelfile))[:,:,0]*255
        #labelImage = resize(labelImage,(120,120), preserve_range=True)
        #thresh = threshold_otsu(image)
        #labelImage = labelImage > thresh
        if len(labelImage.shape) == 2 : 
            labelImageRGB = color.gray2rgb(labelImage)
        else :
            labelImageRGB = labelImage
        label_segments = img_as_ubyte(mark_boundaries(labelImageRGB, segments))            
        avg_label = color.label2rgb(segments, labelImageRGB, kind='avg')
     
  

    # Create graph of superpixels and compute their centers
    vertices, edges = create_graph(segments)
    gridx, gridy = np.mgrid[:segments.shape[0], :segments.shape[1]]
    centers = dict()
    for v in vertices:
        centers[v] = [gridy[segments == v].mean(), gridx[segments == v].mean()]

    xInstance = []
    yInstance = []
    n_features = []            
    n_labels = []
    edge_features = []
    
    
    for v in vertices:
        # unary feature - average rgb of superpixel
        avg_rgb2 = avg_rgb[int(centers[v][1])][int(centers[v][0])]
        hist, hogFeatures = getHistogramFeatures(bgrImage, int(centers[v][1]), int(centers[v][0]), forUnaryFeature=True)
        node_feature = np.concatenate([avg_rgb2, hist, hogFeatures])
        n_features.append(node_feature)
        
        # label 
        if labelfile is not None :        
            minEuclideanDistance = np.inf # simulate infinity
            pixelClass = -1 
            for i in range(0, len(pixelClasses)) :
                # set the label of the superpixel to the pixelClass with minimum euclidean distance
                dist = np.linalg.norm( avg_label[int(centers[v][1])][int(centers[v][0])] - pixelClasses[i] )
                if dist < minEuclideanDistance :
                    pixelClass = i
                    minEuclideanDistance = dist
            n_labels.append(pixelClass)
    
    histogramCache = {}
    for e in edges :
        # pairwise feature - euclidean distance of adjacent superpixels
        dist = np.linalg.norm(avg_rgb[int(centers[e[0]][1])][int(centers[e[0]][0])] - avg_rgb[int(centers[e[1]][1])][int(centers[e[1]][0])] )
        
        
        if e[0] not in histogramCache :
            hist1, lbphist1 = getHistogramFeatures(bgrImage, int(centers[e[0]][1]), int(centers[e[0]][0]))
            histogramCache[e[0]] = {'hist' : hist1, 'lbphist' : lbphist1}
        else :
            hist1 =  histogramCache[e[0]]['hist']
            lbphist1 =  histogramCache[e[0]]['lbphist']
        if e[1] not in histogramCache :
            hist2, lbphist2 = getHistogramFeatures(bgrImage, int(centers[e[1]][1]), int(centers[e[1]][0]))
            histogramCache[e[1]] = {'hist' : hist2, 'lbphist' : lbphist2}
        else :
            hist2 =  histogramCache[e[1]]['hist']
            lbphist2 =  histogramCache[e[1]]['lbphist']
            
      
        histogramDist = cv2.compareHist(hist1, hist2, 3)
        textureSimilarity = kullback_leibler_divergence(lbphist1, lbphist2)
        
        pairwise_feature = np.array([dist, histogramDist, textureSimilarity])
        edge_features.append(pairwise_feature)
    
    
    xInstance.append((np.array(n_features), np.array(edges), np.array(edge_features))) 
    yInstance.append(np.array(n_labels))
    
    # Create superpixeled image
    if labelfile is not None :    
        labeledSuperPixeledRGB = np.zeros(labelImageRGB.shape)
        labeledSuperPixeled = np.zeros(labelImage.shape)
        #print labeledSuperPixeled.shape
        #print labelImageRGB.shape
        for i in range(0,labeledSuperPixeledRGB.shape[0]) :
            for j in range(0,labeledSuperPixeledRGB.shape[1]) :    
                labeledSuperPixeledRGB[i][j] = pixelClasses[n_labels[segments[i][j]]]
                labeledSuperPixeled[i][j] = n_labels[segments[i][j]]
            
    
    # Predict with CRF and build image label
    y_pred = crfmodel.predict(np.array(xInstance))
    labeledPredictionRGB = np.zeros(image.shape)
    labeledPrediction = np.zeros((image.shape[0], image.shape[1]))
    #print labeledPrediction.shape
    for i in range(0,labeledPredictionRGB.shape[0]) :
        for j in range(0,labeledPredictionRGB.shape[1]) :
            labeledPredictionRGB[i][j] = pixelClasses[y_pred[0][segments[i][j]]]
            labeledPrediction[i][j] = y_pred[0][segments[i][j]]
    

    # Print performance
    if labelfile is not None : 
        pixelwise_accuracy = accuracy_score(labelImage.flatten().flatten(),  labeledPrediction.flatten().flatten()) 
        pixelwise_precision = precision_score(labelImage.flatten().flatten(), labeledPrediction.flatten().flatten())
        pixelwise_recall = recall_score(labelImage.flatten().flatten(), labeledPrediction.flatten().flatten())
        pixelwise_f1 = f1_score(labelImage.flatten().flatten(), labeledPrediction.flatten().flatten()) 
        pixelwise_so = foregroundQualityScore(labelImage.flatten().flatten(), labeledPrediction.flatten().flatten())


        
        print ''
        print 'Segmentation completed in ' + str(time.time() - start_time) + ' seconds.'
        print 'Total Pixels: ' + str(labelImage.flatten().flatten().shape[0])
        print 'SLIC Pixelwise Accuracy: ' + str(accuracy_score(labelImage.flatten().flatten(), labeledSuperPixeled.flatten().flatten()))
        print ''
        print 'Pixelwise Accuracy: ' + str(pixelwise_accuracy)
        print 'Pixelwise Precision: ' + str(pixelwise_precision)
        print 'Pixelwise Recall: ' + str(pixelwise_recall)
        print 'Pixelwise F1: ' + str(pixelwise_f1)
        print 'Pixelwise S0: ' + str(pixelwise_so)
    else :
        'There is no label image hence no performance stats...'

    if visualizeSegmentation :
        fig, ax = plt.subplots(2, 3)
        fig.canvas.set_window_title('Image Segmentation')
        ax[0, 0].imshow(image)
        ax[0, 0].set_title("Original Image")
        
        ax[0, 1].imshow(rgb_segments)
        ax[0, 1].set_title("Super Pixels")
        
        if labelfile is not None :
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
    
    # Return metrics
    if labelfile is not None :
        return pixelwise_accuracy, pixelwise_precision, pixelwise_recall, pixelwise_f1, pixelwise_so
    else :
        return


if __name__ == "__main__":

    rgbFile = "dataset/cells/test/images/image-12.jpg"
    labelFile = rgbFile.replace('image', 'mask').replace('jpg', 'png')

    segment_image(rgbfile=rgbFile, labelfile=labelFile)
