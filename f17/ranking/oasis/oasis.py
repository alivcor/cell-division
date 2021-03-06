""" Python implementation of the OASIS algorithm.
Graham Taylor

Based on Matlab implementation of:
Chechik, Gal, et al.
"Large scale online learning of image similarity through ranking."
The Journal of Machine Learning Research 11 (2010): 1109-1135.
"""

from __future__ import division
import numpy as np
from glob import glob
from sklearn.base import BaseEstimator
from datetime import datetime
import matplotlib.pyplot as plt
import cPickle as pickle
import os
import gzip
import cv2
from random import randint
from sys import stdout
import voc_preprocessor
from voc_preprocessor import countObjects, generateFileIDs
from os import listdir
from os.path import isfile, join

def snorm(x):
    """Dot product based squared Euclidean norm implementation

    See: http://fseoane.net/blog/2011/computing-the-vector-norm/
    Doesn't matter if row or column vectors are passed in
    Since everything is flattened
    """
    return np.dot(x.flatten().T, x.flatten())


def make_psd(W):
    """ Make matrix positive semi-definite. """
    w, v = np.linalg.eig(0.5 * (W + W.T))  # eigvec in columns
    D = np.diagflat(np.maximum(w, 0))
    W[:] = np.dot(np.dot(v, D), v.T)


def symmetrize(W):
    """ Symmetrize matrix. """
    W[:] = 0.5 * (W + W.T)


class Oasis(BaseEstimator):
    """ OASIS algorithm. """

    def __init__(self, aggress=0.1, random_seed=None, do_sym=False,
                 do_psd=False, n_iter=10 ** 6, save_every=None, sym_every=1,
                 psd_every=1, save_path=None):

        self.aggress = aggress
        self.random_seed = random_seed
        self.n_iter = n_iter
        self.do_sym = do_sym
        self.do_psd = do_psd
        self.sym_every = sym_every
        self.psd_every = psd_every
        self.save_path = save_path

        if save_every is None:
            self.save_every = int(np.ceil(self.n_iter / 10))
        else:
            self.save_every = save_every

        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

    def _getstate(self):
        return (self._weights, )

    def _setstate(self, state):
        weights, = state
        self._weights = weights

    def _save(self, n=None):
        """ Pickle the model."""
        fname = self.save_path + "/model%04d.pklz" % n
        f = gzip.open(fname, 'wb')
        state = self._getstate()
        pickle.dump(state, f)
        f.close()

    def read_snapshot(self, fname):
        """ Read model state snapshot from gzipped pickle. """
        f = gzip.open(fname, 'rb')
        state = pickle.load(f)
        self._setstate(state)

    def _fit_batch(self, W, X, y, class_start, class_sizes, n_iter,
                   verbose=False):
        """ Train batch inner loop. """

        loss_steps_batch = np.empty((n_iter,), dtype='bool')
        n_samples, n_features = X.shape

        # assert(W.shape[0] == n_features)
        # assert(W.shape[1] == n_features)

        for ii in xrange(n_iter):
            if verbose:
                if np.mod(ii + 1, 100) == 0:
                    print ".",
                if np.mod(ii + 1, 1000) == 0:
                    print "%d" % (ii + 1),
                if np.mod(ii + 1, 10000) == 0:
                    print "[%s]" % str(datetime.now())
                # http://stackoverflow.com/questions/5101151/print-without-newline-under-function-doesnt-work-as-it-should-in-python
                stdout.flush()

            # Sample a query image
            p_ind = self.init.randint(n_samples)
            label = y[p_ind]

            # Draw random positive sample
            pos_ind = class_start[label] + \
                self.init.randint(class_sizes[label])

            # Draw random negative sample
            neg_ind = self.init.randint(n_samples)
            while y[neg_ind] == label:
                neg_ind = self.init.randint(n_samples)

            p = X[p_ind]

            samples_delta = X[pos_ind] - X[neg_ind]

            loss = 1 - np.dot(np.dot(p, W), samples_delta)

            if loss > 0:
                # Update W
                grad_W = np.outer(p, samples_delta)

                loss_steps_batch[ii] = True

                norm_grad_W = np.dot(p, p) * np.dot(samples_delta,
                                                    samples_delta)

                # constraint on the maximal update step size
                tau_val = loss / norm_grad_W  # loss / (V*V.T)
                tau = np.minimum(self.aggress, tau_val)

                W += tau * grad_W
                # plt.figure(10)
                # plt.imshow(W,interpolation='nearest')
                # plt.draw()

            # print "loss = %f" % loss

        return W, loss_steps_batch

    def fit(self, X, y, overwrite_X=True, overwrite_y=True, verbose=True):
        """ Fit an OASIS model. """

        if not overwrite_X:
            X = X.copy()
        if not overwrite_y:
            y = y.copy()

        n_samples, n_features = X.shape

        self.init = np.random.RandomState(self.random_seed)

        print(n_features)
        # Parameter initialization
        self._weights = np.eye(n_features).flatten()
        # self._weights = np.random.randn(n_features,n_features).flatten()
        W = self._weights.view()
        W.shape = (n_features, n_features)

        ind = np.argsort(y)

        y = y[ind]
        X = X[ind, :]

        classes = np.unique(y)
        classes.sort()

        n_classes = len(classes)

        # Translate class labels to serial integers 0, 1, ...
        y_new = np.empty((n_samples,), dtype='int')

        for ii in xrange(n_classes):
            y_new[y == classes[ii]] = ii

        y = y_new
        class_sizes = [None] * n_classes
        class_start = [None] * n_classes

        for ii in xrange(n_classes):
            class_sizes[ii] = np.sum(y == ii)
            # This finds the first occurrence of that class
            class_start[ii] = np.flatnonzero(y == ii)[0]

        loss_steps = np.empty((self.n_iter,), dtype='bool')
        n_batches = int(np.ceil(self.n_iter / self.save_every))
        steps_vec = np.ones((n_batches,), dtype='int') * self.save_every
        steps_vec[-1] = self.n_iter - (n_batches - 1) * self.save_every

        if verbose:
            print 'n_batches = %d, total n_iter = %d' % (n_batches,
                                                         self.n_iter)

        for bb in xrange(n_batches):
            if verbose:
                print 'run batch %d/%d, for %d steps ("." = 100 steps)\n' \
                      % (bb + 1, n_batches, self.save_every)

            W, loss_steps_batch = self._fit_batch(W, X, y,
                                                  class_start,
                                                  class_sizes,
                                                  steps_vec[bb],
                                                  verbose=verbose)

            # print "loss_steps_batch = %d" % sum(loss_steps_batch)
            loss_steps[bb * self.save_every:min(
                (bb + 1) * self.save_every, self.n_iter)] = loss_steps_batch

            if self.do_sym:
                if np.mod(bb + 1, self.sym_every) == 0 or bb == n_batches - 1:
                    if verbose:
                        print "Symmetrizing"
                    symmetrize(W)

            if self.do_psd:
                if np.mod(bb + 1, self.psd_every) == 0 or bb == n_batches - 1:
                    if verbose:
                        print "PSD"
                    make_psd(W)

            if self.save_path is not None:
                self._save(bb + 1)  # back up model state

        return self


    def getFarthestLabels(self, X_test, X_train, y_train, maxk=200):
        '''
        Evaluate an OASIS model by KNN classification
        Examples are in rows
        '''

        W = self._weights.view()
        W.shape = (np.int(np.sqrt(W.shape[0])), np.int(np.sqrt(W.shape[0])))

        maxk = min(maxk, X_train.shape[0])  # K can't be > numcases in X_train

        numqueries = X_test.shape[0]

        precomp = np.dot(W, X_train.T)

        # compute similarity scores
        s = np.dot(X_test, precomp)

        # argsort sorts in ascending order
        # so we need to reverse the second dimension
        ind = np.argsort(s, axis=1)[:, ::-1]

        # Voting based on nearest neighbours
        # make sure it is int

        # Newer version of ndarray.astype takes a copy keyword argument
        # With this, we won't have to check
        if y_train.dtype.kind != 'int':
            queryvotes = y_train[ind[:, :maxk]].astype('int')
        else:
            queryvotes = y_train[ind[:, :maxk]]

        errsum = np.empty((maxk,))

        for kk in xrange(maxk):
            # AFAIK bincount only works on vectors
            # so we must loop here over data items
            anti_labels = np.empty((numqueries,), dtype='int')
            for i in xrange(numqueries):
                b = np.bincount(queryvotes[i, :kk + 1])
                anti_labels[i] = -1*b[int(y_train)-1]  # get (anti)-distance value from the current experiment class

            # errors = anti_labels != y_test
            # errsum[kk] = sum(errors)

        # print("y_test : " + str(y_test) + " \n labels : " + str(labels))
        # errrate = errsum / numqueries
        return anti_labels


    def predict(self, X_test, X_train, y_test, y_train, maxk=200):
        '''
        Evaluate an OASIS model by KNN classification
        Examples are in rows
        '''

        W = self._weights.view()
        W.shape = (np.int(np.sqrt(W.shape[0])), np.int(np.sqrt(W.shape[0])))

        maxk = min(maxk, X_train.shape[0])  # K can't be > numcases in X_train

        numqueries = X_test.shape[0]

        precomp = np.dot(W, X_train.T)

        # compute similarity scores
        s = np.dot(X_test, precomp)

        # argsort sorts in ascending order
        # so we need to reverse the second dimension
        ind = np.argsort(s, axis=1)[:, ::-1]

        # Voting based on nearest neighbours
        # make sure it is int

        # Newer version of ndarray.astype takes a copy keyword argument
        # With this, we won't have to check
        if y_train.dtype.kind != 'int':
            queryvotes = y_train[ind[:, :maxk]].astype('int')
        else:
            queryvotes = y_train[ind[:, :maxk]]

        errsum = np.empty((maxk,))

        for kk in xrange(maxk):
            # AFAIK bincount only works on vectors
            # so we must loop here over data items
            labels = np.empty((numqueries,), dtype='int')
            for i in xrange(numqueries):
                b = np.bincount(queryvotes[i, :kk + 1])
                labels[i] = np.argmax(b)  # get winning class

            errors = labels != y_test
            errsum[kk] = sum(errors)

        print("y_test : " + str(y_test) + " \n labels : " + str(labels))
        errrate = errsum / numqueries
        return errrate, labels


def resizeImage(image_matrix):
    r = 50.0 / image_matrix.shape[1]
    dim = (50, int(image_matrix.shape[0] * r))

    # perform the actual resizing of the image and show it
    resized_image = cv2.resize(image_matrix, dim, interpolation=cv2.INTER_AREA)
    return resized_image



def fetch_files(file_path):
    image_files = []
    file_names = glob.glob(file_path)
    for file_name in file_names:
        image_files.append(resizeImage(cv2.imread(file_name)))
    return image_files



def getFiles(path):
    """
    - returns  a dictionary of all files
    having key => value as  objectname => image path

    - returns total number of files.

    """
    imlist = {}
    count = 0
    for each in glob(path + "*"):
        word = each.split("/")[-1]
        print " #### Reading image category ", word, " ##### "
        imlist[word] = []
        for imagefile in glob(path + word + "/*"):
            print "Reading file ", imagefile
            im = cv2.imread(imagefile, 0)
            imlist[word].append(im)
            count += 1

    return [imlist, count]


def getNextImage(train_image, X_test, num_options, NUM_PIXELS, current_image_exp_num, iteration):

    #train is only current image
    X_train = np.zeros((1, NUM_PIXELS))
    y_train = np.array([str(current_image_exp_num+1)])


    X_train[0, :] = (resizeImage(train_image)).flatten()[0:NUM_PIXELS]

    if(iteration == 0):
        model = Oasis(n_iter=1000, do_psd=True, psd_every=3, save_path="oasis_walk_model_" + str(iteration))\
                        .fit(X_train, y_train,verbose=True)
    else:
        model = Oasis(n_iter=1000, do_psd=True, psd_every=3, save_path="oasis_walk_model_" + str(iteration))
        model.read_snapshot("oasis_walk_model_" + str(iteration-1) + "/model0010.pklz")
        model.fit(X_train, y_train,verbose=True)

    distances = model.getFarthestLabels(X_test, X_train, y_train, maxk=6)
    return np.argmax(distances)



def image_walk():
    current_image_exp_num = randint(0, 6)

    TRAIN_PATH = "../images/large_set/train/"
    TEST_PATH = "../images/large_set/test/"

    NUM_PIXELS = 50 * 50
    cnt = 0

    # train_files, num_files_train = getFiles(TRAIN_PATH)
    next_image_options, num_options = getFiles(TRAIN_PATH)

    # test is all images
    X_test = np.zeros((num_options, NUM_PIXELS))
    y_test = []

    current_train_image = next_image_options[str(current_image_exp_num)][0]


    tcount = -1
    for i in range(len(next_image_options)):
        image_category_chunk = next_image_options[str(i + 1)]
        for j in range(0, len(image_category_chunk)):
            single_image = image_category_chunk[j]
            tcount = tcount + 1
            X_test[tcount, :] = (resizeImage(single_image)).flatten()[0:NUM_PIXELS]
            y_test.append(i+1)
            # print i, j, tcount


    while(cnt < 10):
        next_exp_num = getNextImage(current_train_image, next_image_options, num_options, NUM_PIXELS, current_image_exp_num, cnt)
        print "Suggested Next Image Number : " + next_exp_num
        current_image_exp_num = next_exp_num
        current_train_image = train_files[str(next_exp_num)][0]
        cnt = cnt+1


def cells_main():
    """Function for cells data"""

    TRAIN_PATH = "../images/large_set/train/"
    TEST_PATH = "../images/large_set/test/"

    NUM_PIXELS = 50 * 50

    train_files, num_files_train = getFiles(TRAIN_PATH)
    test_files, num_files_test = getFiles(TEST_PATH)

    X_train = np.zeros((num_files_train, NUM_PIXELS))
    y_train = []
    X_test = np.zeros((num_files_test, NUM_PIXELS))
    y_test = []

    tcount = -1
    for i in range(len(train_files)):
        image_category_chunk = train_files[str(i+1)]
        for j in range(0,len(image_category_chunk)):
            single_image = image_category_chunk[j]
            tcount = tcount + 1
            X_train[tcount, :] = (resizeImage(single_image)).flatten()[0:NUM_PIXELS]
            y_train.append(i + 1)
            # print i,j,tcount

    tcount = -1
    for i in range(len(test_files)):
        image_category_chunk = test_files[str(i + 1)]
        for j in range(0, len(image_category_chunk)):
            single_image = image_category_chunk[j]
            tcount = tcount + 1
            X_test[tcount, :] = (resizeImage(single_image)).flatten()[0:NUM_PIXELS]
            y_test.append(i+1)
            # print i, j, tcount

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print X_train.shape, X_test.shape, y_train.shape, y_test.shape
    # exit(0)

    print("\n\nX_train.shape : " + str(X_train.shape) + "\n" + "X_test.shape : " + str(
        X_test.shape) + "\n" + "y_train.shape : " + str(y_train.shape) + "\n" + "y_test.shape : " + str(y_test.shape))
    # print(X_train[0].shape)
    # exit(0)

    print "Reached 1"
    # model = Oasis(n_iter=1000, do_psd=True, psd_every=3, save_path="oasis_model_cells").fit(X_train, y_train, verbose=True)

    model = Oasis(n_iter=1000, do_psd=True, psd_every=3, save_path="oasis_model_cells2")

    # model_pkl = gzip.open('oasis/oasis_test/model0010.pklz', 'rb')
    # # with open('oasis/oasis_test/model0010.pklz', 'rb') as fid:
    # model = pickle.load(model_pkl)

    model.read_snapshot("oasis_model_cells/model0010.pklz")

    errrate, labels = model.predict(X_test, X_train, y_test, y_train, maxk=2)

    print labels
    print "Min error rate: %6.4f at k=%d" % (min(errrate), np.argmin(errrate) + 1)

    plt.figure()
    plt.plot(errrate)

    n_features = X_train.shape[1]
    W = model._weights.view()
    W.shape = (n_features, n_features)

    # print W[0:5, 0:5]








#
# def dog_cat_main():
#     # from sklearn import datasets
#     # digits = datasets.load_digits()
#     #
#     # X_train = digits.data[500:] / 16
#     # X_test = digits.data[:500] / 16
#     # y_train = digits.target[500:]
#     # y_test = digits.target[:500]
#
#     DATASET_CLASS_PATH = "VOC2007/ImageSets/Main/"
#     DATASET_ANNOTATIONS_PATH = "VOC2007/Annotations/"
#     IMAGE_PATH = "VOC2007/JPEGImages/"
#     NUM_PIXELS = 50 * 50
#     dog_train_ids = voc_preprocessor.preprocessData(DATASET_CLASS_PATH, DATASET_ANNOTATIONS_PATH, "dog", "train")
#     dog_val_ids = voc_preprocessor.preprocessData(DATASET_CLASS_PATH, DATASET_ANNOTATIONS_PATH, "dog", "val")
#     cat_train_ids = voc_preprocessor.preprocessData(DATASET_CLASS_PATH, DATASET_ANNOTATIONS_PATH, "cat", "train")
#     cat_val_ids = voc_preprocessor.preprocessData(DATASET_CLASS_PATH, DATASET_ANNOTATIONS_PATH, "cat", "val")
#     # X_train = np.zeros((len(filtered_ids), ))
#
#     X_train = np.zeros((4, NUM_PIXELS))
#     y_train = np.array([1, 0, 1, 0])
#     X_test = np.zeros((4, NUM_PIXELS))
#     y_test = np.array([0, 1, 0, 1])
#
#     dog1 = resizeImage(cv2.imread(IMAGE_PATH + dog_train_ids[0] + ".jpg"))
#     dog2 = resizeImage(cv2.imread(IMAGE_PATH + dog_train_ids[1] + ".jpg"))
#     dog3 = resizeImage(cv2.imread(IMAGE_PATH + dog_val_ids[0] + ".jpg"))
#     dog4 = resizeImage(cv2.imread(IMAGE_PATH + dog_val_ids[1] + ".jpg"))
#
#     cat1 = resizeImage(cv2.imread(IMAGE_PATH + cat_train_ids[0] + ".jpg"))
#     cat2 = resizeImage(cv2.imread(IMAGE_PATH + cat_train_ids[1] + ".jpg"))
#     cat3 = resizeImage(cv2.imread(IMAGE_PATH + cat_val_ids[0] + ".jpg"))
#     cat4 = resizeImage(cv2.imread(IMAGE_PATH + cat_val_ids[1] + ".jpg"))
#
#     # print dog2.flatten().shape
#     X_train[0, :] = dog1.flatten()[0:NUM_PIXELS]
#     X_train[1, :] = cat1.flatten()[0:NUM_PIXELS]
#     X_train[2, :] = dog2.flatten()[0:NUM_PIXELS]
#     X_train[3, :] = cat2.flatten()[0:NUM_PIXELS]
#
#     X_test[0, :] = dog3.flatten()[0:NUM_PIXELS]
#     X_test[1, :] = cat3.flatten()[0:NUM_PIXELS]
#     X_test[2, :] = dog4.flatten()[0:NUM_PIXELS]
#     X_test[3, :] = cat4.flatten()[0:NUM_PIXELS]
#
#
#     print("\n\nX_train.shape : " + str(X_train.shape) + "\n" + "X_test.shape : " + str(X_test.shape) + "\n" + "y_train.shape : " + str(y_train.shape) + "\n" + "y_test.shape : " + str(y_test.shape))
#     # print(X_train[0].shape)
#     # exit(0)
#
#     print "Reached 1"
#     # model = Oasis(n_iter=1000, do_psd=True, psd_every=3, save_path="oasis/oasis_test").fit(X_train, y_train, verbose=True)
#     # model = Oasis()
#
#     model = Oasis(n_iter=1000, do_psd=True, psd_every=3, save_path="oasis/oasis_test_cells")
#
#     # model_pkl = gzip.open('oasis/oasis_test/model0010.pklz', 'rb')
#     # # with open('oasis/oasis_test/model0010.pklz', 'rb') as fid:
#     # model = pickle.load(model_pkl)
#
#     # model.read_snapshot("oasis/oasis_test/model0010.pklz")
#
#     errrate, labels = model.predict(X_test, X_train, y_test, y_train, maxk=2)
#
#     print labels
#     print "Min error rate: %6.4f at k=%d" % (min(errrate), np.argmin(errrate) + 1)
#
#     plt.figure()
#     plt.plot(errrate)
#
#     n_features = X_train.shape[1]
#     W = model._weights.view()
#     W.shape = (n_features, n_features)
#
#     # print W[0:5, 0:5]

# cells_main()