import numpy as np

class CRFGraph:
    def __init__(self, vertset, eset):
        self.vertices = vertset
        self.edges = eset


def constructCRFGraph(grid):
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
    all_edges = np.sort(all_edges ,axis=1)
    num_vertices = len(vertices)
    edge_hash = all_edges[:, 0] + num_vertices * all_edges[:, 1]
    # find unique connections
    edges = np.unique(edge_hash)
    # undo hashing
    edges = [[vertices[edge % num_vertices],
              vertices[edge / num_vertices]] for edge in edges]

    return CRFGraph(vertices, edges)

def generateTrainData(graph, segments, avg_orig, avg_mask):
    meshX, meshY = np.mgrid[:segments.shape[0], :segments.shape[1]]
    centers = dict()
    for v in graph.vertices:
        centers[v] = [meshY[segments == v].mean(), meshX[segments == v].mean()]

    # Compute features
    data_train_x = []
    data_train_y = []
    n_features = []
    n_labels = []
    edge_features = []

    for v in graph.vertices:
        # unary feature - average rgb of superpixel
        avg_rgb2 = avg_orig[int(centers[v][1])][int(centers[v][0])]
        hist, hogFeatures = imagesegmentation.getHistogramFeatures(bgrImage, int(centers[v][1]), int(centers[v][0]),
                                                                   forUnaryFeature=True)
        node_feature = np.concatenate([avg_rgb2, hist, hogFeatures])
        n_features.append(node_feature)

        minEuclideanDistance = np.inf  # simulate infinity
        pixelClass = -1
        for i in range(0, len(pixelClasses)):
            # set the label of the superpixel to the pixelClass with minimum euclidean distance
            dist = numpy.linalg.norm(avg_label[int(centers[v][1])][int(centers[v][0])] - pixelClasses[i])
            if dist < minEuclideanDistance:
                pixelClass = i
                minEuclideanDistance = dist
        n_labels.append(pixelClass)



    histogramCache = {}
    for e in graph.edges:
        # pairwise feature - euclidean distance of adjacent superpixels
        dist = np.linalg.norm(
            avg_orig[int(centers[e[0]][1])][int(centers[e[0]][0])] - avg_orig[int(centers[e[1]][1])][
                int(centers[e[1]][0])])

        if e[0] not in histogramCache:
            hist1, lbphist1 = getHistogramFeatures(orig_img, int(centers[e[0]][1]), int(centers[e[0]][0]))
            histogramCache[e[0]] = {'hist': hist1, 'lbphist': lbphist1}
        else:
            hist1 = histogramCache[e[0]]['hist']
            lbphist1 = histogramCache[e[0]]['lbphist']
        if e[1] not in histogramCache:
            hist2, lbphist2 = getHistogramFeatures(orig_img, int(centers[e[1]][1]), int(centers[e[1]][0]))
            histogramCache[e[1]] = {'hist': hist2, 'lbphist': lbphist2}
        else:
            hist2 = histogramCache[e[1]]['hist']
            lbphist2 = histogramCache[e[1]]['lbphist']

        histogramDist = cv2.compareHist(hist1, hist2, 3)  # Bhattacharyya distance
        textureSimilarity = kullback_leibler_divergence(lbphist1, lbphist2)  # KL divergence

        pairwise_feature = np.array([dist, histogramDist, textureSimilarity])
        edge_features.append(pairwise_feature)

    data_train_x.append((np.array(n_features), np.array(edges), np.array(edge_features)))
    data_train_y.append(np.array(n_labels))