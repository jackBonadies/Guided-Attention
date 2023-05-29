import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt

def showSelfAttnClustering(fName):
    ca = np.load(fName)
    ca = np.reshape(ca,(256,256))
    tokens_x = []
    inertias_y = []
    sil_score = []
    for i in range(2, 15):
        tokens = i
        kmeans = KMeans(n_clusters=tokens,
                        n_init=10).fit(ca)
        print(f"{tokens} --- {kmeans.inertia_}")
        tokens_x.append(tokens)
        inertias_y.append(kmeans.inertia_)
        lbl = kmeans.labels_
        sil_score.append(silhouette_score(ca, lbl))
        finals = np.reshape(np.reshape(kmeans.labels_,(256)),(16,16))
        plt.imshow(finals, cmap='viridis')
        plt.show()

    plt.plot(tokens_x, inertias_y)
    plt.show()

    plt.plot(tokens_x, sil_score)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Requires File Name")
        exit(-1)
    fname = sys.argv[1]
    showSelfAttnClustering(fname)
    exit(0)