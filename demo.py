from online import search
from model import content_model, style_model
import matplotlib.pyplot as plt
from PIL import Image
import joblib
from config.config import *
import faiss
import os
from sklearn.cluster import DBSCAN

path_prefix = './data'


class Cluster:
    def __init__(self, feature_path, save_path, center_num):
        self.image_paths, self.features = joblib.load(
            os.path.join(feature_path)
        )
        self.ids = [int(path.split('/')[-1].split('.')[0], base=16) for path in self.image_paths]
        self.features = self.features.astype('float32')
        self.save_path = save_path
        self.center_num = center_num

    def run(self):
        ncentroids = self.center_num
        niter = 100
        verbose = True
        d = self.features.shape[1]
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
        kmeans.train(self.features)
        print(kmeans.centroids)

        index = faiss.IndexFlatL2(d)
        index.add(self.features)
        D, I = index.search(kmeans.centroids, 50)

        img_paths = []
        scores = []
        for i, item in enumerate(I):
            temp_paths = []
            temp_scores = []
            for j, index in enumerate(item):
                temp_paths.append(self.image_paths[index])
                temp_scores.append(D[i][j])
            img_paths.append(temp_paths)
            scores.append(temp_scores)

        for i, cluster in enumerate(img_paths):
            plt.figure()
            for j, img in enumerate(cluster):
                plt.subplot(10, 5, j + 1)

                image = Image.open(img[1:])
                image = image.convert('RGB')
                plt.imshow(image)
                plt.axis('off')
            plt.savefig(os.path.join(self.save_path, '%s.png' % str(i)), dpi=600)

    def run_dbscan(self):
        possible_eps = [(0.005 * i + 0.55) for i in range(1, 20)]
        for p_eps in possible_eps:
            dbscan = DBSCAN(eps=p_eps, min_samples=10, algorithm='auto')
            dbscan.fit(self.features)

            labels = dbscan.labels_
            result = {}
            for item in labels:
                if item in result.keys():
                    result[item] = result[item] + 1
                else:
                    result[item] = 0
            print('eps=%s: %s' % (str(p_eps), str(result)))

    def show_features(self):
        print(self.features)


if __name__ == '__main__':
    # remember to change the feature_path and save_path!
    c = Cluster(feature_path='./result-lab-data/pca_features.pkl', save_path='./result-lab-data', center_num=15)
    c.run()
