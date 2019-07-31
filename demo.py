from online import search
from model import content_model, style_model
import matplotlib.pyplot as plt
from PIL import Image
import joblib
from config.config import *
import faiss
import os
from sklearn.cluster import DBSCAN
import shutil


class Cluster:
    def __init__(self, feature_path, center_num):
        self.image_paths, self.features = joblib.load(
            os.path.join(feature_path)
        )
        self.ids = [int(path.split('/')[-1].split('.')[0], base=16) for path in self.image_paths]
        self.features = self.features.astype('float32')
        self.center_num = center_num
        ncentroids = self.center_num
        niter = 100
        verbose = True
        d = self.features.shape[1]
        self.kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
        self.kmeans.train(self.features)
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.features)

    def show_in_pic(self, save_path, num_per_category):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        D, I = self.index.search(self.kmeans.centroids, num_per_category)

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
            plt.savefig(os.path.join(save_path, '%s.png' % str(i)), dpi=600)

    def copy_images(self, save_path, num_per_category):
        D, I = self.index.search(self.kmeans.centroids, num_per_category)

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
            new_path = os.path.join(save_path, str(i))
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            for j, img in enumerate(cluster):
                file_name = img.split('/')[-1]
                label = img.split('/')[-2].split('_')[0]
                # you can decide how to rename the file by yourself
                shutil.copyfile(img, os.path.join(new_path, label + '_' + file_name))


if __name__ == '__main__':
    # remember to change the feature_path and save_path!
    c = Cluster(feature_path='../result-lab-data/pca_features.pkl', center_num=15)
    c.show_in_pic(save_path='../result-lab-data', num_per_category=50)
