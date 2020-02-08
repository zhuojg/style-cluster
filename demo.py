import style_model
import matplotlib.pyplot as plt
from PIL import Image
import joblib
import faiss
import os
from sklearn.cluster import DBSCAN
import shutil
import argparse


class Cluster:
    def __init__(self, feature_path, center_num, num_per_category):
        print('Start clustering...')
        self.image_paths, self.features = joblib.load(
            os.path.join(feature_path)
        )
        self.features = self.features.astype('float32')
        self.center_num = center_num
        d = self.features.shape[1]

        # construct kmeans using faiss
        self.kmeans = faiss.Kmeans(d, self.center_num, niter=100, verbose=True)
        self.kmeans.train(self.features)
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.features)

        # get the results
        # D contains the squared L2 distances
        # I contains the nearest neighbors for each centroid
        self.D, self.I = self.index.search(self.kmeans.centroids, num_per_category)

        self.img_paths = []
        self.scores = []
        # so we enumerate I, to get every centroid's neighbors
        for i, item in enumerate(self.I):
            temp_paths = []
            temp_scores = []
            for j, index in enumerate(item):
                temp_paths.append(self.image_paths[index])
                temp_scores.append(self.D[i][j])
            self.img_paths.append(temp_paths)
            self.scores.append(temp_scores)

        print('Job done.')

    def show_in_pic(self, save_path, data_path):
        """
        generate a big picture for every cluster using images in this cluster

        :param save_path:
        :return:
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i, cluster in enumerate(self.img_paths):
            plt.figure()
            for j, img in enumerate(cluster):
                plt.subplot(10, 5, j + 1)

                image = Image.open(os.path.join(data_path, img))
                image = image.convert('RGB')
                plt.imshow(image)
                plt.axis('off')
            plt.savefig(os.path.join(save_path, '%s.png' % str(i)), dpi=600)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate features.')
    parser.add_argument('--data_path', type=str, help='path to data', required=True)
    parser.add_argument('--feature_path', type=str, help='path to pca feature', required=True)
    parser.add_argument('--save_path', type=str, help='path to save result', required=True)
    args = parser.parse_args()

    c = Cluster(feature_path=args.feature_path, center_num=15, num_per_category=50)
    c.show_in_pic(save_path=args.save_path, data_path=args.data_path)
