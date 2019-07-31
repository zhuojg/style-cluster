import numpy as np
from sklearn.externals import joblib
from sklearn.decomposition import PCA
import os
from model.style_model import StyleModel
import time


def time_convert(second):
    minute = 0
    hour = 0
    if second >= 60:
        minute = int(second / 60)
        second = second % 60
    if minute >= 60:
        hour = int(minute/60)
        minute = minute % 60
    if hour > 0:
        return '%s h %s m %s s' % (str(hour), str(minute), str(second))
    elif minute > 0:
        return '%s m %s s' % (str(minute), str(second))
    else:
        return '%s s' % str(int(second))


def get_image_list(train_path):
    training_names = os.listdir(train_path)
    image_paths = []
    for training_name in training_names:
        if training_name.split('/')[-1] == '.DS_Store':
            continue
        image_path = os.path.join(train_path, training_name)
        image_paths += [image_path]
    return image_paths


def get_image_list_recursion(train_path):
    training_names = os.listdir(train_path)

    image_paths = []
    for training_name in training_names:
        if training_name.split('/')[-1] == '.DS_Store':
            continue
        path = os.path.join(train_path, training_name)

        if os.path.isdir(path):
            image_paths.extend(get_image_list(path))
        else:
            image_paths += [path]
    return image_paths


class StyleFeatureCalculator:
    def __init__(self, model, image_path, result_path, cnt=0):
        self.model = model
        self.cnt = cnt
        self.image_path = image_path
        self.result_path = result_path

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

    def batch_extract_feature(self, image_paths):
        features = []
        images = []
        step = 200
        start_index = 0
        cnn_feature_path = os.path.join(self.result_path, 'cnn_features.pkl')
        if os.path.exists(cnn_feature_path):
            before_images, _ = joblib.load(cnn_feature_path)
            start_index = len(before_images)
            print('start index: %s' % str(start_index))

        start_time = time.time()
        for i, path in enumerate(image_paths):
            if i < start_index:
                continue

            try:
                features.append(self.model.extract_feature(path))
                images.append(path)
                # print("get {} image feature success!".format(path))
            except Exception as e:
                print("get {} image feature failed!".format(path), e)
            if i - start_index > 0 and i % 10 == 0:
                time_used = time.time() - start_time
                time_remained = time_used * (len(image_paths) - i - start_index) / (i - start_index)
                print('%s / %s, time used: %s, time remained: %s'
                      % (str(i), str(len(image_paths)), time_convert(time_used), time_convert(time_remained)))
            if i >= self.cnt > 0:
                break

            if i > 0 and i % step == 0:
                # 存储直接由CNN提取出的特征
                if os.path.exists(cnn_feature_path):
                    before_images, before_features = joblib.load(cnn_feature_path)

                    features = np.concatenate((before_features, features), axis=0)
                    before_images.extend(images)
                    images = before_images
                    joblib.dump((images, features), cnn_feature_path)
                else:
                    features = np.array(features)
                    joblib.dump((images, features), cnn_feature_path)

                images = []
                features = []

        features = np.array(features)
        return images, features

    # mode
    # 1 - extract features using CNN
    # 2 - using existed CNN features
    def run(self, mode, cnn_features_path):
        image_paths = []
        image_paths.extend((get_image_list_recursion(self.image_path)))

        if mode == 1:
            images, features = self.batch_extract_feature(image_paths)
        else:
            images, features = joblib.load(cnn_features_path)

        # # t-SNE
        # tsne = TSNE(init='pca', random_state=0)
        # tsne.fit(features[:3000])
        # joblib.dump(tsne, os.path.join(self.result_path, 'tsne_result.pkl'))

        # PCA
        pca = PCA(n_components=1024, whiten=False, copy=False)
        pca.fit(features)
        joblib.dump(pca, os.path.join(self.result_path, 'pca_result.pkl'))

        # # PQ
        # pq = faiss.ProductQuantizer(131328, 2052, 64)
        # pq.train(features)
        # pq_features = pq.compute_codes(features)
        # joblib.dump((images, pq_features), os.path.join(self.result_path, 'pq_features.pkl'))


        # calculate features using t-SNE
        # tsne = joblib.load(os.path.join(path_prefix, 'tsne-result.pkl'))
        # tsne_features = tsne.fit_transform(features)
        # joblib.dump((images, tsne_features), os.path.join(self.result_path, 'tsne_features.pkl'))

        # calculate features using PCA
        pca_features = pca.fit_transform(features)
        joblib.dump((images, pca_features), os.path.join(self.result_path, 'pca_features.pkl'))
        # lle = LocallyLinearEmbedding(n_components=1024)
        # lle_features = lle.fit_transform(features)
        # joblib.dump((images, lle_features), os.path.join(self.result_path, 'lle_features.pkl'))


if __name__ == '__main__':
    styleModel = StyleModel()

    # remember to change the image_path and result_path!
    job = StyleFeatureCalculator(styleModel,
                                 image_path='../style_data',
                                 result_path='../result-lab-data')

    job.run(mode=1, cnn_features_path='../result-lab-data/cnn_features.pkl')
