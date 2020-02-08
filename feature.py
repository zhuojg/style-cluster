import numpy as np
import joblib
from sklearn.decomposition import PCA
import os
from style_model import StyleModel
import time
from utils import time_convert, get_image_list, get_image_list_recursion
import argparse


def get_style_feature(model, image_path, result_path, cnt=0):
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    image_path = get_image_list_recursion(image_path)

    # only save the file name and category name
    new_image_path = [item.split('/')[-2] + '/' + item.split('/')[-1] for item in image_path]
    image_path = new_image_path

    cnn_feature_path = os.path.join(result_path, 'cnn_features.pkl')

    step = 200      # save result and print status every 200 images

    # if the cnn_feature exists, start from the stop position
    if os.path.exists(cnn_feature_path):
        images, features = joblib.load(cnn_feature_path)
        start_index = len(images)
        print('start index: %s' % str(start_index))
    else:
        features = []
        images = []
        start_index = 0

    for i, path in enumerate(image_path):
        if i < start_index:
            continue
        
        try:
            features.append(model.extract_feature(path))
            images.append(path)
        except Exception as e:
            print('get {} image feature failed: '.format(path), e)
        
        if i - start_index > 0 and i % 10 == 0:
            print('%d / %d' % (i, len(image_path)))
        
        if i >= cnt > 0:
            break
        
        if i > 0 and i % step == 0:
            features = np.array(features)
            joblib.dump((images, features), cnn_feature_path)
        
    features = np.array(features)
    return images, features


def pca(images, features, n_components, result_path):
    print('Constructing PCA model...')
    # construct pca model
    model = PCA(n_components=n_components, whiten=False, copy=False)
    print('Fitting the features...')
    model.fit(features)
    joblib.dump(model, os.path.join(result_path, 'pca_result.pkl'))
    print('Result stored in %s' % os.path.join(result_path, 'pca_result.pkl'))

    print('Processing features using PCA...')
    # process features using PCA
    pca_features = model.fit_transform(features)
    joblib.dump((images, pca_features), os.path.join(result_path, 'pca_features.pkl'))
    print('Result stored in %s' % os.path.join(result_path, 'pca_features.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate features.')
    flag_parser = parser.add_mutually_exclusive_group(required=False)
    flag_parser.add_argument('--exist', dest='exist', action='store_true',
                             help='use existed cnn features to calculate pca features')
    flag_parser.add_argument('--no-exist', dest='exist', action='store_false',
                             help='run get_style_feature function')
    parser.set_defaults(exist=False)

    parser.add_argument('--result_path', type=str, help='path to result files', required=True)
    parser.add_argument('--data_path', type=str, help='path to data', required=True)

    args = parser.parse_args()

    if args.exist:
        images, features = joblib.load(os.path.join(args.result_path, 'cnn_features.pkl'))
    else:
        style_model = StyleModel()
        images, features = get_style_feature(
            model=style_model,
            image_path=args.data_path,
            result_path=args.result_path
        )

    pca(images, features, 1024, args.result_path)
