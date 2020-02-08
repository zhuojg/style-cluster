import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
import numpy as np
from torchvision.transforms import transforms
import time
from PIL import Image
from PIL import ImageFile
from sklearn.preprocessing import normalize as sknormalize
from utils import rmac


ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def image_loader(image_name):
    im = Image.open(image_name)
    im = im.convert('RGB')
    im_size_hw = np.array(im.size[::-1])

    max_side_lengths = [800]
    images = []
    for max_side_length in max_side_lengths:
        ratio = float(max_side_length) / np.max(im_size_hw)
        new_size = tuple(np.round(im_size_hw * ratio.astype(float)).astype(np.int32))
        # fake batch dimension required to fit network's input dimensions
        loader = transforms.Compose(
            [
                transforms.Resize(new_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        image = loader(im).unsqueeze(0)
        image_cuda = image.to(device, torch.float)
        images.append(image)
    return images


def image_loader_eval(image_name):
    im = Image.open(image_name)
    im = im.convert('RGB')
    im_size_hw = np.array(im.size[::-1])

    max_side_lengths = [550, 800, 1050]
    images = []
    for max_side_length in max_side_lengths:
        ratio = float(max_side_length) / np.max(im_size_hw)
        new_size = tuple(np.round(im_size_hw * ratio.astype(float)).astype(np.int32))
        # fake batch dimension required to fit network's input dimensions
        loader = transforms.Compose(
            [
                transforms.Resize(new_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        image = loader(im).unsqueeze(0)
        images.append(image)
    return images


class SiameseNetwork(nn.Module):
    def __init__(self, pretrained=True):
        super(SiameseNetwork, self).__init__()
        self.pretrained_model = models.resnet34(pretrained=pretrained)
        self.cnn1 = nn.Sequential(*list(self.pretrained_model.children())[:-2])
        self.pool = rmac
        self.normal = nn.functional.normalize

    def forward_once(self, x):
        x = self.cnn1(x)
        x = self.pool(x)
        return self.normal(x, ).squeeze(-1).squeeze(-1)

    def style_vec(self, x):
        x = self.cnn1(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2

    def __repr__(self):
        return self.__class__.__name__ + '(resnet50+rmac+l2n)'


class StyleModel:
    def __init__(self, finetuning=False, file_path=None):
        if not finetuning:
            self.net = SiameseNetwork().to(device).eval()
        else:
            net = SiameseNetwork(False).to(device)
            net.load_state_dict(torch.load(file_path))
            self.net = net.eval()

    def normalize(self, x, copy=False):
        """
        A helper function that wraps the function of the same name in sklearn.
        This helper handles the case of a single column vector.
        """
        if type(x) == np.ndarray and len(x.shape) == 1:
            return np.squeeze(sknormalize(x.reshape(1, -1), copy=copy))
        else:
            return sknormalize(x, copy=copy)

    def extract_style_feature(self, image_path):
        imgs = image_loader(image_path)
        final_feature = []
        for img in imgs:
            features = self.net.style_vec(img.to(device)).data.cpu().numpy()
            features = np.transpose(features, [0, 2, 3, 1])
            features = self.get_style_gram(features)
            final_feature.append(features)

        return self.normalize(np.array(final_feature, dtype=np.float32).sum(axis=0)).squeeze()

    def extract_feature(self, image_path):
        return self.extract_style_feature(image_path)

    def get_style_gram(self, style_features):
        """
        get the gram matrix
        :param style_features:
        :return:
        """
        _, height, width, channel = style_features.shape[0], style_features.shape[1], style_features.shape[2], \
                                    style_features.shape[3]
        size = height * width * channel
        style_features = np.reshape(style_features, (-1, channel))
        style_features_t = np.transpose(style_features)
        style_gram = np.matmul(style_features_t, style_features) / size
        gram = style_gram.astype(np.float32)
        vector = []
        for i in range(len(gram)):
            vector.extend(gram[i][0:i + 1])
        vector = self.normalize(np.array(vector, dtype=np.float32))
        return vector


if __name__ == '__main__':
    model = StyleModel()
    since = time.time()
    feature = model.extract_feature("404.jpg")

    print(feature)
    print(len(feature))
    print(np.dot(feature, feature.T))
