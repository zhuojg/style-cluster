import os

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


def rmac(x, L=3, eps=1e-6):
    ovr = 0.4  # desired overlap of neighboring regions
    steps = torch.Tensor([2, 3, 4, 5, 6, 7])  # possible regions for the long dimension

    W = x.size(3)
    H = x.size(2)

    w = min(W, H)
    w2 = math.floor(w / 2.0 - 1)

    b = (max(H, W) - w) / (steps - 1)
    (tmp, idx) = torch.min(torch.abs(((w ** 2 - w * b) / w ** 2) - ovr), 0)  # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd = 0
    Hd = 0
    if H < W:
        Wd = idx.item() + 1
    elif H > W:
        Hd = idx.item() + 1

    v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
    v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v)

    for l in range(1, L + 1):
        wl = math.floor(2 * w / (l + 1))
        wl2 = math.floor(wl / 2 - 1)

        if l + Wd == 1:
            b = 0
        else:
            b = (W - wl) / (l + Wd - 1)
        cenW = torch.floor(wl2 + torch.Tensor(range(l - 1 + Wd + 1)) * b) - wl2  # center coordinates
        if l + Hd == 1:
            b = 0
        else:
            b = (H - wl) / (l + Hd - 1)
        cenH = torch.floor(wl2 + torch.Tensor(range(l - 1 + Hd + 1)) * b) - wl2  # center coordinates

        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                R = x[:, :, (int(i_) + torch.Tensor(range(wl)).long()).tolist(), :]
                R = R[:, :, :, (int(j_) + torch.Tensor(range(wl)).long()).tolist()]
                vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + eps).expand_as(vt)
                v += vt

    return v