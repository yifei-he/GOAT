import torch
from torch.utils.data import Dataset
import scipy.io
import numpy as np
import sklearn.preprocessing
from scipy import ndimage
import pandas as pd
from tensorflow.keras.datasets import mnist


class DomainDataset(Dataset):
    def __init__(self, x, weight, transform=None):
        self.data = x.cpu().detach()
        self.targets = -1 * torch.ones(len(self.data))
        self.weight = weight
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.data[idx]), self.targets[idx], self.weight[idx]
        return self.data[idx], self.targets[idx], self.weight[idx]


class EncodeDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.data = x
        self.targets = y
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform is not None:
            # if isinstance(self.data[idx], np.ndarray):
            #     # data = Image.fromarray((self.data[idx].squeeze(2)*255).astype(np.uint8))
            #     data = Image.fromarray((self.data[idx]*255).astype(np.uint8))
            #     return self.transform(data).float(), self.targets[idx]
            # else:
            return self.transform(self.data[idx]).float(), self.targets[idx]
        return self.data[idx], self.targets[idx]


"""
    Make portraits dataset
"""
def shuffle(xs, ys):
    indices = list(range(len(xs)))
    np.random.shuffle(indices)
    return xs[indices], ys[indices]


def split_sizes(array, sizes):
    indices = np.cumsum(sizes)
    return np.split(array, indices)


def load_portraits_data(load_file='dataset_32x32.mat'):
    data = scipy.io.loadmat('./' + load_file)
    return data['Xs'], data['Ys'][0]

def make_portraits_data(n_src_tr, n_src_val, n_inter, n_target_unsup, n_trg_val, n_trg_tst,
                        load_file='dataset_32x32.mat'):
    xs, ys = load_portraits_data(load_file)
    src_end = n_src_tr + n_src_val
    inter_end = src_end + n_inter
    trg_end = inter_end + n_trg_val + n_trg_tst
    src_x, src_y = shuffle(xs[:src_end], ys[:src_end])
    trg_x, trg_y = shuffle(xs[inter_end:trg_end], ys[inter_end:trg_end])
    [src_tr_x, src_val_x] = split_sizes(src_x, [n_src_tr])
    [src_tr_y, src_val_y] = split_sizes(src_y, [n_src_tr])
    [trg_val_x, trg_test_x] = split_sizes(trg_x, [n_trg_val])
    [trg_val_y, trg_test_y] = split_sizes(trg_y, [n_trg_val])
    inter_x, inter_y = xs[src_end:inter_end], ys[src_end:inter_end]
    dir_inter_x, dir_inter_y = inter_x[-n_target_unsup:], inter_y[-n_target_unsup:]
    return (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y,
            dir_inter_x, dir_inter_y, trg_val_x, trg_val_y, trg_test_x, trg_test_y)


"""
    make covertype dataset
"""
def make_data(n_src_tr, n_src_val, n_inter, n_target_unsup, n_trg_val, n_trg_tst, xs, ys):
    src_end = n_src_tr + n_src_val
    inter_end = src_end + n_inter
    trg_end = inter_end + n_trg_val + n_trg_tst
    src_x, src_y = shuffle(xs[:src_end], ys[:src_end])
    trg_x, trg_y = shuffle(xs[inter_end:trg_end], ys[inter_end:trg_end])
    [src_tr_x, src_val_x] = split_sizes(src_x, [n_src_tr])
    [src_tr_y, src_val_y] = split_sizes(src_y, [n_src_tr])
    [trg_val_x, trg_test_x] = split_sizes(trg_x, [n_trg_val])
    [trg_val_y, trg_test_y] = split_sizes(trg_y, [n_trg_val])
    inter_x, inter_y = xs[src_end:inter_end], ys[src_end:inter_end]
    dir_inter_x, dir_inter_y = inter_x[-n_target_unsup:], inter_y[-n_target_unsup:]
    return (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y,
            dir_inter_x, dir_inter_y, trg_val_x, trg_val_y, trg_test_x, trg_test_y)


def load_covtype_data(load_file, normalize=True):
    df = pd.read_csv(load_file, header=None)
    data = df.to_numpy()
    xs = data[:, :54]
    if normalize:
        xs = (xs - np.mean(xs, axis=0)) / np.std(xs, axis=0)
    ys = data[:, 54] - 1

    # Keep the first 2 types of crops, these comprise majority of the dataset.
    keep = (ys <= 1)
    print(len(xs))
    xs = xs[keep]
    ys = ys[keep]
    print(len(xs))

    # Sort by (horizontal) distance to water body.
    dist_to_water = xs[:, 3]
    indices = np.argsort(dist_to_water, axis=0)
    xs = xs[indices]
    ys = ys[indices]
    return xs, ys

def make_cov_data(n_src_tr, n_src_val, n_inter, n_target_unsup, n_trg_val, n_trg_tst,
                  load_file="covtype.data", normalize=True):
    xs, ys = load_covtype_data(load_file)
    return make_data(n_src_tr, n_src_val, n_inter, n_target_unsup, n_trg_val, n_trg_tst, xs, ys)


def cov_data_func():
    return make_cov_data(40000, 10000, 400000, 50000, 25000, 20000)

def cov_data_small_func():
    return make_cov_data(10000, 40000, 400000, 50000, 25000, 20000)

def cov_data_func_no_normalize():
    return make_cov_data(40000, 10000, 400000, 50000, 25000, 20000, normalize=False)

"""
    Make Color-shift MNIST dataset
"""
def shift_color_images(xs, shift):
    return xs + shift

def get_preprocessed_mnist():
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_x, test_x = train_x / 255.0, test_x / 255.0
    train_x, train_y = shuffle(train_x, train_y)
    train_x = np.expand_dims(np.array(train_x), axis=-1)
    test_x = np.expand_dims(np.array(test_x), axis=-1)
    return (train_x, train_y), (test_x, test_y)

def ColorShiftMNIST(shift=10):
    (train_x, train_y), (test_x, test_y) = get_preprocessed_mnist()
    src_train_end, src_val_end, inter_end, target_end = 5000, 6000, 48000, 50000
    src_tr_x, src_tr_y = train_x[:src_train_end], train_y[:src_train_end]
    src_val_x, src_val_y = train_x[src_train_end:src_val_end], train_y[src_train_end:src_val_end]
    dir_inter_x, dir_inter_y = train_x[src_val_end:inter_end], train_y[src_val_end:inter_end]
    trg_val_x, trg_val_y = train_x[inter_end:target_end], train_y[inter_end:target_end]
    trg_test_x, trg_test_y = test_x, test_y
    trg_val_x, trg_test_x = shift_color_images(trg_val_x, shift), shift_color_images(trg_test_x, shift)
    return (src_tr_x, src_tr_y, src_val_x, src_val_y, dir_inter_x, dir_inter_y,
            dir_inter_x, dir_inter_y, trg_val_x, trg_val_y, trg_test_x, trg_test_y)


def transform_inter_data(dir_inter_x, dir_inter_y, source_scale, target_scale, transform_func=shift_color_images, interval=2000, n_domains=20, n_classes=10, class_balanced=False, reverse_point=None):
        all_domain_x = []
        all_domain_y = []
        path_length = target_scale - source_scale
        if reverse_point is not None:
            assert reverse_point >= source_scale and reverse_point <= target_scale
            path_length += reverse_point * 2
        scales = source_scale + np.flip(np.linspace(path_length,0,n_domains))
        for domain_idx in range(n_domains):
            domain_scale = source_scale + path_length / n_domains * (domain_idx + 1)
            if class_balanced:
                domain_data_idxes = []
                n_domain_class_data = int(interval / n_classes)
                for label in range(n_classes):
                    class_idxes = np.where(dir_inter_y == label)[0]
                    domain_data_idxes.append(np.random.choice(class_idxes, n_domain_class_data, replace=False))
                domain_data_idxes = np.concatenate(domain_data_idxes, axis=0)
            else:
                domain_data_idxes = np.random.choice(dir_inter_x.shape[0], interval, replace=False)
            domain_x = dir_inter_x[domain_data_idxes]
            domain_y = dir_inter_y[domain_data_idxes]
            domain_x = transform_func(domain_x, domain_scale)
            all_domain_x.append(domain_x)
            all_domain_y.append(domain_y)
        all_domain_x = np.concatenate(all_domain_x, axis=0)
        all_domain_y = np.concatenate(all_domain_y, axis=0)
        return all_domain_x, all_domain_y
