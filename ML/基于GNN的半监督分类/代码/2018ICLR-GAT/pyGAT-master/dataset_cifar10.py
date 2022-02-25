from __future__ import print_function
import numpy as np
import torch
import torchvision
from PIL import Image
import pickle as pkl
import os
import glob
import csv

from CNN import CNN13
from utils import img_normalize
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, Sampler
import torchvision.datasets
import itertools
import torchvision.transforms as transforms
# from MT.mean_teacher.data import *
# from MT.mean_teacher.utils import assert_exactly_one


NO_LABEL = -1


class TransformOnce:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        # out2 = self.transform(inp)
        return out1



class RandomTranslateWithReflect:
    """Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))

        return new_image




class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


def assert_exactly_one(lst):
    assert sum(int(bool(el)) for el in lst) == 1, ", ".join(str(el) for el in lst)


def relabel_dataset(dataset, labels):
    unlabeled_idxs = []
    for idx in range(len(dataset.imgs)):
        path, _ = dataset.imgs[idx]
        filename = os.path.basename(path)
        if filename in labels:
            label_idx = dataset.class_to_idx[labels[filename]]
            dataset.imgs[idx] = path, label_idx
            del labels[filename]
        else:
            dataset.imgs[idx] = path, NO_LABEL
            unlabeled_idxs.append(idx)

    if len(labels) != 0:
        message = "List of unlabeled contains {} unknown files: {}, ..."
        some_missing = ', '.join(list(labels.keys())[:5])
        raise LookupError(message.format(len(labels), some_missing))

    labeled_idxs = sorted(set(range(len(dataset.imgs))) - set(unlabeled_idxs))

    return labeled_idxs, unlabeled_idxs




class dataset_cifar10(object):
    def __init__(self, args, split = 'train'):
        self.split = split
        self.n_label = args.n_label
        self.root_dir = args.data_root
        self.n_class = 10
        self.n_sample = 10000
        self.n_val = 1000
        self.args = args
        self.train_subdir = 'train'
        self.eval_subdir = 'val'

    def create_data_loaders(self):
        channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616])
        transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ])

        datadir = os.path.join(self.root_dir, self.eval_subdir)

        dataset = torchvision.datasets.ImageFolder(datadir, transformation)

        data_loader = torch.utils.data.DataLoader(  dataset,
                                                    batch_size=self.args.batch_size,
                                                    num_workers=0,
                                                    pin_memory=True)

        return data_loader


    def feature_extract(self, data_loader):
        Encoder = CNN13()
        Encoder.cuda(0)
        Encoder.load_state_dict(torch.load('/mnt/Code/CNNmodels/cnn13_cifar10/models/best_model.ckpt'))

        # ****************************  Feature Extract  ***************************************************
        print("Feature Extract...")
        Encoder.eval()
        # with torch.no_grad():
        #     _, features = Encoder(train_image_data.cuda(0))
        #     _, test_features = Encoder(test_image_data.cuda(0))
        with torch.no_grad():
            features = None
            labels = None
            for i, (input, target) in enumerate(data_loader):
                _, image_feature = Encoder(input.cuda())
                if features is None:
                    features = image_feature
                    labels = target.cuda()
                else:
                    features = torch.cat((features, image_feature), dim=0)
                    labels = torch.cat((labels, target.cuda()), dim=0)

            torch.cuda.empty_cache()

        print("features:", features.shape)  # (10000, 128)
        print("label:", labels.shape)  # (10000)

        return features, labels

    def load_data(self):
        data_loader = self.create_data_loaders()
        features, labels = self.feature_extract(data_loader)


        train_set = []
        val_set = []
        test_set = []
        train_label = []
        val_label = []
        test_label = []
        ind_train = int(self.n_label / self.n_class)
        ind_val = int((self.n_label + self.n_val) / self.n_class)

        for i in range(10):
            class_set_ind = torch.nonzero(labels == i).squeeze()
            class_set = torch.index_select(features, 0, class_set_ind)
            class_set_labels = torch.index_select(labels, 0, class_set_ind)

            train_set.append(class_set[:ind_train])
            val_set.append(class_set[ind_train:ind_val])
            test_set.append(class_set[ind_val:])

            train_label.append(class_set_labels[:ind_train])
            val_label.append(class_set_labels[ind_train:ind_val])
            test_label.append(class_set_labels[ind_val:])
            continue

        train_set = torch.cat(train_set, dim=0)
        val_set = torch.cat(val_set, dim=0)
        test_set = torch.cat(test_set, dim=0)

        train_label = torch.cat(train_label, dim=0)
        val_label = torch.cat(val_label, dim=0)
        test_label = torch.cat(test_label, dim=0)

        train_set = train_set.cpu().numpy()
        val_set = val_set.cpu().numpy()
        test_set = test_set.cpu().numpy()
        train_label = train_label.cpu().numpy().astype(int)
        val_label = val_label.cpu().numpy().astype(int)
        test_label = test_label.cpu().numpy().astype(int)

        np.random.seed(200)
        np.random.shuffle(train_set)
        np.random.seed(200)
        np.random.shuffle(val_set)
        np.random.seed(200)
        np.random.shuffle(test_set)
        np.random.seed(200)
        np.random.shuffle(train_label)
        np.random.seed(200)
        np.random.shuffle(val_label)
        np.random.seed(200)
        np.random.shuffle(test_label)

        features = np.concatenate([train_set, val_set, test_set])
        labels = np.concatenate([train_label, val_label, test_label])

        # one-hot labels
        # one_labels = np.zeros([self.n_sample, self.n_class], dtype=np.uint8)  # label -> (N,c)
        # for ind, key in zip(range(self.n_sample), labels):
        #     one_labels[ind][key] = 1
        # labels = one_labels

        train_index = range(0, self.n_label)
        val_index = range(self.n_label, self.n_label + self.n_val)
        test_index = range(self.n_label + self.n_val, self.n_sample)

        # # one-hot mask
        # train_mask = np.zeros([self.n_sample], dtype=np.uint8)
        # for i in train_index:
        #     train_mask[i] = 1
        #
        # # one-hot mask
        # val_mask = np.zeros([self.n_sample], dtype=np.uint8)
        # for i in val_index:
        #     val_mask[i] = 1
        #
        # # one-hot mask
        # test_mask = np.zeros([self.n_sample], dtype=np.uint8)
        # for i in test_index:
        #     test_mask[i] = 1
        #
        # print("features", features.shape)  # (10000, 784)
        # print("labels", labels.shape)  # (10000, c)
        # print("train_mask", train_mask.shape)  # (10000,)
        # print("val_mask", val_mask.shape)  # (10000,)
        # print("test_mask", test_mask.shape)  # (10000, 784)

        return features, labels, train_index, val_index, test_index