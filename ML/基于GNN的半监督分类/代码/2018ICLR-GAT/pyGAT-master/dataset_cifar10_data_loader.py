from __future__ import print_function
import numpy as np
import torch
import torchvision
from PIL import Image
import pickle as pkl
import os
import glob
import csv
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
        self.args = args
        self.train_subdir = 'train'
        self.eval_subdir = 'val'

    def create_data_loaders(self):
        channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616])
        train_transformation = transforms.Compose([
            RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ])
        eval_transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ])


        traindir = os.path.join(self.root_dir, self.train_subdir)
        evaldir = os.path.join(self.root_dir, self.eval_subdir)

        assert_exactly_one([self.args.exclude_unlabeled, self.args.labeled_batch_size])

        dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)

        if self.args.labels:
            with open(self.args.labels) as f:
                labels = dict(line.split(' ') for line in f.read().splitlines())
            labeled_idxs, unlabeled_idxs = relabel_dataset(dataset, labels)

        if self.args.exclude_unlabeled:
            sampler = SubsetRandomSampler(labeled_idxs)
            batch_sampler = BatchSampler(sampler, self.args.batch_size, drop_last=True)
        elif self.args.labeled_batch_size:
            batch_sampler = TwoStreamBatchSampler(
                unlabeled_idxs, labeled_idxs, self.args.batch_size, self.args.labeled_batch_size)
        else:
            assert False, "labeled batch size {}".format(self.args.labeled_batch_size)

        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_sampler=batch_sampler,
                                                   num_workers=0,
                                                   pin_memory=True)

        dataset_val = torchvision.datasets.ImageFolder(evaldir, eval_transformation)
        if self.args.labels_val:
            with open(self.args.labels_val) as f:
                labels_val = dict(line.split(' ') for line in f.read().splitlines())
            labeled_idxs_val, unlabeled_idxs_val = relabel_dataset(dataset_val, labels_val)

        if self.args.exclude_unlabeled:
            sampler = SubsetRandomSampler(labeled_idxs_val)
            batch_sampler_val = BatchSampler(sampler, self.args.batch_size, drop_last=False)
        elif self.args.labeled_batch_size:
            batch_sampler_val = TwoStreamBatchSampler(
                unlabeled_idxs_val, labeled_idxs_val, self.args.batch_size, self.args.labeled_batch_size)
        else:
            assert False, "labeled batch size {}".format(self.args.labeled_batch_size)

        eval_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_sampler=batch_sampler_val,
            num_workers=0,
            pin_memory=True)

        return train_loader, eval_loader


    def load_cifar10_batch(self,cifar10_dataset_folder_path, batch_name):
        with open(cifar10_dataset_folder_path + '/' + str(batch_name), mode='rb') as file:
            batch = pkl.load(file, encoding='latin1')

        # features and labels
        features = batch.data.reshape((len(batch.data), 3, 32, 32)).transpose(0, 2, 3, 1)
        labels = batch.labels

        return features, labels


    def load_data_pkl(self):
        # 加载所有训练数据
        if self.split == 'train':
            # 一共有5个batch的训练数据
            image_data, y = self.load_cifar10_batch(self.root_dir, 'data_batch_'+str(1))
            for i in range(2, 2):
                features, labels = self.load_cifar10_batch(self.root_dir, 'data_batch_'+str(i))
                image_data, y = np.concatenate([image_data, features]), np.concatenate([y, labels])

        # 加载测试数据
        elif self.split == 'val':
            image_data, y = self.load_cifar10_batch(self.root_dir, 'test_batch')

        else:
            print("No such data:"+self.split)
            return
        y = np.array(y)

        # 图像数据归一化、标准化
        image_data = img_normalize(image_data)

        np.random.seed(200)
        np.random.shuffle(image_data)
        np.random.seed(200)
        np.random.shuffle(y)

        self.n_examples = image_data.shape[0]
        self.n_unlabel = self.n_examples - self.n_label

        label_mask = np.zeros([self.n_examples], dtype=np.uint8)
        index_label = np.random.choice(self.n_examples, size=self.n_label, replace=False)
        for i in index_label:
            label_mask[i] = 1
        unlabel_mask = np.abs(1 - label_mask)

        label = np.zeros([self.n_examples, self.n_class], dtype=np.uint8)  # label -> (N,c)
        for ind, key in zip(range(self.n_examples),y):
            label[ind][key] = 1

        image_data = image_data.reshape((self.n_examples, 3, 32, 32))
        print(self.split,"image_data", image_data.shape)      # self.image_data (50000, 32, 32, 3)
        print(self.split,"label", label.shape)                # self.label (50000, c)
        print(self.split,"label_mask", label_mask.shape)      # self.label_mask (50000,)
        print(self.split,"unlabel_mask", unlabel_mask.shape)  # self.unlabel_mask (50000,)
        print(self.split,'n_classes:{}, n_label:{}, n_unlabel:{}'.format(self.n_class, self.n_label, self.n_unlabel))

        return image_data, label, label_mask, unlabel_mask



# loader_train = dataset_cifar10('val', 0.5)
# loader_train.load_data_pkl()
# loader_train = dataset_cifar10('val', 0.5)
# loader_train.load_data_pkl()