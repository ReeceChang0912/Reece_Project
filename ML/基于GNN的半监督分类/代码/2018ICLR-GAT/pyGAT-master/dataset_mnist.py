import numpy as np
import torch
import os
import struct

from utils import img_normalize_mnist


class dataset_mnist(object):
    def __init__(self, args):
        self.root_dir = args.data_root
        self.n_class = 10
        self.n_sample = 10000
        self.n_val = 1000
        self.n_label = args.n_label
        self.args = args
        self.train_split = 'train'
        self.eval_split = 't10k'


    def read_data(self, kind):
        labels_path = os.path.join(self.root_dir, '%s-labels-idx1-ubyte' % kind)
        images_path = os.path.join(self.root_dir, '%s-images-idx3-ubyte' % kind)
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        return images, labels


    def load_data(self):
        train_datas, train_labels = self.read_data(self.train_split)
        # test_datas, test_labels = self.read_data(self.eval_split)

        train_datas = torch.tensor(train_datas)
        train_labels = torch.tensor(train_labels)
        # test_datas = torch.tensor(test_datas)
        # test_labels = torch.tensor(test_labels)

        size = 1000
        train_set = None
        train_labels_set = None
        for i in range(10):
            class_set_ind = torch.nonzero(train_labels==i).squeeze()
            class_set = torch.index_select(train_datas, 0, class_set_ind)
            index = torch.LongTensor(np.random.choice(class_set.shape[0], size=size, replace=False))
            class_set = torch.index_select(class_set, 0, index)
            class_set_labels = torch.ones([size])*i
            if train_set is None:
                train_set = class_set
                train_labels_set = class_set_labels
                continue

            train_set = torch.cat((train_set, class_set), dim=0)
            train_labels_set = torch.cat((train_labels_set, class_set_labels), dim=0)

        train_datas = train_set.numpy()
        train_labels = train_labels_set.numpy().astype(int)
        np.random.seed(200)
        np.random.shuffle(train_datas)
        np.random.seed(200)
        np.random.shuffle(train_labels)
        # np.random.seed(200)
        # np.random.shuffle(test_datas)
        # np.random.seed(200)
        # np.random.shuffle(test_labels)

        # 图像数据归一化、标准化
        train_datas = img_normalize_mnist(train_datas)
        # test_datas = img_normalize_mnist(test_datas)

        train_index = range(0, self.n_label)
        val_index = range(self.n_label, self.n_label + self.n_val)
        test_index = range(self.n_label + self.n_val, self.n_sample)


        print("train_datas", train_datas.shape)  # (10000, 784)
        print("train_label", train_labels.shape)  # (10000, c)
        # print("train_label_mask", train_label_mask.shape)  # (10000,)
        # print("train_unlabel_mask", train_unlabel_mask.shape)  # (10000,)
        # print("test_datas", test_datas.shape)  # (10000, 784)
        # print("test_label", test_labels.shape)  # (10000, c)
        # print("test_label_mask", test_label_mask.shape)  # (10000,)
        # print("test_unlabel_mask", test_unlabel_mask.shape)  # (10000,)
        # print('n_examples:{}, n_classes:{}, n_label:{}, n_unlabel:{}'.format(self.n_examples, self.n_class, self.n_label, self.n_examples-self.n_label))


        return train_datas, train_labels, train_index, val_index, test_index
