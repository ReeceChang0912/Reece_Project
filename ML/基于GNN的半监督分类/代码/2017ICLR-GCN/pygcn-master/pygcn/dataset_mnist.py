import numpy as np
import torch
import os
import struct



class dataset_mnist(object):
    def __init__(self, n_label, data_root):
        self.n_label = n_label
        self.root_dir = data_root
        self.n_label = 1000
        self.n_class = 10
        self.n_sample = 10000
        self.n_val = 1000
        self.n_test = self.n_sample - self.n_val - self.n_label
        self.train_split = 'train'
        self.eval_split = 't10k'

    # 图像数据归一化、标准化
    def img_normalize_mnist(self, img):
        img = img.astype(np.float32) / 255.0  # 归一化为[0.0,1.0]
        means = np.mean(img)
        stdevs = np.std(img)
        img = (img - means) / stdevs
        # img = (img - 0.1307) / 0.3081
        return img

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
        datas, labels = self.read_data(self.train_split)

        datas = torch.tensor(datas)
        labels = torch.tensor(labels)

        size = 1000
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
            class_set = torch.index_select(datas, 0, class_set_ind)
            index = torch.LongTensor(np.random.choice(class_set.shape[0], size=size, replace=False))
            class_set = torch.index_select(class_set, 0, index)
            class_set_labels = torch.ones([size]) * i

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

        train_set = train_set.numpy()
        val_set = val_set.numpy()
        test_set = test_set.numpy()
        train_label = train_label.numpy().astype(int)
        val_label = val_label.numpy().astype(int)
        test_label = test_label.numpy().astype(int)

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

        datas = np.concatenate([train_set, val_set, test_set])
        labels = np.concatenate([train_label, val_label, test_label])

        # 图像数据归一化、标准化
        datas = self.img_normalize_mnist(datas)

        # one_labels = np.zeros([self.n_examples, self.n_class], dtype=np.uint8)  # label -> (N,c)
        # for ind, key in zip(range(self.n_examples), labels):
        #     one_labels[ind][key] = 1
        # labels = one_labels

        train_index = range(0, self.n_label)
        val_index = range(self.n_label, self.n_label + self.n_val)
        test_index = range(self.n_label + self.n_val, self.n_sample)

        # train_mask = np.zeros([self.n_examples], dtype=np.uint8)
        # for i in train_index:
        #     train_mask[i] = 1
        #
        # val_mask = np.zeros([self.n_examples], dtype=np.uint8)
        # for i in val_index:
        #     val_mask[i] = 1
        #
        # test_mask = np.zeros([self.n_examples], dtype=np.uint8)
        # for i in test_index:
        #     test_mask[i] = 1

        print("datas", datas.shape)  # (10000, 784)
        print("labels", labels.shape)  # (10000, c)
        # print("train_mask", train_mask.shape)  # (10000,)
        # print("val_mask", val_mask.shape)  # (10000,)
        # print("test_mask", test_mask.shape)  # (10000, 784)

        return datas, labels, train_index, val_index, test_index

