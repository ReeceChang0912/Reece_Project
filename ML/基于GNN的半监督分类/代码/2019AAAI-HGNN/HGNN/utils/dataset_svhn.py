from __future__ import print_function
import numpy as np
import torch
import torchvision
import os
import torchvision.datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.CNN import CNN13




class dataset_svhn(object):
    def __init__(self, args):
        self.n_label = args.n_label
        self.root_dir = args.data_root
        self.n_class = 10
        self.n_sample = 10000
        self.n_val = 1000
        self.args = args
        self.train_subdir = 'train'
        self.eval_subdir = 'val'

    def create_data_loaders(self):
        channel_stats = dict(mean=[0.4524, 0.4525, 0.4690],
                             std=[0.1225, 0.1283, 0.1144])
        transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ])

        dataset = torchvision.datasets.SVHN(root=self.root_dir, split='test',
                                            download=False, transform=transformation)

        data_loader = torch.utils.data.DataLoader(  dataset,
                                                    batch_size=500,
                                                    num_workers=0,
                                                    pin_memory=True)

        return data_loader


    def feature_extract(self, data_loader):
        Encoder = CNN13()
        Encoder.cuda(0)
        Encoder.load_state_dict(torch.load('CNNmodels/cnn13_svhn/models/best_model.ckpt'))

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
            class_set = torch.index_select(features, 0, class_set_ind)
            index = torch.LongTensor(np.random.choice(class_set.shape[0], size=size, replace=False))
            class_set = torch.index_select(class_set, 0, index.cuda(0))
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

        # # one-hot labels
        # one_labels = np.zeros([self.n_sample, self.n_class], dtype=np.uint8)  # label -> (N,c)
        # for ind, key in zip(range(self.n_sample), labels):
        #     one_labels[ind][key] = 1
        # labels = one_labels

        train_index = range(0, self.n_label)
        val_index = range(self.n_label, self.n_label + self.n_val)
        test_index = range(self.n_label + self.n_val, self.n_sample)

        # one-hot mask
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

        print("features", features.shape)  # (10000, 784)
        print("labels", labels.shape)  # (10000, c)
        # print("train_mask", train_mask.shape)  # (10000,)
        # print("val_mask", val_mask.shape)  # (10000,)
        # print("test_mask", test_mask.shape)  # (10000, 784)

        return features, labels, train_index, val_index, test_index