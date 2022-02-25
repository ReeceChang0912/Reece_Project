from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from CNN import CNN13
from dataset_cifar10_data_loader import dataset_cifar10
from utils import load_data, accuracy, graph_construct, get_label_mask
from models import GAT, SpGAT


N_LABEL = 1000
CIFAR10_NUM_INPUT_FEATURES = 128
CIFAR10_NUM_CLASSES = 10
DATA_SET = 'cifar10'


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=2, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

# dataset params
parser.add_argument('--dataset', type=str, default=DATA_SET, metavar='DATASET', help="cifar10, mini or tiered")
parser.add_argument('--n_label', type=int, default=N_LABEL, metavar='n_label', help="number of labeled data")
parser.add_argument('--data_root', default='/mnt/Code/data/cifar10_10000/images/', type=str, metavar='FILE',
                        help='dir of image data')
parser.add_argument('--labels', default='/mnt/Code/data/cifar10_10000/labels/train/'+str(N_LABEL)+'_balanced_labels/00.txt', type=str, metavar='FILE',
                        help='list of image labels (default: based on directory structure)')
parser.add_argument('--labels_val', default='/mnt/Code/data/cifar10_10000/labels/val/'+str(N_LABEL)+'_balanced_labels/00.txt', type=str, metavar='FILE',
                        help='list of image labels (default: based on directory structure)')

parser.add_argument('--batch_size', default=5000, type=int, help='mini-batch size (default: 10000)')
parser.add_argument('--labeled_batch_size', default=500, type=int, help="labeled examples per minibatch (default: no constrain)")
parser.add_argument('--exclude_unlabeled', default=False, type=bool, help='exclude unlabeled examples from the training set')

args = parser.parse_args()
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
loader = dataset_cifar10(args)
train_loader, eval_loader = loader.create_data_loaders()
# ****************************  Feature Extract  ***************************************************
print("Feature Extract...")
Encoder = CNN13()
Encoder.cuda(0)
Encoder.load_state_dict(torch.load('/mnt/Code/CNNmodels/cnn13_50000/models/best_model.ckpt'))
Encoder.eval()
with torch.no_grad():
    train_features = None
    train_label = None
    for i, (input, target) in enumerate(train_loader):
        _, image_feature = Encoder(input.cuda())
        if train_features is None:
            train_features = image_feature
            train_label = target.cuda()
        else:
            train_features = torch.cat((train_features, image_feature), dim=0)
            train_label = torch.cat((train_label, target.cuda()), dim=0)

    test_features = None
    test_label = None
    for i, (input, target) in enumerate(eval_loader):
        _, image_feature = Encoder(input.cuda(0))
        if test_features is None:
            test_features = image_feature
            test_label = target.cuda()
        else:
            test_features = torch.cat((test_features, image_feature), dim=0)
            test_label = torch.cat((test_label, target.cuda()), dim=0)

    torch.cuda.empty_cache()

idx_train = torch.nonzero(train_label!=-1).squeeze().long()
idx_test = torch.nonzero(test_label!=-1).squeeze().long()


print("Graph construct...")
train_adj = graph_construct(train_features.cpu())
test_adj = graph_construct(test_features.cpu())
train_adj = torch.from_numpy(train_adj)
test_adj = torch.from_numpy(test_adj)

print("train_features:", train_features.shape)  # (10000, 128)
print("train_label:", train_label.shape)  # (10000)
print("test_features:", test_features.shape)  # (10000, 128)
print("test_label:", test_label.shape)  # (10000)


# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=train_features.shape[1],
                nhid=args.hidden, 
                nclass=int(train_label.max()) + 1,
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
else:
    model = GAT(nfeat=train_features.shape[1],
                nhid=args.hidden, 
                nclass=int(train_label.max()) + 1,
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)


if args.cuda:
    model.cuda()
    train_features = train_features.cuda()
    train_adj = train_adj.cuda()
    train_label = train_label.long().cuda()

    test_features = test_features.cuda()
    test_adj = test_adj.cuda()
    test_label = test_label.long().cuda()

    idx_train = idx_train.cuda()
    idx_test = idx_test.cuda()


train_features, train_adj, train_label = Variable(train_features), Variable(train_adj), Variable(train_label)
test_features, test_adj, test_label = Variable(test_features), Variable(test_adj), Variable(test_label)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(train_features, train_adj)
    loss_train = F.nll_loss(output[idx_train], train_label[idx_train])
    acc_train = accuracy(output[idx_train], train_label[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation
        model.eval()
        output_test = model(test_features, test_adj)

    loss_val = F.nll_loss(output_test[idx_test], test_label[idx_test])
    acc_val = accuracy(output_test[idx_test], test_label[idx_test])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(test_features, test_adj)
    loss_test = F.nll_loss(output[idx_test], test_label[idx_test])
    acc_test = accuracy(output[idx_test], test_label[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
compute_test()


