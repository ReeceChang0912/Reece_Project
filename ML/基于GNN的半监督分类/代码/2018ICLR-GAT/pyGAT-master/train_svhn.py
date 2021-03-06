
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
from dataset_svhn import dataset_svhn
from utils import load_data, accuracy, graph_construct, get_label_mask
from models import GAT, SpGAT


N_LABEL = 2000
DATA_SET = 'svhn'


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
parser.add_argument('--data_root', default='/mnt/Code/data/svhn/', type=str, metavar='FILE',
                        help='dir of image data')
parser.add_argument('--batch_size', default=5000, type=int, help='mini-batch size (default: 10000)')
parser.add_argument('--cnn_model', default='/mnt/Code/CNNmodels/cnn13_svhn/models/best_model.ckpt', type=str, metavar='FILE',
                        help='dir of image data')

args = parser.parse_args()
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
loader = dataset_svhn(args)
features, label, idx_train, idx_val, idx_test = loader.load_data()

features = torch.from_numpy(features).float()
label = torch.from_numpy(label).long()
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

'''
adj         2708*2708
features    2708*1433
labels      2708
idx_train   140
'''

print("Graph construct...")
adj = graph_construct(features.cpu())
adj = torch.from_numpy(adj)



# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=features.shape[1],
                nhid=args.hidden, 
                nclass=10,
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
else:
    model = GAT(nfeat=features.shape[1],
                nhid=args.hidden, 
                nclass=10,
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    label = label.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


features, adj, label = Variable(features), Variable(adj), Variable(label)



def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], label[idx_train])
    acc_train = accuracy(output[idx_train], label[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], label[idx_val])
    acc_val = accuracy(output[idx_val], label[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], label[idx_test])
    acc_test = accuracy(output[idx_test], label[idx_test])
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
files = glob.glob('*.pkl')
for file in files:
    os.remove(file)
