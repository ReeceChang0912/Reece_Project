from __future__ import division
from __future__ import print_function

import glob
import os
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from pygcn.dataset_svhn import dataset_svhn
from pygcn.utils import accuracy, graph_construct
from pygcn.models import GCN


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=8150,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
n_label= 500
data_root = 'D:\\PythonCode\\PythonCode_MyMethod\\data\\svhn\\'
loader = dataset_svhn(n_label, data_root)
features, labels, idx_train, idx_val, idx_test = loader.load_data()

features = torch.from_numpy(features).float()
labels = torch.from_numpy(labels).long()
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

print("Graph construct...")
adj = graph_construct(features.cpu())
adj = torch.from_numpy(adj).float()


# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.item()


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
patience = 500
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

#     torch.save(model.state_dict(), '{}.pkl'.format(epoch))
#     if loss_values[-1] < best:
#         best = loss_values[-1]
#         best_epoch = epoch
#         bad_counter = 0
#     else:
#         bad_counter += 1
#
#     if bad_counter >= patience:
#         break
#
#     files = glob.glob('*.pkl')
#     for file in files:
#         epoch_nb = int(file.split('.')[0])
#         if epoch_nb < best_epoch:
#             os.remove(file)
#
# files = glob.glob('*.pkl')
# for file in files:
#     epoch_nb = int(file.split('.')[0])
#     if epoch_nb > best_epoch:
#         os.remove(file)

# print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
#
# # Restore best model
# print('Loading {}th epoch'.format(best_epoch))
# model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
# Testing
test()

# files = glob.glob('*.pkl')
# for file in files:
#     os.remove(file)