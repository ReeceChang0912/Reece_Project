import argparse
import os
import time
import copy
import torch
import torch.optim as optim
import pprint as pp
import utils.hypergraph_utils as hgut
from models import HGNN
from config import get_config
from datasets import load_feature_construct_H
from utils.dataset_cifar10_data_loader import dataset_cifar10
from utils.dataset_mnist import dataset_mnist
from utils.ops import *
from utils.CNN import CNN13


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cfg = get_config('config/config.yaml')

N_LABEL = 1000
CIFAR10_NUM_INPUT_FEATURES = 128
CIFAR10_NUM_CLASSES = 10
DATA_SET = 'cifar10'
parser = argparse.ArgumentParser()
# dataset params
parser.add_argument('--dataset', type=str, default=DATA_SET, metavar='DATASET', help="cifar10, mini or tiered")
parser.add_argument('--n_label', type=int, default=N_LABEL, metavar='n_label', help="number of labeled data")
parser.add_argument('--data_root', default='data/cifar10/cifar10_10000/images/', type=str, metavar='FILE',
                        help='dir of image data')
parser.add_argument('--labels', default='data/cifar10/cifar10_10000/labels/train/'+str(N_LABEL)+'_balanced_labels/00.txt', type=str, metavar='FILE',
                        help='list of image labels (default: based on directory structure)')
parser.add_argument('--labels_val', default='data/cifar10/cifar10_10000/labels/val/'+str(N_LABEL)+'_balanced_labels/00.txt', type=str, metavar='FILE',
                        help='list of image labels (default: based on directory structure)')

parser.add_argument('--batch_size', default=2000, type=int, help='mini-batch size (default: 10000)')
parser.add_argument('--labeled_batch_size', default=200, type=int, help="labeled examples per minibatch (default: no constrain)")
parser.add_argument('--exclude_unlabeled', default=False, type=bool, help='exclude unlabeled examples from the training set')

args = parser.parse_args()
# Load data
loader = dataset_cifar10(args)
train_loader, eval_loader = loader.create_data_loaders()
# ****************************  Feature Extract  ***************************************************
print("Feature Extract...")


Encoder = CNN13()
Encoder.cuda(0)
Encoder.load_state_dict(torch.load('CNNmodels/cnn13_50000/models/best_model.ckpt'))
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


print("train_features:", train_features.shape)  # (10000, 128)
print("train_label:", train_label.shape)  # (10000)
print("test_features:", test_features.shape)  # (10000, 128)
print("test_label:", test_label.shape)  # (10000)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_features = train_features.to(device)
test_features = test_features.to(device)
train_adj = torch.from_numpy(train_adj).float().to(device)
test_adj = torch.from_numpy(test_adj).float().to(device)
train_index = idx_train.to(device)
val_index = idx_test.to(device)



n_class = 10


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, print_freq=500, save_loss_pepoch=10, save_model_path='/'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_losses = []

    bad_counter = 0
    loss_best = 1000
    patience = 500

    for epoch in range(num_epochs):
        if epoch % print_freq == 0:
            print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs - 1}')

        # Train *************************************************************
        scheduler.step()
        model.train()
        running_loss = 0.0
        running_corrects = 0

        outputs = model(train_features, train_adj)
        loss = criterion(outputs[train_index], train_label[train_index])
        _, preds = torch.max(outputs, 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item() * test_features.size(0)
        running_corrects += torch.sum(preds[train_index] == train_label.data[train_index])

        epoch_loss = running_loss / len(train_index)
        epoch_acc = running_corrects.double() / len(train_index)

        if epoch % print_freq == 0:
            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        train_losses.append(epoch_loss)


        # Val *************************************************************
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            outputs = model(test_features, test_adj)
            loss = criterion(outputs[val_index], test_label[val_index])
            _, preds = torch.max(outputs, 1)

            # statistics
            running_loss += loss.item() * test_features.size(0)
            running_corrects += torch.sum(preds[val_index] == test_label.data[val_index])

            epoch_loss = running_loss / len(val_index)
            epoch_acc = running_corrects.double() / len(val_index)

            if epoch % print_freq == 0:
                print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if epoch_loss < loss_best:
                loss_best = epoch_loss
                bad_counter = 0
                best_model_wts = copy.deepcopy(model.state_dict())
            else:
                bad_counter += 1

            if bad_counter == patience:
                break

        if epoch % 100 == 0:
            # Test *************************************************************
            test(model, criterion)


    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    test(model, criterion)

    # Write losses to file
    save_loss(train_losses, num_epochs)
    # Show losses
    show_loss(train_losses, num_epochs)

    # load best model weights
    model.load_state_dict(best_model_wts)
    # Save model
    torch.save(model.state_dict(), save_model_path+"/model_"+str(num_epochs)+"_.pkl")
    return model


def _main():
    model_ft = HGNN(in_ch=train_features.shape[1],
                    n_class=n_class,
                    n_hid=cfg['n_hid'],
                    dropout=cfg['drop_out'])
    model_ft = model_ft.to(device)

    optimizer = optim.Adam(model_ft.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    # optimizer = optim.SGD(model_ft.parameters(), lr=0.01, weight_decay=cfg['weight_decay)
    schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg['milestones'],
                                               gamma=cfg['gamma'])
    criterion = torch.nn.CrossEntropyLoss()

    model_ft = train_model(model_ft, criterion, optimizer, schedular, 10000, print_freq=10,
                           save_loss_pepoch=cfg['save_loss_pepoch'], save_model_path=cfg['ckpt_folder'])



def test(model,criterion):
    running_loss = 0.0
    running_corrects = 0
    model.eval()
    outputs = model(test_features, test_adj)
    loss = criterion(outputs[val_index], test_label[val_index])
    _, preds = torch.max(outputs, 1)
    running_loss += loss.item() * test_features.size(0)
    running_corrects += torch.sum(preds[val_index] == test_label.data[val_index])
    epoch_loss = running_loss / len(val_index)
    epoch_acc = running_corrects.double() / len(val_index)
    print(f'Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')



if __name__ == '__main__':
    _main()
