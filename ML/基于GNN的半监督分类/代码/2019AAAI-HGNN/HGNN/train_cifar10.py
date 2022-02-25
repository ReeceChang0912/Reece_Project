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
from utils.dataset_cifar10 import dataset_cifar10
from utils.ops import *
from utils.CNN import CNN13


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cfg = get_config('config/config.yaml')

N_LABEL = 500
CIFAR10_NUM_INPUT_FEATURES = 128
CIFAR10_NUM_CLASSES = 10
DATA_SET = 'cifar10'
parser = argparse.ArgumentParser()
# dataset params
parser.add_argument('--dataset', type=str, default=DATA_SET, metavar='DATASET', help="cifar10, mini or tiered")
parser.add_argument('--n_label', type=int, default=N_LABEL, metavar='n_label', help="number of labeled data")
parser.add_argument('--data_root', default='data/cifar10/cifar10_10000/images/', type=str, metavar='FILE',help='dir of image data')

parser.add_argument('--batch_size', default=1000, type=int, help='mini-batch size (default: 10000)')

args = parser.parse_args()


loader = dataset_cifar10(args)
features, labels, train_index, val_index, test_index = loader.load_data()

print("Graph construct...")
H = graph_construct(features)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

H = torch.from_numpy(H).float().to(device)
features = torch.from_numpy(features).float().to(device)
labels = torch.from_numpy(labels).long().to(device)
train_index = torch.tensor(train_index).to(device)
val_index = torch.tensor(val_index).to(device)
test_index = torch.tensor(test_index).to(device)


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

        outputs = model(features, H)
        loss = criterion(outputs[train_index], labels[train_index])
        _, preds = torch.max(outputs, 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item() * features.size(0)
        running_corrects += torch.sum(preds[train_index] == labels.data[train_index])

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
            outputs = model(features, H)
            loss = criterion(outputs[val_index], labels[val_index])
            _, preds = torch.max(outputs, 1)

            # statistics
            running_loss += loss.item() * features.size(0)
            running_corrects += torch.sum(preds[val_index] == labels.data[val_index])

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
    model_ft = HGNN(in_ch=features.shape[1],
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

    save_result(model_ft)




def test(model,criterion):
    running_loss = 0.0
    running_corrects = 0
    model.eval()
    outputs = model(features, H)
    loss = criterion(outputs[test_index], labels[test_index])
    _, preds = torch.max(outputs, 1)
    running_loss += loss.item() * features.size(0)
    running_corrects += torch.sum(preds[test_index] == labels.data[test_index])
    epoch_loss = running_loss / len(test_index)
    epoch_acc = running_corrects.double() / len(test_index)
    print(f'Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


def save_result(model):
    model.eval()
    outputs = model(features, H)
    feature = outputs[test_index]
    label = labels[test_index]
    save_log(feature.detach().cpu().numpy(), 'feature')
    save_log(label.detach().cpu().numpy(), 'label')

def save_log(losses, name):
    losses = np.array(losses)
    np.save('./' +name + ".npy", losses)
    np.savetxt('./' +name + ".txt", losses)


if __name__ == '__main__':
    _main()
