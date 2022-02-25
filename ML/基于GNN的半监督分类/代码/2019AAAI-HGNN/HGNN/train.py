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
from utils.ops import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cfg = get_config('config/config.yaml')

# initialize data
data_dir = cfg['modelnet40_ft'] if cfg['on_dataset'] == 'ModelNet40' \
    else cfg['ntu2012_ft']

# 加载数据，构建关联矩阵H
fts, lbls, idx_train, idx_test, H = \
    load_feature_construct_H(data_dir,
                             m_prob=cfg['m_prob'],
                             K_neigs=cfg['K_neigs'],
                             is_probH=cfg['is_probH'],
                             use_mvcnn_feature=cfg['use_mvcnn_feature'],
                             use_gvcnn_feature=cfg['use_gvcnn_feature'],
                             use_mvcnn_feature_for_structure=cfg['use_mvcnn_feature_for_structure'],
                             use_gvcnn_feature_for_structure=cfg['use_gvcnn_feature_for_structure'])

# 构建超图 G = Dv^(-1⁄2) * H * W * De^(-1) * H^T * Dv^(-1⁄2)
since = time.time()
G = hgut.generate_G_from_H(H)
time_elapsed = time.time() - since
print(f'\nGenerate Hypergraph complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
save_G(G)

n_class = int(lbls.max()) + 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# transform data to device
fts = torch.Tensor(fts).to(device)
lbls = torch.Tensor(lbls).squeeze().long().to(device)
G = torch.Tensor(G).to(device)
idx_train = torch.Tensor(idx_train).long().to(device)
idx_test = torch.Tensor(idx_test).long().to(device)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, print_freq=500, save_loss_pepoch=10, save_model_path='/'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_losses = []

    for epoch in range(num_epochs):
        if epoch % print_freq == 0:
            print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs - 1}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            idx = idx_train if phase == 'train' else idx_test

            # Iterate over data.
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(fts, G)
                loss = criterion(outputs[idx], lbls[idx])
                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * fts.size(0)
            running_corrects += torch.sum(preds[idx] == lbls.data[idx])

            epoch_loss = running_loss / len(idx)
            epoch_acc = running_corrects.double() / len(idx)

            if epoch % print_freq == 0:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save train losses
            if (phase == 'train') and (epoch % save_loss_pepoch == 0):
                train_losses.append(epoch_loss)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if epoch % print_freq == 0:
            print(f'Best val Acc: {best_acc:4f}')
            print('-' * 20)

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

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
    print(f"Classification on {cfg['on_dataset']} dataset!!! class number: {n_class}")
    print(f"use MVCNN feature: {cfg['use_mvcnn_feature']}")
    print(f"use GVCNN feature: {cfg['use_gvcnn_feature']}")
    print(f"use MVCNN feature for structure: {cfg['use_mvcnn_feature_for_structure']}")
    print(f"use GVCNN feature for structure: {cfg['use_gvcnn_feature_for_structure']}")
    print('Configuration -> Start')
    pp.pprint(cfg)
    print('Configuration -> End')

    model_ft = HGNN(in_ch=fts.shape[1],
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

    model_ft = train_model(model_ft, criterion, optimizer, schedular, cfg['max_epoch'], print_freq=cfg['print_freq'],
                           save_loss_pepoch=cfg['save_loss_pepoch'], save_model_path=cfg['ckpt_folder'])


if __name__ == '__main__':
    _main()
