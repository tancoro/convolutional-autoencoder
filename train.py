# import comet_ml in the top of your file
from comet_ml import Experiment

import numpy as np

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data.sampler import Sampler

from model import Classifier

# Matplotlib
import matplotlib.pyplot as plt

## (for Mac) OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def imshow(epoch, imgs, dec):
    fig, ax = plt.subplots(nrows=2, ncols=12, sharex=True, sharey=True, figsize=(18,6))
    # ax = ax.flatten()
    for k in range(0, 12):
        newimg = imgs[k].cpu().numpy().transpose((1, 2, 0))
        decimg = dec[k].cpu().numpy().transpose((1, 2, 0))
        ax[0, k].imshow(newimg, interpolation='none')
        ax[1, k].imshow(decimg, interpolation='none')
    # ax[0].set_xticks([])
    # ax[0].set_yticks([])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
    # plt.tight_layout()
    plt.savefig(f'./img/epoch{epoch}.png')
    # plt.show()


class CompositeLoss():
    def __init__(self, ce_weights):
        self.ce_weights = ce_weights
        self.bce_instance = nn.BCELoss()

    def __call__(self, x, decoded, data, labels):
        # Binary Cross Entropy Loss
        bce_loss = self.bce_instance(decoded, data)
        # Cross Entropy Loss
        x = x.log()
        ce_loss = F.nll_loss(x, labels, reduction='mean')

        return ce_loss * self.ce_weights + bce_loss * (1.0 - self.ce_weights), ce_loss, bce_loss


class ReductionSampler(Sampler):
    def __init__(self, data_source, sampling_rate={}):
        self.data_source = data_source

        label_hash = {}
        for idx, (img, label) in enumerate(self.data_source):
            if not label in label_hash:
                label_hash[label] = []
            label_hash[label].append(idx)

        self.data_count_map = {}
        self.indices = []
        for k in label_hash.keys():
            if k in sampling_rate:
                label_size = len(label_hash[k])
                sampling_count = int(label_size * sampling_rate[k])
                rand_idx = torch.randint(high=label_size - 1, size=(sampling_count,), dtype=torch.int64).numpy()
                self.indices.extend(np.array(label_hash[k])[rand_idx].tolist())
                self.data_count_map[k] = len(rand_idx)
            else:
                self.indices.extend(label_hash[k])
                self.data_count_map[k] = len(label_hash[k])

    def get_data_count_map(self):
        return self.data_count_map

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(args, model, device, train_loader, data_count, optimizer, epoch, experiment, criterion=None):
    model.train()
    iter_num = len(train_loader)
    print('lr: {0} epoch:[{1}]'.format(get_lr(optimizer), epoch))
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        x, decoded = model(data)

        loss, ce_loss, bce_loss = criterion(x, decoded, data, labels)
        loss.backward()
        optimizer.step()

        g_step = iter_num * (epoch - 1) + batch_idx
        experiment.log_metric("loss", loss.item(), step=g_step)
        experiment.log_metric("ce_loss", ce_loss.item(), step=g_step)
        experiment.log_metric("bce_loss", bce_loss.item(), step=g_step)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} CE_Loss: {:.6f} BCE_Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), data_count,
                100. * batch_idx / iter_num, loss.item(), ce_loss.item(), bce_loss.item()))

    experiment.log_metric("lr", get_lr(optimizer), step=epoch)


def test(args, model, device, test_loader, data_count, epoch, experiment, lr, pref='', criterion=None):
    model.eval()
    ref_data = None
    out_data = None
    correct = 0
    iter_num = len(test_loader)
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            data, labels = data.to(device), labels.to(device)
            x, decoded = model(data)

            # cal Loss
            loss, ce_loss, bce_loss = criterion(x, decoded, data, labels)

            print('test_loss    : {}'.format(loss.item()))
            print('test_ce_loss : {}'.format(ce_loss.item()))
            print('test_bce_loss: {}'.format(bce_loss.item()))
            g_step = iter_num * (epoch - 1) + batch_idx
            experiment.log_metric("loss", loss.item(), step=g_step)
            experiment.log_metric("ce_loss", ce_loss.item(), step=g_step)
            experiment.log_metric("bce_loss", bce_loss.item(), step=g_step)
        
            ref_data = data
            out_data = decoded

            pred = x.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct = pred.eq(labels.view_as(pred)).sum().item()
            print('accuracy: {}'.format(correct / len(labels)))
            experiment.log_metric("accuracy", correct / len(labels), step=g_step)
            experiment.log_metric("lr", lr, step=g_step)

    # experiment.log_metric("loss", test_loss, step=(epoch-1))
    # experiment.log_metric("accuracy", correct / data_count, step=(epoch-1))
    # print('accuracy: {}'.format(correct / data_count))

    imshow(epoch, ref_data, out_data)
    # test_loss /= data_count
    # experiment.log_metric(pref + "_loss", test_loss, step=(epoch-1))
    # experiment.log_metric(pref + "_accuracy", correct / data_count, step=(epoch-1))
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #    test_loss, correct, data_count, 100. * correct / data_count))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Cifar10 Convolutional Autoencoder Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 25)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--model-path', type=str, default='', metavar='M', help='model param path')
    parser.add_argument('--loss-type', type=str, default='CE', metavar='L', help='B or CE or F or ICF_CE or ICF_F or CB_CE or CB_F')
    parser.add_argument('--beta', type=float, default=0.999, metavar='B', help='Beta for ClassBalancedLoss')
    parser.add_argument('--gamma', type=float, default=2.0, metavar='G', help='Gamma for FocalLoss')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=2, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--balanced-data', action='store_true', default=False, help='For sampling rate. Default is Imbalanced-data.')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()

    # Add the following code anywhere in your machine learning file
    experiment = Experiment(api_key="5Yl3Rxz9S3E0PUKQTBpA0QJPi", project_name="convolutional-autoencoder", workspace="tancoro")

    # ブラウザの実験ページを開く
    # experiment.display(clear=True, wait=True, new=0, autoraise=True)
    # 実験キー(実験を一意に特定するためのキー)の取得
    exp_key = experiment.get_key()
    print('KEY: ' + exp_key)
    ce_weights = 1.0
    print('ce_weights: {}'.format(ce_weights))
    # HyperParamの記録
    hyper_params = {
        'batch_size': args.batch_size,
        'epoch': args.epochs,
        'learning_rate': args.lr,
        'sgd_momentum' : args.momentum,
        'model_path' : args.model_path,
        'loss_type' : args.loss_type,
        'beta' : args.beta,
        'gamma' : args.gamma,
        'torch_manual_seed': args.seed,
        'balanced_data' : args.balanced_data,
        'ce_weights': ce_weights
    }
    experiment.log_parameters(hyper_params)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('use_cuda {}'.format(use_cuda))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # train dataset
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    # train loader
    train_sampler = ReductionSampler(train_dataset) #, sampling_rate={1:0.5, 4:0.5, 6:0.5})
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    # test dataset
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transforms.Compose([transforms.ToTensor()]))
    # test majority loader
    # test_majority_sampler = ReductionSampler(test_dataset, sampling_rate={1:0, 4:0, 6:0})
    # test_majority_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, sampler=test_majority_sampler, **kwargs)
    # test minority loader
    # test_minority_sampler = ReductionSampler(test_dataset, sampling_rate={0:0, 2:0, 3:0, 5:0, 7:0, 8:0, 9:0})
    # test_minority_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, sampler=test_minority_sampler, **kwargs)
    # test alldata loader
    test_alldata_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Classifier().to(device)
    # train loss
    criterion = CompositeLoss(ce_weights)

    # load param
    if len(args.model_path) > 0:
        model.load_state_dict(torch.load(args.model_path))

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=args.momentum, weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # lr = 0.1 if epoch < 15
    # lr = 0.01 if 15 <= epoch < 20
    # lr = 0.001 if 20 <= epoch < 25
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15], gamma=0.1)

    for epoch in range(1, args.epochs + 1):
        with experiment.train():
            experiment.log_current_epoch(epoch)
            train(args, model, device, train_loader, len(train_sampler), optimizer, epoch, experiment, criterion=criterion)
        with experiment.test():
        #    test(args, model, device, test_minority_loader, len(test_minority_sampler), epoch, experiment, pref='minority')
        #    test(args, model, device, test_majority_loader, len(test_majority_sampler), epoch, experiment, pref='majority')
            test(args, model, device, test_alldata_loader, len(test_alldata_loader.dataset), epoch, experiment, get_lr(optimizer), pref='all', criterion=criterion)
        if (args.save_model) and (epoch % 2 == 0):
            print('saving model to ./model/conv_autoencoder_{0}_{1:04d}.pt'.format(exp_key, epoch))
            torch.save(model.state_dict(), "./model/conv_autoencoder_{0}_{1:04d}.pt".format(exp_key, epoch))
        scheduler.step()

if __name__ == '__main__':
    main()
