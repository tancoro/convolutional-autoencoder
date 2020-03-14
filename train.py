# import comet_ml in the top of your file
from comet_ml import Experiment

from collections import OrderedDict
import numpy as np

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

from model import Classifier

import matplotlib.pyplot as plt

# (for Mac) OMP: Error #15:
# Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def imshow(epoch, imgs, dec):
    fig, ax = plt.subplots(
        nrows=2, ncols=12, sharex=True, sharey=True, figsize=(18, 6))
    for k in range(0, 12):
        newimg = imgs[k].cpu().numpy().transpose((1, 2, 0))
        decimg = dec[k].cpu().numpy().transpose((1, 2, 0))
        ax[0, k].imshow(newimg, interpolation='none')
        ax[1, k].imshow(decimg, interpolation='none')

    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
    plt.savefig(f'./img/epoch{epoch}.png')


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
            if label not in label_hash:
                label_hash[label] = []
            label_hash[label].append(idx)

        self.data_count_map = {}
        self.indices = []
        for k in label_hash.keys():
            if k in sampling_rate:
                label_size = len(label_hash[k])
                sampling_count = int(label_size * sampling_rate[k])
                rand_idx = torch.randint(
                    high=label_size - 1, size=(sampling_count,),
                    dtype=torch.int64).numpy()
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


class AverageMeter(object):
    def __init__(self, formater):
        self.sum = 0
        self.count = 0
        self.formater = formater

    def update(self, val, n=1):
        self.sum += val
        self.count += n

    def avg(self):
        return self.formater.format(self.sum / self.count)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(args, model, device, train_loader,
          data_count, optimizer, epoch,
          experiment, criterion=None):
    model.train()
    iter_num = len(train_loader)
    lr = get_lr(optimizer)

    loss_m = AverageMeter('{:.6f}')
    ce_loss_m = AverageMeter('{:.6f}')
    be_loss_m = AverageMeter('{:.6f}')

    with tqdm(train_loader) as _tqdm:
        for batch_idx, (data, labels) in enumerate(_tqdm):
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
            experiment.log_metric("lr", lr, step=g_step)

            sample_num = x.size(0)
            loss_m.update(loss.item() * sample_num, sample_num)
            ce_loss_m.update(ce_loss.item() * sample_num, sample_num)
            be_loss_m.update(bce_loss.item() * sample_num, sample_num)

            _tqdm.set_description('[train] Epoch {:02d}'.format(epoch))
            _tqdm.set_postfix(OrderedDict(loss=loss_m.avg(),
                ce_loss=ce_loss_m.avg(), bce_loss=be_loss_m.avg(), lr=lr))


def test(args, model, device, test_loader,
         data_count, epoch, experiment,
         lr, pref='', criterion=None):
    model.eval()
    ref_data = None
    out_data = None
    correct = 0
    iter_num = len(test_loader)

    loss_m = AverageMeter('{:.6f}')
    ce_loss_m = AverageMeter('{:.6f}')
    be_loss_m = AverageMeter('{:.6f}')
    correct_m = AverageMeter('{:.3f}')

    with torch.no_grad():
        with tqdm(test_loader) as _tqdm:
            for batch_idx, (data, labels) in enumerate(_tqdm):
                data, labels = data.to(device), labels.to(device)
                x, decoded = model(data)

                # cal Loss
                loss, ce_loss, bce_loss = criterion(x, decoded, data, labels)

                g_step = iter_num * (epoch - 1) + batch_idx
                experiment.log_metric("loss", loss.item(), step=g_step)
                experiment.log_metric("ce_loss", ce_loss.item(), step=g_step)
                experiment.log_metric("bce_loss", bce_loss.item(), step=g_step)

                ref_data = data
                out_data = decoded

                # get the index of the max log-probability
                pred = x.argmax(dim=1, keepdim=True)
                correct = pred.eq(labels.view_as(pred)).sum().item()

                sample_num = x.size(0)
                experiment.log_metric("accuracy",
                    correct / sample_num, step=g_step)
                experiment.log_metric("lr", lr, step=g_step)

                loss_m.update(loss.item() * sample_num, sample_num)
                ce_loss_m.update(ce_loss.item() * sample_num, sample_num)
                be_loss_m.update(bce_loss.item() * sample_num, sample_num)
                correct_m.update(correct, sample_num)

                _tqdm.set_description('[valid] Epoch {:02d}'.format(epoch))
                _tqdm.set_postfix(OrderedDict(loss=loss_m.avg(),
                    ce_loss=ce_loss_m.avg(), bce_loss=be_loss_m.avg(), lr=lr, acc=correct_m.avg()))

    # save decoded images
    imshow(epoch, ref_data, out_data)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Cifar10 Convolutional Autoencoder Example')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: 25)')
    parser.add_argument('--ce_weights', type=float, default=0.5, metavar='C', help='cross entropy loss weights (default: 0.5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--model_path', type=str, default='', metavar='M', help='model param path')
    parser.add_argument('--imbalance', type=str, default='N', metavar='I', choices=['N', 'US'], help='N or US')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=2, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()

    # comet.ml
    experiment = Experiment(api_key="5Yl3Rxz9S3E0PUKQTBpA0QJPi", project_name="convolutional-autoencoder", workspace="tancoro")

    # experiment.display(clear=True, wait=True, new=0, autoraise=True)
    exp_key = experiment.get_key()
    print('KEY: ' + exp_key)
    print('ce_weights: {}'.format(args.ce_weights))
    print('imbalance: {}'.format(args.imbalance))

    # HyperParam
    hyper_params = {
        'epoch': args.epochs,
        'ce_weights': args.ce_weights,
        'learning_rate': args.lr,
        'model_path': args.model_path,
        'imbalance': args.imbalance,
        'seed': args.seed
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
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))

    # normal sampling
    if args.imbalance == 'N':
        train_sampler = ReductionSampler(train_dataset, sampling_rate={2: 0.5, 4: 0.5, 9: 0.5})
    # under sampling
    elif args.imbalance == 'US':
        train_sampler = ReductionSampler(train_dataset, sampling_rate={0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5, 6: 0.5, 7: 0.5, 8: 0.5, 9: 0.5})

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, sampler=train_sampler, **kwargs)

    # test dataset
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor()
        ]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True, **kwargs)

    model = Classifier().to(device)
    criterion = CompositeLoss(args.ce_weights)

    # load param
    if len(args.model_path) > 0:
        model.load_state_dict(torch.load(args.model_path))

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # lr = 0.01 if epoch <= 15
    # lr = 0.001 if 15 < epoch <= 20
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)

    for epoch in range(1, args.epochs + 1):
        with experiment.train():
            experiment.log_current_epoch(epoch)
            train(args, model, device, train_loader, len(train_sampler), optimizer, epoch, experiment, criterion=criterion)
        with experiment.test():
            test(args, model, device, test_loader, len(test_loader.dataset), epoch, experiment, get_lr(optimizer), pref='all', criterion=criterion)
        if (args.save_model) and (epoch % 2 == 0):
            print('saving model to ./model/conv_autoencoder_{0}_{1:04d}.pt'.format(exp_key, epoch))
            torch.save(model.state_dict(), "./model/conv_autoencoder_{0}_{1:04d}.pt".format(exp_key, epoch))
        scheduler.step()


if __name__ == '__main__':
    main()
