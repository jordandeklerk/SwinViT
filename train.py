import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import tqdm
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

from utils.dataloader import datainfo, dataload
from model.swin_vit import SwinTransformer
from utils.loss import LabelSmoothingCrossEntropy
from utils.scheduler import build_scheduler  
from utils.optimizer import get_adam_optimizer


def get_args_parser():
    parser = argparse.ArgumentParser('SWIN ViT for CIFAR-10', add_help=False)
    parser.add_argument('--dir', type=str, default='./data',
                    help='Data directory')
    parser.add_argument('--num_classes', type=int, default=10, choices=[10, 100, 1000],
                    help='Dataset name')

    # Model parameters
    parser.add_argument('--patch_size', default=2, type=int, help="""Size in pixels of input square patches - default 4 (for 4x4 patches) """)
    parser.add_argument('--out_dim', default=1024, type=int, help="""Dimensionality of the SSL MLP head output. For complex and large datasets large values (like 65k) work well.""")

    parser.add_argument('--norm_last_layer', default=False, type=bool,
        help="""Whether or not to weight normalize the last layer of the MLP head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--use_bn_in_head', default=False, type=bool,
        help="Whether to use batch normalizations in projection head (Default: False)")

    parser.add_argument('--image_size', default=32, type=int, help=""" Size of input image. """)
    parser.add_argument('--in_channels',default=3, type=int, help=""" input image channels. """)
    parser.add_argument('--embed_dim',default=192, type=int, help=""" dimensions of vit """)
    parser.add_argument('--num_layers',default=9, type=int, help=""" No. of layers of ViT """)
    parser.add_argument('--num_heads',default=12, type=int, help=""" No. of heads in attention layer
                                                                                 in ViT """)
    parser.add_argument('--vit_mlp_ratio',default=2, type=int, help=""" MLP hidden dim """)
    parser.add_argument('--qkv_bias',default=True, type=bool, help=""" Bias in Q K and V values """)
    parser.add_argument('--drop_rate',default=0., type=float, help=""" dropout """)

    # Training/Optimization parameters
    parser.add_argument('--weight_decay', type=float, default=1e-1, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--batch_size', default=128, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. Recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                    help='Label smoothing for optimizer')
    parser.add_argument('--gamma', type=float, default=1.0,
                    help='Gamma value for Cosine LR schedule')

    # Misc
    parser.add_argument('--dataset', default='CIFAR10', type=str, choices=['CIFAR10', 'CIFAR100'], help='Please specify path to the training data.')
    parser.add_argument('--seed', default=42, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--mlp_head_in", default=192, type=int, help="input dimension going inside MLP projection head")

    return parser


class History:
    def __init__(self):
        self.values = defaultdict(list)

    def append(self, key, value):
        self.values[key].append(value)

    def reset(self):
        for k in self.values.keys():
            self.values[k] = []

    def _begin_plot(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

    def _end_plot(self, ylabel):
        self.ax.set_xlabel('epoch')
        self.ax.set_ylabel(ylabel)
        plt.show()

    def _plot(self, key, line_type='-', label=None):
        if label is None: label=key
        xs = np.arange(1, len(self.values[key])+1)
        self.ax.plot(xs, self.values[key], line_type, label=label)

    def plot(self, key):
        self._begin_plot()
        self._plot(key, '-')
        self._end_plot(key)

    def plot_train_val(self, key):
        self._begin_plot()
        self._plot('train ' + key, '.-', 'train')
        self._plot('val ' + key, '.-', 'val')
        self.ax.legend()
        self._end_plot(key)


class Learner:
    def __init__(self, model, loss, optimizer, train_loader, val_loader, device,
                 epoch_scheduler=None, batch_scheduler=None, seed=42):
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epoch_scheduler = epoch_scheduler
        self.batch_scheduler = batch_scheduler
        self.history = History()
    
    
    def iterate(self, loader, msg="", backward_pass=False):
        total_loss = 0.0
        num_samples = 0
        num_correct = 0
        
        pbar = tqdm(enumerate(loader), total=len(loader))
        for it, (X, Y) in pbar:
            X, Y = X.to(self.device), Y.to(self.device)
            Y_pred = self.model(X)
            batch_size = X.size(0)
            batch_loss = self.loss(Y_pred, Y)
            if backward_pass:
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                if self.batch_scheduler is not None:
                    self.batch_scheduler.step()
            
            Y_pred.detach_() # conserve memory
            labels_pred = torch.argmax(Y_pred, -1)
            total_loss += batch_size * batch_loss.item()
            num_correct += (labels_pred == Y).sum().item()
            num_samples += batch_size
            
            pbar.set_description(msg)
            pbar.set_postfix(loss=total_loss / num_samples, acc=float(num_correct) / num_samples)
    
        avg_loss = total_loss / num_samples
        accuracy = float(num_correct) / num_samples
        return avg_loss, accuracy
    
    
    def train(self, msg):
        self.model.train()
        train_loss, train_acc = self.iterate(self.train_loader, msg + ' train:', backward_pass=True)
        self.history.append('train loss', train_loss)
        self.history.append('train acc', train_acc)
        return train_loss, train_acc

        
    def validate(self, msg):
        self.model.eval()
        with torch.no_grad():
            val_loss, val_acc = self.iterate(self.val_loader, msg + ' val:')
        self.history.append('val loss', val_loss)
        self.history.append('val acc', val_acc)
        return val_loss, val_acc


    def fit(self, epochs):
        pbar = tqdm(range(epochs))
        for e in pbar:
            msg = f'epoch {e+1}/{epochs}'
            train_loss, train_acc = self.train(msg)
            val_loss, val_acc = self.validate(msg)
            if self.epoch_scheduler is not None:
                self.epoch_scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            pbar.set_description(msg)
            pbar.set_postfix(train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc, lr=lr)


def main():
    args, unknown = get_args_parser().parse_known_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    data_info = datainfo(args)
    normalize = [transforms.Normalize(mean=data_info['stat'][0], std=data_info['stat'][1])]

    train_dataset, val_dataset = dataload(args, normalize, data_info)   

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                                                num_workers=args.num_workers, pin_memory=True)

    model = SwinTransformer(args.num_classes, args.image_size,
                        num_blocks_list=[4, 4], dims=[128, 128, 256],
                        head_dim=32, patch_size=args.patch_size, window_size=4,
                        emb_p_drop=0., trans_p_drop=0., head_p_drop=0.3).to(device)

    loss = LabelSmoothingCrossEntropy()
    optimizer = get_adam_optimizer(model.parameters(), lr=args.lr, wd=args.weight_decay)
    lr_scheduler = build_scheduler(args, optimizer)
    
    learner = Learner(model, loss, optimizer, train_loader, val_loader, device)
    learner.batch_scheduler = lr_scheduler

    learner.fit(args.epochs)

if __name__ == "__main__":
    main()
