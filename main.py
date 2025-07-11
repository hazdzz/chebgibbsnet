import argparse
import gc
import math
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from model.models import ChebGibbsNet
from utils import data_loader, early_stopping
from torch_geometric.utils import to_undirected
from tqdm import tqdm
from typing import Optional, Tuple, Union


def set_env(seed):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def get_parameters():
    parser = argparse.ArgumentParser(description='ChebGibbsNet')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable or disable CUDA (default: True)')
    parser.add_argument('--model_name', type=str, default='chebgibbsnet', help='model name (default: \'chebgibbsnet\')')
    parser.add_argument('--dataset_name', type=str, default='film', 
                        choices=['film', 'computers', 'photo', 
                        'cornell', 'texas', 'wisconsin', 'chameleon', 'squirrel', 
                        'roman-empire', 'amazon-ratings', 
                        'minesweeper', 'tolokers', 'questions', 
                        'cora_full', 'cora_ml', 'citeseer', 'dblp', 'pubmed', 
                        'wikics'], help='dataset')
    parser.add_argument('--order', type=int, default=10, help='polynomial order (default: 10)')
    parser.add_argument('--gibbs_type', type=str, default='jackson', choices=['none', 'dirichlet', \
                        'fejer', 'jackson', 'lanczos', 'lorentz', 'vekic', 'wang'], 
                        help='Gibbs damping factor type (default: jackson)')
    parser.add_argument('--mu', type=int, default=3, help='mu for Lanczos (default: 3)')
    parser.add_argument('--xi', type=float, default=4.0, help='xi for Lorentz (default: 4.0)')
    parser.add_argument('--stigma', type=float, default=0.5, help='stigma for Vekic (default: 0.5)')
    parser.add_argument('--heta', type=int, default=2, help='heta for Wang (default: 2)')
    parser.add_argument('--dropout_pre', type=float, default=0.1, help='dropout rate for Dropout before MLP (default: 0.1)')
    parser.add_argument('--dropout_in', type=float, default=0.1, help='dropout rate for Dropout inside MLP (default: 0.1)')
    parser.add_argument('--dropout_suf', type=float, default=0.1, help='dropout rate for Dropout after MLP (default: 0.1)')
    parser.add_argument('--num_hid', type=int, default=32, help='the feature size of the hidden layer (default: 32)')
    parser.add_argument('--bs', type=int, default=1, help='batch size (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=10000, help='epochs (default: 10000)')
    parser.add_argument('--opt', type=str, default='adam', choices=['adam', 'adamw', 'nadam', 'radam'], 
                        help='optimizer (default: adam)')
    parser.add_argument('--patience', type=int, default=50, help='early stopping patience (default: 50)')
    parser.add_argument('--num_splits', type=int, default=10, help='number of random splits (default: 10)')
    args = parser.parse_args()

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        gc.collect()

    return args, device


def prepare_data(args, device, split_idx):
    dataset, data = data_loader.data_loader(args.dataset_name)
    args.length = data.x.size(0)

    if (hasattr(dataset, 'is_undirected')) and (not dataset.is_undirected()):
        if data.edge_weight is None:
            data.edge_index = to_undirected(edge_index=data.edge_index, reduce='max')
        else:
            data.edge_index, data.edge_weight = to_undirected(
                                                                edge_index=data.edge_index,
                                                                edge_attr=data.edge_weight,
                                                                num_nodes=data.x.size(0),
                                                                reduce='max'
                                                            )

    data = data.to(device)

    if args.dataset_name == 'film':
        train_mask = data.train_mask[:, split_idx]
        val_mask = data.val_mask[:, split_idx]
        test_mask = data.test_mask[:, split_idx]
    elif args.dataset_name in ['computers', 'photo']:
        train_mask = data.train_mask[:, split_idx]
        val_mask = data.val_mask[:, split_idx]
        test_mask = data.test_mask[:, split_idx]
    elif args.dataset_name in ['cornell', 'texas', 'wisconsin', 
        'chameleon', 'squirrel']:
        train_mask = data.train_mask[:, split_idx]
        val_mask = data.val_mask[:, split_idx]
        test_mask = data.test_mask[:, split_idx]
    elif args.dataset_name in ['roman-empire', 'amazon-ratings', \
        'minesweeper', 'tolokers', 'questions']:
        train_mask = data.train_mask[:, split_idx]
        val_mask = data.val_mask[:, split_idx]
        test_mask = data.test_mask[:, split_idx]
    elif args.dataset_name in ['cora_full', 'cora_ml', 'citeseer', \
        'dblp', 'pubmed']:
        train_mask = data.train_mask[:, split_idx]
        val_mask = data.val_mask[:, split_idx]
        test_mask = data.test_mask[:, split_idx]
    elif args.dataset_name == 'wikics':
        train_mask = data.train_mask[:, split_idx]
        val_mask = data.val_mask[:, split_idx]
        test_mask = data.test_mask

    return data, train_mask, val_mask, test_mask


def prepare_model(args, device, split_idx):
    dataset, _ = data_loader.data_loader(args.dataset_name)
    model = ChebGibbsNet(dataset, args).to(device)
    loss = nn.CrossEntropyLoss()
    es = early_stopping.EarlyStopping(
                                        delta=0.0, 
                                        patience=args.patience, 
                                        verbose=False, 
                                        path=args.model_name + '_' \
                                            + args.dataset_name \
                                            + '_split_' + str(split_idx) \
                                            + "_order_" + str(args.order) \
                                            + ".pt"
                                    )

    if args.opt == 'adam':
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'nadam':
        optimizer = optim.NAdam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'radam':
        optimizer = optim.RAdam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f'ERROR: The {args.optimizer} optimizer is undefined.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    return loss, model, es, optimizer, scheduler

def run(args, model, optimizer, data, loss, train_mask, val_mask, test_mask, scheduler, es, split_idx):
    pbar = tqdm(range(1, args.epochs + 1), desc=f"Split {split_idx}/{args.num_splits-1}")
    for epoch in pbar:
        loss_train, acc_train = train(model, optimizer, data, loss, train_mask, scheduler)
        loss_val, acc_val = val(model, data, loss, val_mask)
        
        pbar.set_postfix({
            'Epoch': f'{epoch}/{args.epochs}',
            'Train Loss': f'{loss_train:.4f}',
            'Train Acc': f'{acc_train:.4f}',
            'Val Loss': f'{loss_val:.4f}',
            'Val Acc': f'{acc_val:.4f}'
        })
        
        es(loss_val, model)
        if es.early_stop:
            pbar.close()
            print(f"Early stopping at epoch {epoch} for Split {split_idx}.")
            break
        
        acc_train_list.append(acc_train)
    
    if not es.early_stop:
        pbar.close()
    
    acc_val_list.append(acc_val)
    loss_test, acc_test = test(args, model, data, loss, test_mask, split_idx)
    acc_test_list.append(acc_test)
    
    print(f'Split {split_idx} - Test Loss: {loss_test:.4f}, Test Acc: {acc_test * 100:.2f}%\n')


def train(model, optimizer, data, loss, mask, scheduler):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss_train= loss(out[mask], data.y[mask])
    acc_train = calc_accuracy(out, data, mask)
    loss_train.backward()
    optimizer.step()
    # scheduler.step()

    return loss_train, acc_train


@torch.no_grad()
def val(model, data, loss, mask):
    model.eval()
    out = model(data)
    loss_val = loss(out[mask], data.y[mask])
    acc_val = calc_accuracy(out, data, mask)
    
    return loss_val, acc_val


@torch.no_grad()
def test(args, model, data, loss, mask, split_idx):
    model.load_state_dict(
                            torch.load(
                                        args.model_name + '_' \
                                        + args.dataset_name \
                                        + '_split_' + str(split_idx) \
                                        + "_order_" + str(args.order) \
                                        + ".pt"
                                    )
                        )
    model.eval()
    out = model(data)
    loss_test = loss(out[mask], data.y[mask])
    acc_test = calc_accuracy(out, data, mask)
        
    return loss_test, acc_test


def calc_accuracy(out, data, mask):
    preds = out.argmax(dim=-1)
    acc = int((preds[mask] == data.y[mask]).sum()) / int(mask.sum())

    return acc


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    SEED = 3407
    args, device = get_parameters()

    acc_train_list = []
    acc_val_list = []
    acc_test_list = []

    for split_idx in range(args.num_splits):
        set_env(SEED)
        data, train_mask, val_mask, test_mask = prepare_data(args, device, split_idx)
        loss, model, es, optimizer, scheduler = prepare_model(args, device, split_idx)
        run(args, model, optimizer, data, loss, train_mask, val_mask, test_mask, scheduler, es, split_idx)

    acc_train_mean = np.mean(acc_train_list)
    acc_train_std = np.std(acc_train_list)
    acc_val_mean = np.mean(acc_val_list)
    acc_val_std = np.std(acc_val_list)
    acc_test_mean = np.mean(acc_test_list)
    acc_test_std = np.std(acc_test_list)

    print(f'test acc mean: {acc_test_mean * 100:.2f}%')
    print(f'test acc std: {acc_test_std * 100:.2f}%')