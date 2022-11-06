import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.utils as utils

from tqdm import tqdm
from script import dataloader, early_stopping
from model import models

def set_env(seed):
    # Set available CUDA devices
    # This option is crucial for an multi-GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

def get_parameters():
    parser = argparse.ArgumentParser(description='ChebGibbsNet')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable or disable CUDA (default: True)')
    parser.add_argument('--model_name', type=str, default='chebgibbsnet')
    parser.add_argument('--dataset_name', type=str, default='cora')
    parser.add_argument('--order', type=int, default=10, help='polynomial order (default: 10)')
    parser.add_argument('--gibbs_type', type=str, default='zhang', choices=['none', 'jackson', 'lanczos', 'zhang'], help='Gibbs damping factor type (default: jackson)')
    parser.add_argument('--mu', type=float, default=3, help='mu for Lanczos (default: 3)')
    parser.add_argument('--droprate_pre', type=float, default=0, help='dropout rate for Dropout before MLP (default: 0)')
    parser.add_argument('--droprate_in', type=float, default=0, help='dropout rate for Dropout inside MLP (default: 0)')
    parser.add_argument('--droprate_suf', type=float, default=0, help='dropout rate for Dropout after MLP (default: 0)')
    parser.add_argument('--num_hid', type=int, default=64, help='the channel size of hidden layer feature (default: 64)')
    parser.add_argument('--bs', type=int, default=1, help='batch size (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay (default: 0.0005)')
    parser.add_argument('--epochs', type=int, default=1000, help='epochs (default: 1000)')
    parser.add_argument('--opt', type=str, default='adam', help='optimizer (default: adam)')
    parser.add_argument('--patience', type=int, default=50, help='early stopping patience (default: 50)')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return args, device

def prepare_model(args, device):
    dataset, data = dataloader.data_loader(args.dataset_name)
    if hasattr(dataset, 'is_undirected'):
        if data.edge_weight is None:
            data.edge_index = utils.to_undirected(edge_index=data.edge_index, reduce='max')
        else:
            data.edge_index, data.edge_weight = utils.to_undirected(edge_index=data.edge_index, 
                                                edge_attr=data.edge_weight, num_nodes=data.x.size()[0], 
                                                reduce='max')

    homophily = utils.homophily(edge_index=data.edge_index, y=data.y, method='node')
    data, model = data.to(device), models.ChebGibbsNet(dataset, args, homophily).to(device)

    if args.bs == 1:
        if args.dataset_name in ['chameleon', 'cornell', 'film', 'squirrel', 'texas', \
                                 'wisconsin', 'cora_ml', 'citeseer_dir', 'telegram']:
            train_mask = data.train_mask[:, 0]
            val_mask = data.val_mask[:, 0]
            test_mask = data.test_mask[:, 0]
        elif args.dataset_name == 'wikics':
            train_mask = data.train_mask[:, 0]
            val_mask = data.val_mask[:, 0]
            test_mask = data.test_mask
        else:
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask

    loss = nn.NLLLoss()
    es = early_stopping.EarlyStopping(mode='min', min_delta=0.0, patience=args.patience)

    if args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=False)
    else:
        raise ValueError(f'ERROR: The {args.opt} optimizer is undefined.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    return data, loss, train_mask, val_mask, test_mask, model, es, optimizer, scheduler

def run(args, model, optimizer, data, loss, train_mask, val_mask, test_mask, scheduler, es):
    for epoch in tqdm(range(1, args.epochs+1)):
        loss_train, acc_train = train(model, optimizer, data, loss, train_mask, scheduler)
        loss_val, acc_val = val(model, data, loss, val_mask)

        if es.step(loss_val) and epoch >= (args.patience + 1):
            # print('Early stopping.')
            break
    
    acc_train_list.append(acc_train)
    acc_val_list.append(acc_val)
    loss_test, acc_test = test(model, data, loss, test_mask)
    acc_test_list.append(acc_test)

    print(f'test acc: {acc_test * 100:.2f}%')

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
def test(model, data, loss, mask):
    model.eval()
    out = model(data)
    loss_test = loss(out[mask], data.y[mask])
    acc_test = calc_accuracy(out, data, mask)
        
    return loss_test, acc_test

def calc_accuracy(out, data, mask):
    accs = []
    # preds = out.max(1)[1].type_as(data.y[mask])
    # acc = preds[mask].eq(data.y[mask]).sum().item() / mask.sum().item()
    preds = out.argmax(dim=-1)
    acc = int((preds[mask] == data.y[mask]).sum()) / int(mask.sum())

    return acc

if __name__ == '__main__':
    seeds = [1, 42, 3407, 10076, 934890, 74512355, 124, 2132134, 43059, 2354367]

    acc_train_list = []
    acc_val_list = []
    acc_test_list = []

    args, device = get_parameters()
    for i in range(len(seeds)):
        set_env(seeds[i])
        data, loss, train_mask, val_mask, test_mask, model, es, optimizer, scheduler = prepare_model(args, device)
        run(args, model, optimizer, data, loss, train_mask, val_mask, test_mask, scheduler, es)

    # set_env(42)
    # args, device = get_parameters()
    # data, loss, train_mask, val_mask, test_mask, model, es, optimizer, scheduler = prepare_model(args, device)
    # run(args, model, optimizer, data, loss, train_mask, val_mask, test_mask, scheduler, es)

    acc_train_mean = np.mean(acc_train_list)
    acc_train_std = np.std(acc_train_list)
    acc_val_mean = np.mean(acc_val_list)
    acc_val_std = np.std(acc_val_list)
    acc_test_mean = np.mean(acc_test_list)
    acc_test_std = np.std(acc_test_list)

    print(f'test acc mean: {acc_test_mean * 100:.2f}%')
    print(f'test acc std: {acc_test_std * 100:.2f}%')