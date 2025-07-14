import os.path as osp
import numpy as np
import pickle
import torch
import torch_geometric.transforms as T

from src.utils.data_creator import Amazon, CitationFull
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Actor, HeterophilousGraphDataset, WebKB, WikiCS, WikipediaNetwork
from typing import Optional, Callable


def data_loader(name):
    name = name.lower()
    assert name in ['computers', 'photo', 'film', 'cornell', 'texas', 'wisconsin', 'chameleon', 'squirrel', \
                    'roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', \
                    'questions', 'cora_full', 'cora_ml', 'citeseer', 'dblp', 'pubmed', 'wikics']

    ### Undirected Graphs
    if name in ['film']:
        dataset = Actor(root='./data/film/', transform=T.NormalizeFeatures())
        data = dataset[0]
    elif name in ['computers', 'photo']:
        dataset = Amazon(root='./data/', name=name, transform=T.NormalizeFeatures())
        data = dataset[0]
    elif name in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(root='./data/', name=name, transform=T.NormalizeFeatures())
        data = dataset[0]
    elif name in ['chameleon', 'squirrel']:
    # geom_gcn_preprocess (bool): If set to :obj:`True`, will load the
    #                             pre-processed data as introduced in the `"Geom-GCN: Geometric
    #                             Graph Convolutional Networks" <https://arxiv.org/abs/2002.05287>_`,
    #                             in which the average monthly traffic of the web page is converted
    #                             into five categories to predict.
        preProcDs = WikipediaNetwork(root='./data/', name=name, geom_gcn_preprocess=False, transform=T.NormalizeFeatures())
        # The dataset 'crocodile' is not available in case 'geom_gcn_preprocess=True'
        dataset = WikipediaNetwork(root='./data/', name=name, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.edge_index = preProcDs[0].edge_index
    elif name in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
        dataset = HeterophilousGraphDataset(root='./data/', name=name, transform=T.NormalizeFeatures())
        data = dataset[0]
    ### Directed Graphs
    elif name in ['cora_full', 'cora_ml', 'citeseer', 'dblp', 'pubmed']:
        if name == 'cora_full':
            name = 'cora'
        dataset = CitationFull(root='./data/', name=name, transform=T.NormalizeFeatures(), to_undirected=True)
        data = dataset[0]
    elif name == 'wikics':
        dataset = WikiCS(root='./data/wikics/', transform=T.NormalizeFeatures(), is_undirected=True)
        data = dataset[0]
        std, mean = torch.std_mean(data.x, dim=0, unbiased=False)
        data.x = (data.x - mean) / std
    else:
        raise ValueError(f'Dataset {name} is not included.')

    return dataset, data