import os.path as osp
import numpy as np
import torch

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_npz
from typing import Callable, List, Optional


class Amazon(InMemoryDataset):
    r"""The Amazon Computers and Amazon Photo networks from the
    `"Pitfalls of Graph Neural Network Evaluation"
    <https://arxiv.org/abs/1811.05868>`_ paper, modified to include
    10-fold cross validation splits similar to WebKB datasets.
    
    Nodes represent goods and edges represent that two goods are frequently
    bought together. Given product reviews as bag-of-words node features, 
    the task is to map goods to their respective product category.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"Computers"`,
            :obj:`"Photo"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
        split_seed (int, optional): Random seed for generating reproducible
            splits. (default: :obj:`42`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #nodes
          - #edges
          - #features
          - #classes
        * - Computers
          - 13,752
          - 491,722
          - 767
          - 10
        * - Photo
          - 7,650
          - 238,162
          - 745
          - 8

    **Note:** This implementation creates 10 random splits for cross-validation.
    The train_mask, val_mask, and test_mask will have shape [num_nodes, 10]
    where each column represents a different split.
    """

    url = 'https://github.com/shchur/gnn-benchmark/raw/master/data/npz/'

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
        split_seed: int = 42,
    ) -> None:
        self.name = name.lower()
        self.split_seed = split_seed
        assert self.name in ['computers', 'photo']
        
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        
        self.load(self.processed_paths[0])
        
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name.lower(), 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name.lower(), 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        # Main data file plus 10 split files
        out = [f'amazon_electronics_{self.name.lower()}.npz']
        out += [f'{self.name}_split_0.6_0.2_{i}.npz' for i in range(10)]
        return out

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        # Check if file already exists
        raw_path = osp.join(self.raw_dir, self.raw_file_names[0])
        if osp.exists(raw_path):
            print(f"Using existing file {self.raw_file_names[0]}")
            return
            
        # Download the main data file
        download_url(self.url + self.raw_file_names[0], self.raw_dir)
        print(f"Downloaded main data file: {self.raw_file_names[0]}")
        print("Split files will be generated during processing.")

    def _generate_splits(self, num_nodes: int, num_splits: int = 10, seed: int = 42) -> tuple:
        """Generate reproducible random train/val/test splits for cross-validation.
        
        Args:
            num_nodes (int): Number of nodes in the graph
            num_splits (int): Number of splits to generate (default: 10)
            seed (int): Random seed for reproducibility (default: 42)
            
        Returns:
            tuple: (train_masks, val_masks, test_masks) each of shape [num_nodes, num_splits]
        """
        with torch.random.fork_rng():
            torch.manual_seed(seed)

            train_masks = torch.zeros(num_nodes, num_splits, dtype=torch.bool)
            val_masks = torch.zeros(num_nodes, num_splits, dtype=torch.bool)
            test_masks = torch.zeros(num_nodes, num_splits, dtype=torch.bool)

            # Standard split ratios: 60% train, 20% val, 20% test
            train_ratio = 0.6
            val_ratio = 0.2

            for split_idx in range(num_splits):
                # Create random permutation for this split with different seed per split
                # This ensures different splits while maintaining reproducibility
                torch.manual_seed(seed + split_idx)
                perm = torch.randperm(num_nodes)

                train_end = int(train_ratio * num_nodes)
                val_end = int((train_ratio + val_ratio) * num_nodes)

                train_idx = perm[:train_end]
                val_idx = perm[train_end:val_end]
                test_idx = perm[val_end:]

                train_masks[train_idx, split_idx] = True
                val_masks[val_idx, split_idx] = True
                test_masks[test_idx, split_idx] = True
            
        return train_masks, val_masks, test_masks

    def process(self) -> None:
        # Check if processed file already exists with splits
        if osp.exists(self.processed_paths[0]) and not self.force_reload:
            # File exists, no need to process again
            return

        print(f"Generating new splits with seed {self.split_seed}...")
        
        data = read_npz(self.raw_paths[0], to_undirected=True)

        num_nodes = data.x.size(0)
        train_masks, val_masks, test_masks = self._generate_splits(
            num_nodes, seed=self.split_seed
        )

        device = data.x.device
        data.train_mask = train_masks.to(device)
        data.val_mask   = val_masks.to(device)
        data.test_mask  = test_masks.to(device)

        data.split_seed = self.split_seed
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])
        print(f"Splits saved to {self.processed_paths[0]}")

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}{self.name.capitalize()}()'

    def get_split(self, split_idx: int = 0):
        """Get masks for a specific split.
        
        Args:
            split_idx (int): Index of the split (0-9)
            
        Returns:
            tuple: (train_mask, val_mask, test_mask) for the specified split
        """
        if split_idx < 0 or split_idx >= 10:
            raise ValueError(f"Split index must be between 0 and 9, got {split_idx}")
            
        data = self[0]
        return (
            data.train_mask[:, split_idx],
            data.val_mask[:, split_idx], 
            data.test_mask[:, split_idx]
        )

    def get_split_info(self) -> dict:
        """Get information about all splits.
        
        Returns:
            dict: Statistics about the splits
        """
        data = self[0]
        info = {
            'num_splits': 10,
            'num_nodes': data.x.size(0),
            'num_features': data.x.size(1),
            'num_classes': data.y.unique().numel(),
            'split_stats': []
        }
        
        for i in range(10):
            train_mask, val_mask, test_mask = self.get_split(i)
            split_stat = {
                'split_idx': i,
                'train_nodes': int(train_mask.sum()),
                'val_nodes': int(val_mask.sum()),
                'test_nodes': int(test_mask.sum())
            }
            info['split_stats'].append(split_stat)
            
        return info


class CitationFull(InMemoryDataset):
    r"""The full citation network datasets from the
    `"Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via Ranking"
    <https://arxiv.org/abs/1707.03815>`_ paper, modified to include
    10-fold cross validation splits similar to WebKB datasets.
    
    Nodes represent documents and edges represent citation links.
    Datasets include :obj:`"Cora"`, :obj:`"Cora_ML"`, :obj:`"CiteSeer"`,
    :obj:`"DBLP"`, :obj:`"PubMed"`.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"Cora"`, :obj:`"Cora_ML"`
            :obj:`"CiteSeer"`, :obj:`"DBLP"`, :obj:`"PubMed"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        to_undirected (bool, optional): Whether the original graph is
            converted to an undirected one. (default: :obj:`True`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
        split_seed (int, optional): Random seed for generating reproducible
            splits. (default: :obj:`42`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #nodes
          - #edges
          - #features
          - #classes
        * - Cora
          - 19,793
          - 126,842
          - 8,710
          - 70
        * - Cora_ML
          - 2,995
          - 16,316
          - 2,879
          - 7
        * - CiteSeer
          - 4,230
          - 10,674
          - 602
          - 6
        * - DBLP
          - 17,716
          - 105,734
          - 1,639
          - 4
        * - PubMed
          - 19,717
          - 88,648
          - 500
          - 3

    **Note:** This implementation creates 10 random splits for cross-validation.
    The train_mask, val_mask, and test_mask will have shape [num_nodes, 10]
    where each column represents a different split.
    """

    url = 'https://github.com/abojchevski/graph2gauss/raw/master/data/{}.npz'

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        to_undirected: bool = True,
        force_reload: bool = False,
        split_seed: int = 42,
    ) -> None:
        self.name = name.lower()
        self.to_undirected = to_undirected
        self.split_seed = split_seed
        assert self.name in ['cora', 'cora_ml', 'citeseer', 'dblp', 'pubmed']
        
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name.lower(), 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name.lower(), 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        suffix = 'undirected' if self.to_undirected else 'directed'
        return f'data_{suffix}.pt'

    def download(self) -> None:
        # Check if file already exists
        raw_path = osp.join(self.raw_dir, self.raw_file_names)
        if osp.exists(raw_path):
            print(f"Using existing file {self.raw_file_names}")
            return
            
        download_url(self.url.format(self.name), self.raw_dir)
        print(f"Downloaded main data file: {self.raw_file_names}")

    def _generate_splits(self, num_nodes: int, num_splits: int = 10, seed: int = 42) -> tuple:
        """Generate reproducible random train/val/test splits for cross-validation.
        
        Args:
            num_nodes (int): Number of nodes in the graph
            num_splits (int): Number of splits to generate (default: 10)
            seed (int): Random seed for reproducibility (default: 42)
            
        Returns:
            tuple: (train_masks, val_masks, test_masks) each of shape [num_nodes, num_splits]
        """
        with torch.random.fork_rng():
            torch.manual_seed(seed)

            train_masks = torch.zeros(num_nodes, num_splits, dtype=torch.bool)
            val_masks = torch.zeros(num_nodes, num_splits, dtype=torch.bool)
            test_masks = torch.zeros(num_nodes, num_splits, dtype=torch.bool)

            # Standard split ratios: 60% train, 20% val, 20% test
            train_ratio = 0.6
            val_ratio = 0.2

            for split_idx in range(num_splits):
                # Create random permutation for this split with different seed per split
                # This ensures different splits while maintaining reproducibility
                torch.manual_seed(seed + split_idx)
                perm = torch.randperm(num_nodes)

                train_end = int(train_ratio * num_nodes)
                val_end = int((train_ratio + val_ratio) * num_nodes)

                train_idx = perm[:train_end]
                val_idx = perm[train_end:val_end]
                test_idx = perm[val_end:]

                train_masks[train_idx, split_idx] = True
                val_masks[val_idx, split_idx] = True
                test_masks[test_idx, split_idx] = True
            
        return train_masks, val_masks, test_masks

    def process(self) -> None:
        # Check if processed file already exists with splits
        if osp.exists(self.processed_paths[0]) and not self.force_reload:
            # File exists, no need to process again
            return

        print(f"Generating new splits with seed {self.split_seed}...")

        data = read_npz(self.raw_paths[0], to_undirected=self.to_undirected)

        num_nodes = data.x.size(0)
        train_masks, val_masks, test_masks = self._generate_splits(
            num_nodes, seed=self.split_seed
        )

        device = data.x.device
        data.train_mask = train_masks.to(device)
        data.val_mask   = val_masks.to(device)
        data.test_mask  = test_masks.to(device)

        data.split_seed = self.split_seed
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])
        print(f"Splits saved to {self.processed_paths[0]}")

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}Full()'

    def get_split(self, split_idx: int = 0):
        """Get masks for a specific split.
        
        Args:
            split_idx (int): Index of the split (0-9)
            
        Returns:
            tuple: (train_mask, val_mask, test_mask) for the specified split
        """
        if split_idx < 0 or split_idx >= 10:
            raise ValueError(f"Split index must be between 0 and 9, got {split_idx}")
            
        data = self[0]
        return (
            data.train_mask[:, split_idx],
            data.val_mask[:, split_idx], 
            data.test_mask[:, split_idx]
        )

    def get_split_info(self) -> dict:
        """Get information about all splits.
        
        Returns:
            dict: Statistics about the splits
        """
        data = self[0]
        info = {
            'num_splits': 10,
            'num_nodes': data.x.size(0),
            'num_features': data.x.size(1),
            'num_classes': data.y.unique().numel(),
            'split_stats': []
        }
        
        for i in range(10):
            train_mask, val_mask, test_mask = self.get_split(i)
            split_stat = {
                'split_idx': i,
                'train_nodes': int(train_mask.sum()),
                'val_nodes': int(val_mask.sum()),
                'test_nodes': int(test_mask.sum())
            }
            info['split_stats'].append(split_stat)
            
        return info