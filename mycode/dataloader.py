from torch.utils.data import DataLoader, Dataset, Sampler
import torch
import numpy as np
import pandas as pd
import random
import scipy.sparse as sp


class AnnDataDataset(Dataset):
    def __init__(self, adata, celltype_key=None, source_key=None):
        if not hasattr(adata, 'obs') or not hasattr(adata, 'X'):
            raise ValueError("Invalid AnnData object")
        if not hasattr(adata, 'obsm'):
            raise ValueError("AnnData missing obsm attribute")

        self.X = self._convert_to_tensor(adata.X)
        self.celltypes = self._extract_obs_data(adata, celltype_key)
        self.sources = self._encode_sources(adata, source_key)
        self.link_feat = self._convert_to_tensor(adata.obsm['link_feat']) if 'link_feat' in adata.obsm else None
    def _extract_obs_data(self, adata, key):
        if key is None:
            return None
        if key not in adata.obs:
            raise KeyError(f"Column '{key}' not found in adata.obs")
        return adata.obs[key].to_numpy()

    def _encode_sources(self, adata, source_key):
        if source_key is None:
            return None
        if source_key not in adata.obs:
            raise KeyError(f"Column '{source_key}' not found in adata.obs")
            
        source_data = adata.obs[source_key]
        if isinstance(source_data, pd.Categorical):
            encoded = source_data.cat.codes
        else:
            encoded = pd.Categorical(source_data).codes
            
        encoded = np.where(encoded == -1, 0, encoded)
        return encoded.astype(np.int64)

    def _convert_to_tensor(self, data):
        if sp.issparse(data):
            data = data.toarray()
        if isinstance(data, (pd.Series, pd.Categorical)):
            data = np.array(data)
        return torch.from_numpy(data).float() if isinstance(data, np.ndarray) else torch.tensor(data).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {
            'expression': self.X[idx],
            'index': idx,
        }
        
        if self.link_feat is not None:
            sample['link_feat'] = self.link_feat[idx]
            
        if self.celltypes is not None:
            sample['celltype'] = self.celltypes[idx]
            
        if self.sources is not None:
            sample['source'] = torch.tensor(self.sources[idx], dtype=torch.long)
            
        return sample


    @property
    def feature_shapes(self):
        shapes = {'expression': self.X.shape[1]}
        if self.link_feat is not None:
            shapes['link_feat'] = self.link_feat.shape[1] if self.link_feat.ndim > 1 else 1
        return shapes
    @property
    def source_categories(self):
        if self.sources is None:
            return 1
        return len(np.unique(self.sources))
    
    @property
    def celltype_categories(self):
        if self.celltypes is None:
            return None
        return len(np.unique(self.celltypes))

class InfiniteRandomSampler(Sampler):
    def __init__(self, data_source, batch_size):
        super().__init__(data_source)
        self.sample_num = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            #yield random.sample(range(self.sample_num), k=self.batch_size)
            yield random.choices(range(self.sample_num), k=self.batch_size)  #有放回采样

    def __len__(self):
        return 10**10


def load_data(dataset, batch_size):
    sampler = InfiniteRandomSampler(dataset, batch_size)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    return dataloader

