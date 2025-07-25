import torch
import numpy as np
import pandas as pd
import warnings
import random
from typing import Optional, Union
from scipy.sparse import issparse, spmatrix
from torch.utils.data import Dataset, Sampler, DataLoader
from anndata import AnnData


class AnnDataDataset(Dataset):
    def __init__(self, 
                 adata: AnnData, 
                 celltype_key: Optional[str] = None, 
                 source_key: Optional[str] = None) -> None:
        if not isinstance(adata, AnnData):
            raise TypeError("adata must be an AnnData object.")
        if celltype_key is not None and not isinstance(celltype_key, str):
            raise TypeError("celltype_key must be a string or None.")
        if source_key is not None and not isinstance(source_key, str):
            raise TypeError("source_key must be a string or None.")

        if not hasattr(adata, "obs") or not hasattr(adata, "X"):
            raise ValueError("adata must have attributes 'obs' and 'X'.")
        if not hasattr(adata, "obsm"):
            raise ValueError("adata must have attribute 'obsm'.")

        self.X = self._convert_to_tensor(adata.X)
        self.celltypes = self._extract_obs_column(adata, celltype_key)
        self.sources = self._encode_source_column(adata, source_key)
        self.link_feat = self._convert_to_tensor(adata.obsm['link_feat']) if 'link_feat' in adata.obsm else None

    def _extract_obs_column(self, adata: AnnData, key: Optional[str]) -> Optional[np.ndarray]:
        if key is None:
            return None
        if key not in adata.obs:
            raise KeyError(f"Column '{key}' not found in adata.obs.")
        return adata.obs[key].to_numpy()

    def _encode_source_column(self, adata: AnnData, key: Optional[str]) -> Optional[np.ndarray]:
        if key is None:
            return None
        if key not in adata.obs:
            raise KeyError(f"Column '{key}' not found in adata.obs.")
        
        col = adata.obs[key]
        codes = col.cat.codes if isinstance(col, pd.Categorical) else pd.Categorical(col).codes
        return np.where(codes == -1, 0, codes).astype(np.int64)

    def _convert_to_tensor(
        self,
        data: Union[np.ndarray, spmatrix]
    ) -> torch.Tensor:
        if issparse(data):
            warnings.warn("adata.X is sparse; converting to dense.")
            data = data.toarray()
        return torch.from_numpy(data).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict:
        if not isinstance(idx, int):
            raise TypeError("Index must be an integer.")
        
        sample = {'expression': self.X[idx], 'index': idx}

        if self.link_feat is not None:
            sample['link_feat'] = self.link_feat[idx]
        if self.celltypes is not None:
            sample['celltype'] = self.celltypes[idx]
        if self.sources is not None:
            sample['source'] = torch.tensor(self.sources[idx], dtype=torch.long)

        return sample

    @property
    def feature_shapes(self) -> dict:
        shapes = {'expression': self.X.shape[1]}
        if self.link_feat is not None:
            shapes['link_feat'] = self.link_feat.shape[1] if self.link_feat.ndim > 1 else 1
        return shapes

    @property
    def source_categories(self) -> int:
        return 1 if self.sources is None else len(np.unique(self.sources))

    @property
    def celltype_categories(self) -> Optional[int]:
        return None if self.celltypes is None else len(np.unique(self.celltypes))


class InfiniteRandomSampler(Sampler):
    def __init__(self, data_source: Dataset, batch_size: int) -> None:
        if not isinstance(data_source, Dataset):
            raise TypeError("data_source must be a Dataset.")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        
        self.sample_num = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield random.choices(range(self.sample_num), k=self.batch_size)

    def __len__(self) -> int:
        return 10**10


def load_data(dataset: Dataset, batch_size: int) -> DataLoader:
    if not isinstance(dataset, Dataset):
        raise TypeError("dataset must be a torch.utils.data.Dataset instance.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    sampler = InfiniteRandomSampler(dataset, batch_size)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    return dataloader
