import torch
import numpy as np
import pandas as pd
import warnings
import random
from scipy.sparse import issparse, spmatrix
from torch.utils.data import Dataset, Sampler, DataLoader
from anndata import AnnData
from typing import Optional, Union, Dict


class AnnDataDataset(Dataset):
    def __init__(
        self,
        adata: AnnData,
        input_key: Optional[str] = None,
        output_layer: Optional[str] = None,
        celltype_key: Optional[str] = None,
        source_key: Optional[str] = None,
    ) -> None:
        if not isinstance(adata, AnnData):
            raise TypeError("adata must be an AnnData object.")
        if not hasattr(adata, "obs"):
            raise ValueError("adata must have 'obs' attribute.")

        self.input = self._get_input_tensor(adata, input_key)
        self.output = self._get_output_tensor(adata, output_layer)
        self.celltypes = (
            self._encode_obs_column(adata, celltype_key)
            if celltype_key is not None
            else None
        )
        self.sources = (
            self._encode_obs_column(adata, source_key)
            if source_key is not None
            else None
        )

    def _get_input_tensor(self, adata: AnnData, input_key: Optional[str]) -> torch.Tensor:
        if input_key is None:
            data = adata.X
        elif input_key in adata.obsm:
            data = adata.obsm[input_key]
        elif hasattr(adata, input_key):
            data = getattr(adata, input_key)
        else:
            warnings.warn(f"{input_key} not found; using adata.X instead.")
            data = adata.X
        return self._convert_to_tensor(data)

    def _get_output_tensor(self, adata: AnnData, output_layer: Optional[str]) -> Optional[torch.Tensor]:
        if output_layer is None:
            return None
        if output_layer not in adata.layers:
            warnings.warn(f"{output_layer} not found in adata.layers; output=None.")
            return None
        return self._convert_to_tensor(adata.layers[output_layer])

    def _encode_obs_column(self, adata: AnnData, key: Optional[str]) -> Optional[np.ndarray]:
        if key not in adata.obs:
            raise KeyError(f"{key} not found in adata.obs.")
        col = adata.obs[key]
        cat = pd.Categorical(col)
        codes = cat.codes
        return np.where(codes == -1, 0, codes + 1).astype(np.int64)

    def _convert_to_tensor(self, data: Union[np.ndarray, spmatrix]) -> torch.Tensor:
        if issparse(data):
            warnings.warn("Sparse data detected; converting to dense.")
            data = data.toarray()
        return torch.from_numpy(np.asarray(data)).float()

    def __len__(self) -> int:
        return self.input.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not isinstance(idx, int):
            raise TypeError("Index must be an integer.")
        sample = {"input": self.input[idx], "index": idx}
        if self.output is not None:
            sample["output"] = self.output[idx]
        if self.celltypes is not None:
            sample["celltype"] = torch.tensor(self.celltypes[idx], dtype=torch.long)
        if self.sources is not None:
            sample["source"] = torch.tensor(self.sources[idx], dtype=torch.long)
        return sample

    @property
    def feature_shapes(self) -> Dict[str, int]:
        shapes = {"input": self.input.shape[1]}
        if self.output is not None:
            shapes["output"] = self.output.shape[1]
        return shapes

    @property
    def source_categories(self) -> Optional[int]:
        if self.sources is None:
            return 1
        return int(len(np.unique(self.sources)))

    @property
    def celltype_categories(self) -> Optional[int]:
        if self.celltypes is None:
            return None
        return int(len(np.unique(self.celltypes)))


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

