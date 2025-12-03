import torch
import numpy as np
import pandas as pd
import warnings
import random
from scipy.sparse import issparse, spmatrix
from torch.utils.data import Dataset, Sampler, DataLoader, RandomSampler, SequentialSampler, BatchSampler
from anndata import AnnData
from typing import Optional, Union, Dict, Literal, Iterator


class AnnDataDataset(Dataset):
    def __init__(
        self,
        adata: AnnData,
        input_key: Optional[str] = None,
        output_layer: Optional[str] = None,
        celltype_key: Optional[str] = None,
        source_key: Optional[str] = None,
        mode: Literal["integration", "imputation"] = "integration"
    ):
        self.mode = mode
        if not isinstance(adata, AnnData):
            raise TypeError("adata must be an AnnData object.")
        if not hasattr(adata, "obs"):
            raise ValueError("adata must have 'obs' attribute.")

        self.input = self._get_input_tensor(adata, input_key)

        if self.mode == "imputation":
            if output_layer is None:
                raise ValueError("`output_layer` must be provided in imputation mode.")
            self.output = self._get_output_tensor(adata, output_layer)
        else:
            self.output = None

        self.link_feat = self._convert_to_tensor(
            adata.obsm["link_feat"] if "link_feat" in adata.obsm else None
            )

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

    def _get_output_tensor(self, adata: AnnData, output_layer: Optional[str]) -> torch.Tensor:
        if output_layer not in adata.layers:
            raise KeyError(f"Output layer '{output_layer}' not found in adata.layers.")
        return self._convert_to_tensor(adata.layers[output_layer])

    def _encode_obs_column(self, adata: AnnData, key: str) -> np.ndarray:
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
        sample = {"input": self.input[idx], "index": idx}

        if self.mode == "integration":
            if self.celltypes is not None:
                sample["celltype"] = torch.tensor(self.celltypes[idx], dtype=torch.long)
            if self.sources is not None:
                sample["source"] = torch.tensor(self.sources[idx], dtype=torch.long)
            if self.link_feat is not None:
                sample['link_feat'] = self.link_feat[idx]
        elif self.mode == "imputation":
            if self.output is None:
                raise ValueError("output_layer must be specified for imputation mode.")
            sample["output"] = self.output[idx]

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
    """Sampler that yields infinite random batches."""
    def __init__(self, data_source: Dataset, batch_size: int) -> None:
        self.sample_num = len(data_source)
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[list[int]]:
        while True:
            yield random.choices(range(self.sample_num), k=self.batch_size)

    def __len__(self) -> int:
        return 10**10


def load_data(
    dataset: Dataset,
    batch_size: int,
    mode: Literal["integration", "imputation"] = "integration",
    shuffle: bool = True
) -> DataLoader:
    if mode == "integration":
        sampler = InfiniteRandomSampler(dataset, batch_size)
        dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    else:  # imputation
        if shuffle:
            sampler = RandomSampler(dataset, replacement=False)
        else:
            sampler = SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0)

    return dataloader