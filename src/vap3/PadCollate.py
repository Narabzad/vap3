from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch


class PadCollate:
    def __init__(
        self,
        pad_token_id: int,
        pad_token_type_id: int,
    ):
        self._pad_token_id = pad_token_id
        self._pad_token_type_id = pad_token_type_id

    def pad_to(self, tensor, size, pad_value=0):
        # Convert size to tuple of integers
        size = tuple(int(s) for s in size)
        
        # Create a new tensor of the desired size with the pad value
        padded = torch.full(size, pad_value, dtype=tensor.dtype)
        
        # Copy the original tensor into the padded tensor
        slices = tuple(slice(0, min(s, d)) for s, d in zip(tensor.size(), size))
        padded[slices] = tensor
        return padded

    def get_pad_id(self, key: str) -> int:
        if key.endswith("input_ids"):
            return self._pad_token_id
        elif key.endswith("attention_mask"):
            return 0
        elif key.endswith("token_type_ids"):
            return self._pad_token_type_id
        else:
            raise ValueError(f"Unknown key {key}")

    def __call__(
        self, batch: Sequence[Mapping[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        from torch.utils.data.dataloader import default_collate  # type: ignore

        keys = {k for k, v in batch[0][0].items() if isinstance(v, torch.Tensor)}
        unalign = {k for k in keys if len(set(x[0][k].size() for x in batch)) > 1}
        sizes = {k: np.max([x[0][k].size() for x in batch], axis=0) for k in unalign}
        
        data = [
            {
                k: self.pad_to(v, sizes[k], self.get_pad_id(k)) if k in unalign else v
                for k, v in x[0].items()
            }
            for x in batch
        ]
        
        queries = [x[1] for x in batch]
        MAPScore = [x[2] for x in batch]
        
        collated: Dict[str, torch.Tensor] = default_collate(data)
        
        return collated, queries, MAPScore
