
from typing import Sequence, Tuple
import torch
from torch.utils.data import DistributedSampler
import math
from typing import TypeVar, Optional, Iterator

import torch

import torch.distributed as dist

__all__ = ["DistributedSampler", ]

T_co = TypeVar('T_co', covariant=True)

class MaskedBatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) masked batch.
    """

    def __init__(self, alphabet, truncation_seq_length: int = None):
        self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        batch_labels, seq_str_list = zip(*raw_batch)
        seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list]
        if self.truncation_seq_length:
            seq_encoded_list = [seq_str[:self.truncation_seq_length] for seq_str in seq_encoded_list]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        tokens = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        ### masking ###
        masks = torch.zeros_like(tokens)
        masked_tokens = tokens.clone()
        random_tokens = torch.randint_like(masked_tokens, 4, 31) # amino acid 'L' to '-' encode as 4 to 30 
        corrupt_prob = torch.randn_like(masked_tokens, dtype=float)
        corrupt_prob = (corrupt_prob - 0.85) / 0.15
        corrupt_prob.clamp_(min=0)

        for i, (label, seq_str, seq_encoded) in enumerate(
            zip(batch_labels, seq_str_list, seq_encoded_list)
        ):
            labels.append(label)
            strs.append(seq_str)
            start_idx = int(self.alphabet.prepend_bos)
            end_idx = len(seq_encoded) + int(self.alphabet.prepend_bos)
            
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
                masked_tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[i, start_idx : end_idx] = seq

            # 10% change random acid
            masked_tokens[i, start_idx : end_idx] = (tokens \
                + (random_tokens-tokens) * (corrupt_prob>0.1))[i, start_idx : end_idx]
                
            # 80% change to mask
            masked_tokens[i, start_idx : end_idx].masked_fill_(
                    (corrupt_prob>0.2)[i, start_idx : end_idx], 
                    self.alphabet.mask_idx)
            
            masks[i, start_idx : end_idx] = 1 * (corrupt_prob > 0)[i, start_idx : end_idx]
            
            if self.alphabet.append_eos:
                tokens[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
                masked_tokens[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx

        return labels, strs, tokens, masked_tokens, masks
    
class DistributedBatchSampler(DistributedSampler):
    def __init__(self, dataset, batch_index, num_replicas: Optional[int] = None, rank: Optional[int] = None, shuffle: bool = True, seed: int = 0, drop_last: bool = False) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.batch_index = batch_index
        
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.batch_index) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.batch_index) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.batch_index) / self.num_replicas)  # type: ignore[arg-type]

        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.batch_index), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.batch_index)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter([self.batch_index[i] for i in indices])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
