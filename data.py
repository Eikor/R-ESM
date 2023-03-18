
from typing import Sequence, Tuple
import torch
from torch.utils.data import DistributedSampler
import math
from typing import TypeVar, Optional, Iterator
from esm.data import Alphabet, FastaBatchedDataset
import itertools
import random 
from typing import Sequence, Tuple, List, Union
from tqdm import tqdm
import torch.nn.functional as F



__all__ = ["DistributedSampler", ]

T_co = TypeVar('T_co', covariant=True)

RNAseq_toks = {
    'toks': ['T', 'A', 'G', 'C'],
    'amb_toks': ['M', 'R', 'W', 'S', 'Y', 'K', 'V', 'H', 'D', 'B'],
    'pair_toks': ['AC', 'AG', 'AT', 'CG', 'CT', 'GT', 'ACG', 'ACT', 'AGT', 'CGT']
}

Rand_toks = {

}

class Alphabet_RNA(Alphabet):
    def __init__(
        self,
        prepend_toks: Sequence[str] = ("<null_0>", "<pad>", "<eos>", "<unk>"),
        append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"),
        prepend_bos: bool = True,
        append_eos: bool = False,
        use_msa: bool = False,
        coden_size: int = 2,
    ):
        self.standard_toks = RNAseq_toks['toks']
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.use_msa = use_msa
        
        ####
        self.coden_size = coden_size
        self.amb_toks = RNAseq_toks['amb_toks']
        self.amb_to_pair = {
                amb_tok: pair_tok \
                for amb_tok, pair_tok in zip(RNAseq_toks['amb_toks'], RNAseq_toks['pair_toks'])
                }
        ####

        self.all_toks = list(self.prepend_toks)
        for tok in itertools.product(self.standard_toks, repeat=self.coden_size):
            self.all_toks.append(''.join(tok))
        for i in range((8 - (len(self.all_toks) % 8)) % 8):
            self.all_toks.append(f"<null_{i  + 1}>")
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")
        self.all_special_tokens = ['<eos>', '<unk>', '<pad>', '<cls>', '<mask>']
        self.unique_no_split_tokens = self.prepend_toks + self.standard_toks + \
                self.append_toks + self.amb_toks
        self.kernel = torch.tensor([1]).expand(1, 1, coden_size)

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    @classmethod
    def RNA(cls, coden_size=2) -> "Alphabet":
        prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
        append_toks = ("<mask>",)
        prepend_bos = True
        append_eos = True
        use_msa = False
        return cls(prepend_toks, append_toks, prepend_bos, append_eos, use_msa, coden_size=coden_size)

    def _tokenize(self, text) -> str:
        if text in self.amb_toks:
            pair = self.amb_to_pair[text]
            tok = pair[int(random.random()*len(pair))]
        else:
            tok = text.split()
        return tok

    def tokenize(self, text, **kwargs) -> List[str]:
        """
        Inspired by https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py
        Converts a string in a sequence of tokens, using the tokenizer.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.

        Returns:
            :obj:`List[str]`: The list of tokens.
        """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # AddedToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # We strip left and right by default
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        padding = self.coden_size -1
        if padding > 0:
            tokenized_text.extend(tokenized_text[-1:] * padding)
        tokenized_text = [''.join(tokenized_text[i:i+self.coden_size]) for i in range(len(tokenized_text) - padding)]
        return tokenized_text

    def encode(self, text):
        return [self.tok_to_idx[tok] for tok in self.tokenize(text)]


class RNADataset(FastaBatchedDataset):
    def __init__(self, sequence_labels, sequence_strs):
        self.sequence_labels = list(sequence_labels)
        self.sequence_strs = list(sequence_strs)

    @classmethod
    def from_file(cls, fasta_file):
        sequence_labels, sequence_strs = [], []
        cur_seq_label = None
        buf = []

        def _flush_current_seq():
            nonlocal cur_seq_label, buf
            if cur_seq_label is None:
                return
            sequence_strs.append("".join(buf))
            if 'N' in sequence_strs[-1]:
                sequence_strs.pop()
            else:
                sequence_labels.append(cur_seq_label)
            cur_seq_label = None
            buf = []

        with open(fasta_file, "r") as infile:
            for line_idx, line in tqdm(enumerate(infile)):
                if line.startswith(">"):  # label line
                    _flush_current_seq()
                    line = line[1:].strip()
                    if len(line) > 0:
                        cur_seq_label = line
                    else:
                        cur_seq_label = f"seqnum{line_idx:09d}"
                else:  # sequence line
                    buf.append(line.strip())

        _flush_current_seq()

        assert len(set(sequence_labels)) == len(
            sequence_labels
        ), "Found duplicate sequence labels"

        return cls(sequence_labels, sequence_strs)

    def __len__(self):
        return len(self.sequence_labels)

    def __getitem__(self, idx):
        return self.sequence_labels[idx], self.sequence_strs[idx]

    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
        sizes = [(len(s), i) for i, s in enumerate(self.sequence_strs)]
        sizes.sort()
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0

        for sz, i in sizes:
            sz += extra_toks_per_seq
            if max(sz, max_len) * (len(buf) + 1) > toks_per_batch:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)

        _flush_current_buf()
        return batches

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
        
        max_len = max(len(seq_str) for seq_str in seq_str_list)
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
        corrupt_prob = torch.rand_like(masked_tokens, dtype=torch.float)
        corrupt_prob = (corrupt_prob - 0.85) / 0.15
        corrupt_prob.clamp_(min=0)
        mask_prob = corrupt_prob > 0.2 # 80% change to mask
        all_masks = corrupt_prob > 0

        for shift in range(self.alphabet.coden_size - 1):
            mask_prob[:, :-shift-1] += mask_prob.roll(-shift-1)[:, :-shift-1]
            all_masks[:, :-shift-1] += all_masks.roll(-shift-1)[:, :-shift-1]
        
        #rectify mask prob
        corrupt_prob[:, :int(self.alphabet.prepend_bos)] = 0
        mask_prob[:, :int(self.alphabet.prepend_bos)] = False

        for i, (label, seq_str) in enumerate(
            zip(batch_labels, seq_str_list)
        ):
            labels.append(label)
            strs.append(seq_str)
            seq_encoded = self.alphabet.encode(seq_str)
            if self.truncation_seq_length:
                seq_encoded = seq_encoded[:self.truncation_seq_length]
            
            start_idx = int(self.alphabet.prepend_bos)
            end_idx = len(seq_encoded) + int(self.alphabet.prepend_bos)
            
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[i, start_idx : end_idx] = seq

            # 10% change random acid
            corrupt_str = seq_str
            for random_i in torch.where(corrupt_prob[i, start_idx : end_idx]>0.1)[0]:
                corrupt_str = corrupt_str[:random_i]+\
                    random.choice(RNAseq_toks['toks'])+corrupt_str[random_i+1:]
            corrupt_tokens = self.alphabet.encode(corrupt_str)
            if self.truncation_seq_length:
                corrupt_tokens = corrupt_tokens[:self.truncation_seq_length]

            corrupt_seq = torch.tensor(corrupt_tokens, dtype=torch.int64)
            masked_tokens[i, start_idx : end_idx] = corrupt_seq
            
            # 80% change to mask
            masked_tokens[i, start_idx : end_idx].masked_fill_(
                    mask_prob[i, start_idx : end_idx], 
                    self.alphabet.mask_idx)
            
            masks[i, start_idx : end_idx] = 1 * all_masks[i, start_idx : end_idx]

            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
                masked_tokens[i, 0] = self.alphabet.cls_idx    
        
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
