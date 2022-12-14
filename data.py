
from typing import Sequence, Tuple
import torch

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