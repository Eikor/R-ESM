import torch
import torch.nn as nn

class MaskedPredictionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, toks, masks=None):
        if masks is not None:
            return torch.mean(self.criterion(logits, toks) * masks)
        return torch.mean(self.criterion(logits, toks))