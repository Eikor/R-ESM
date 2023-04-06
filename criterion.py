import torch
import torch.nn as nn

class MaskedPredictionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, toks, masks=None):
        if masks is not None:
            return torch.sum(self.criterion(logits, toks) * masks) / torch.sum(masks)
        return torch.mean(self.criterion(logits, toks))
    
class CLS_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, gt):
        return torch.mean(self.criterion(pred, gt))