
import torch
import math
import sys

def train_one_epoch(model, data_loader, criterion, training_scheduler, epoch, args):

    accum_iter = 1 if args.accum_iter < 1 else args.accum_iter

    for batch_idx, (labels, strs, toks, masktoks, masks) in enumerate(data_loader):
        if torch.cuda.is_available() and not args.nogpu:
            toks = toks.to(device="cuda", non_blocking=True)
            masktoks = masktoks.to(device="cuda", non_blocking=True)

        with torch.cuda.amp.autocast():
            out = model(masktoks, repr_layers=args.repr_layers, return_contacts=args.return_contacts)
            logits = out["logits"].permute(0, 2, 1) # B*C*D
            loss = criterion(logits, toks)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        
        if (batch_idx + 1) % accum_iter == 0:
            training_scheduler.zero_grad()
            training_scheduler.loss_scale_and_backward(loss)
            training_scheduler.step_and_lr_schedule(data_iter_step / len(data_loader) + epoch, )

        torch.cuda.synchronize()

        representations = {
            layer: t.to(device="cpu") for layer, t in out["representations"].items()
        }



