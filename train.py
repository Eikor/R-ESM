import torch
import math
import sys

import dist_misc

def train_one_epoch(model, data_loader, criterion, training_scheduler, epoch, log_writer=None, args=None, finetune='cls'):
    model.train()
    metric_logger = dist_misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', dist_misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200
    accum_iter = 1 if args.accum_iter < 1 else args.accum_iter

    for batch_idx, (labels, strs, toks, masktoks, masks) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if finetune == 'cls':
            labels = torch.tensor(labels).long()
        else:
            labels = torch.tensor(labels)

        if torch.cuda.is_available() and not args.nogpu:
            toks = toks.to(device="cuda", non_blocking=True) 
            masktoks = masktoks.to(device="cuda", non_blocking=True)
            masks = masks.to(device="cuda", non_blocking=True)
            labels = labels.to(device='cuda', non_blocking=True)

        with torch.cuda.amp.autocast():
            if finetune:
                pred = model(toks)
                loss = criterion(pred, labels)
            else:
                out = model(masktoks)
                logits = out["logits"].permute(0, 2, 1) # B*C*D
                loss = criterion(logits, toks, masks)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        
        if (batch_idx + 1) % accum_iter == 0:
            training_scheduler.zero_grad()
            training_scheduler.loss_scale_and_backward(loss)
            training_scheduler.step_and_lr_schedule(batch_idx / len(data_loader) + epoch)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = training_scheduler.optim.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = dist_misc.all_reduce_mean(loss_value)
        if log_writer is not None and (batch_idx + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((batch_idx / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


