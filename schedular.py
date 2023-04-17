import torch
import math

class Scheduler:
    def __init__(self, model, optim, loss_scaler, lr_scheduler) -> None:
        self.model = model
        self.optim = optim
        self.scaler = loss_scaler
        self.scheduler = lr_scheduler

    def loss_scale(self, loss:torch.Tensor)->torch.Tensor:
        return self.scaler.scale(loss)
    
    def zero_grad(self):
        self.optim.zero_grad()

    def loss_scale_and_backward(self, loss:torch.Tensor, create_graph=False):
        loss = self.loss_scale(loss)
        loss.backward(create_graph=create_graph)

    def step_and_lr_schedule(self, epoch, clip_grad=None, update_grad=True):
        if update_grad:
            if clip_grad is not None:
                self.scaler.unscale_(self.optim)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            else:
                self.scaler.unscale_(self.optim)
                norm = get_grad_norm_(self.model.parameters())
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            norm = None
        lr = self.scheduler.lr_schedule(self.optim, epoch)
        return lr

class Scheduler1:
    def __init__(self, model, optim, lr_scheduler) -> None:
        self.model = model
        self.optim = optim
        self.scheduler = lr_scheduler

    def zero_grad(self):
        self.optim.zero_grad()

    def loss_scale_and_backward(self, loss:torch.Tensor, create_graph=False):
        loss.backward(create_graph=create_graph)

    def step_and_lr_schedule(self, epoch, clip_grad=None, update_grad=True):
        if update_grad:
            # if clip_grad is not None:
            #     self.scaler.unscale_(self.optim)  # unscale the gradients of optimizer's assigned params in-place
            #     norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            # else:
            #     self.scaler.unscale_(self.optim)
            #     norm = get_grad_norm_(self.model.parameters())
            self.optim.step()
            # self.optim.update()
        else:
            norm = None
        lr = self.scheduler.lr_schedule(self.optim, epoch)
        return lr
    
def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == math.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

class LinearScheduler:
    def __init__(self, args) -> None:
        self.warmup_epochs = args.warmup_epochs
        self.epochs = args.epochs
        self.lr = args.lr
        self.min_lr = args.min_lr
    
    def lr_schedule(self, optimizer, epoch):
        if epoch < self.warmup_epochs:
            lr = self.lr * epoch / self.warmup_epochs 
        else:
            lr = self.lr - (self.lr - self.min_lr) * \
               (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)

        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr
