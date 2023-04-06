import torch
import torch.utils.data
import torch.nn as nn
import time
import datetime
import os
import json
from data import MaskedBatchConverter, DistributedBatchSampler, Alphabet_RNA, RNADataset
from RESM import RESM
from args import create_parser
from criterion import CLS_loss
from schedular import Scheduler, LinearScheduler
import dist_misc
from train import train_one_epoch
import numpy as np
import random

class CLS_model(nn.Module):
    def __init__(self, pretrain_url, nb_classes, resume=False, repr_layers=[12], reduce='mean'):
        super().__init__()
        pretrain_info = torch.load(pretrain_url)
        self.args = pretrain_info['args']
        self.alphabet = Alphabet_RNA.RNA(coden_size=self.args.coden_size)
        self.resm = RESM(self.alphabet, num_layers=self.args.num_layers, embed_dim=self.args.embed_dim, attention_heads=20)
        if resume:
            self.resm.load_state_dict(pretrain_info['model'])
        self.reduce = reduce
        self.repr_layers = repr_layers
        self.cls_head = nn.Linear(self.args.embed_dim, nb_classes)

    def forward(self, tokens):
        repr = self.resm(tokens, self.repr_layers)['representations'][self.repr_layers[0]]
        if self.reduce == 'mean':
            repr = repr[:, 1:-1].mean(dim=1)
        else:
            repr = repr[:, 0]
        return torch.softmax(self.cls_head(repr), dim=-1)
    
    def run_test(self, testset):
        self.eval()
        pass

class Seq_Dataset(torch.utils.data.Dataset):
    def __init__(self, seq_data) -> None:
        super().__init__()
        self.seqs = seq_data.reset_index()
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, index):
        return self.seqs.loc[index, 'labels'], self.seqs.loc[index, 'seq'][0]

def finetune_cls(pretrain_url, dataset, label_names, 
                 save_name=None, epochs=10, resume=True, repr_layers=[12], reduce='mean'):
    # load training result
    model = CLS_model(pretrain_url, len(label_names), resume=resume, repr_layers=repr_layers, reduce=reduce)
    args = model.args
    # distribute init
    dist_misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + dist_misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # prepare model
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.98), eps=10e-8, weight_decay=0.01)
    criterion = CLS_loss()
    training_scheduler = Scheduler(model, optimizer, torch.cuda.amp.GradScaler(), LinearScheduler(args))
    

    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")

    
    data_loader_train = torch.utils.data.DataLoader(
        Seq_Dataset(dataset), collate_fn=MaskedBatchConverter(model.alphabet, args.truncation_seq_length), num_workers=4,
    )

    print(f"Fine-tuning with {len(dataset)} sequences")

    # prepare logger
    os.makedirs(save_name, exist_ok=True)
    log_writer = None
    
    dist_misc.load_model(args=args, model_without_ddp=model, scheduler=training_scheduler)

    print(f"Start training for {epochs} epochs")
    start_time = time.time()
    for epoch in range(epochs):
        if args.distributed:
            data_loader_train.batch_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            criterion,training_scheduler,  epoch,
            log_writer=None,
            args=args,
            finetune=True
        )
        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            dist_misc.save_model(
                args=args, model=model, model_without_ddp=model, scheduler=training_scheduler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and dist_misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def finetune_reg():
    pass
