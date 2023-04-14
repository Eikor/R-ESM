import torch
import torch.utils.data
import torch.nn as nn
import time
import datetime
import os
import json
from tqdm import tqdm
from data import MaskedBatchConverter, Alphabet_RNA, DistributedBatchSampler
from RESM import RESM
from esm.modules import ESM1bLayerNorm, gelu
from criterion import CLS_loss
from schedular import Scheduler, LinearScheduler
import dist_misc
from train import train_one_epoch
import numpy as np
import random
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def classification_summary(labels=[], preds=[], label_names=[], save_name='cls_summary'):
    cls_res = classification_report(labels, preds, target_names=label_names, digits=6)
    print(cls_res)
    if save_name is not None:
        with open(save_name + '.txt', 'a') as f:
            f.write(cls_res)

        cm = confusion_matrix(labels, preds, labels=label_names)
        plt.figure(figsize=(10, 10))
        ax = sn.heatmap(cm, vmax=10000, annot=True, xticklabels=label_names, yticklabels=label_names)
        plt.savefig(save_name + f'.png')

class CLS_model(nn.Module):
    def __init__(self, pretrain_url, label_names, resume=False, repr_layers=[12], reduce='mean', linear=False):
        super().__init__()
        pretrain_info = torch.load(pretrain_url)
        self.args = pretrain_info['args']
        self.label_names = label_names
        self.alphabet = Alphabet_RNA.RNA(coden_size=self.args.coden_size)
        self.resm = RESM(self.alphabet, num_layers=self.args.num_layers, embed_dim=self.args.embed_dim, attention_heads=20)
        if resume:
            self.resm.load_state_dict(pretrain_info['model'])
        self.reduce = reduce
        self.repr_layers = repr_layers
        self.linear = linear
        
        self.dense = nn.Linear(self.args.embed_dim, self.args.embed_dim)
        self.cls_norm = ESM1bLayerNorm(self.args.embed_dim)
        self.cls_head = nn.Linear(self.args.embed_dim, len(self.label_names))

    def forward(self, tokens):
        if self.linear:
            with torch.no_grad():
                repr = self.resm(tokens, self.repr_layers)['representations'][self.repr_layers[0]]
        else:
            repr = self.resm(tokens, self.repr_layers)['representations'][self.repr_layers[0]]
        
        if self.reduce == 'mean':
            repr = repr.mean(dim=1)
        else:
            repr = repr[:, 0]

        logits = self.cls_head(self.cls_norm(gelu(self.dense(repr))))
        return torch.softmax(logits, dim=-1)
    
    def run_test(self, testset, save_name=None):
        self.eval()
        testset = Seq_Dataset(testset)
        num_tasks = dist_misc.get_world_size()
        global_rank = dist_misc.get_rank()
        batch_index = np.arange(len(testset))[:, None].tolist()
        sampler_train = DistributedBatchSampler(
            testset, batch_index, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        
        data_loader_test = torch.utils.data.DataLoader(
            testset, collate_fn=MaskedBatchConverter(self.alphabet, self.args.truncation_seq_length), 
            num_workers=4, batch_sampler=sampler_train
        )
    
        labels = []
        preds = []
        with torch.no_grad():
            for batch in tqdm(data_loader_test):
                labels.append(batch[0][0])
                seq_str = batch[1][0]
                toks = torch.tensor(self.alphabet.encode(seq_str)).unsqueeze(0)
                if torch.cuda.is_available() and not self.args.nogpu:
                    toks = toks.to(device="cuda", non_blocking=True) 
                preds.append(self.forward(toks).argmax().cpu().item())
        classification_summary(labels, preds, self.label_names, save_name=save_name)

class Seq_Dataset(torch.utils.data.Dataset):
    def __init__(self, seq_data) -> None:
        super().__init__()
        self.seqs = seq_data.reset_index()
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, index):
        return self.seqs.loc[index, 'labels'], self.seqs.loc[index, 'seq'][0]

def finetune_cls(pretrain_url, dataset, label_names, args,
                    save_name=None,
                    resume=True, repr_layers=[12], reduce='cls', linear=False):
    # load training result
    model = CLS_model(pretrain_url, label_names, resume=resume, repr_layers=repr_layers, reduce=reduce, linear=linear)
    model.args = args # update fine tune args
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

    num_tasks = dist_misc.get_world_size()
    global_rank = dist_misc.get_rank()
    batch_index = np.arange(len(dataset))[:, None].tolist()
    sampler_train = DistributedBatchSampler(
        Seq_Dataset(dataset), batch_index, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    
    data_loader_train = torch.utils.data.DataLoader(
        Seq_Dataset(dataset), collate_fn=MaskedBatchConverter(model.alphabet, args.truncation_seq_length), 
        num_workers=4, batch_sampler=sampler_train
    )

    print(f"Fine-tuning with {len(dataset)} sequences")

    # prepare logger
    log_writer = None
    
    dist_misc.load_model(args=args, model_without_ddp=model, scheduler=training_scheduler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.epochs):
        if args.distributed:
            data_loader_train.batch_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            criterion,training_scheduler,  epoch,
            log_writer=None,
            args=args,
            finetune='cls'
        )
        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            dist_misc.save_model(
                args=args, model=model, model_without_ddp=model, scheduler=training_scheduler, epoch=save_name+str(epoch))

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
    return model

def finetune_reg():
    pass
