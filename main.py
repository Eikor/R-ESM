import torch
import torch.utils.data
import time
import datetime
import os
import json
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5"
from esm import FastaBatchedDataset, pretrained
from data import MaskedBatchConverter, DistributedBatchSampler, Alphabet_RNA, RNADataset
from RESM import RESM
from args import create_parser
from criterion import MaskedPredictionLoss
from schedular import Scheduler, LinearScheduler
import dist_misc
from train import train_one_epoch
import numpy as np
import random

def main(args):
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
    alphabet = Alphabet_RNA.RNA(coden_size=args.coden_size)
    model = RESM(alphabet, num_layers=args.num_layers, embed_dim=args.embed_dim, attention_heads=args.num_heads)
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.98), eps=10e-8, weight_decay=0.01)
    criterion = MaskedPredictionLoss()
    training_scheduler = Scheduler(model, optimizer, torch.cuda.amp.GradScaler(), LinearScheduler(args))
    
    return_contacts = "contacts" in args.include
    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]

    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")

    # prepare dataset
    dataset = RNADataset.from_file(args.fasta_file)
    ### change to dist sampler ###
    batch_index = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=2)

    # ####
    # counter = []
    # RNAseg_toks = set('ATCGU')
    # ambigues = []
    # for item in dataset:
    #     counter.append(len(item[1]))
    #     if not set(item[1]) - RNAseg_toks:
    #         ambigues.append(item[0]) 
    # ####
    
    num_tasks = dist_misc.get_world_size()
    global_rank = dist_misc.get_rank()
    sampler_train = DistributedBatchSampler(
        dataset, batch_index, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset, collate_fn=MaskedBatchConverter(alphabet, args.truncation_seq_length), num_workers=4, batch_sampler=sampler_train
    )
    ####
    # counter1 = 0
    # counter2 = 0
    # for item in dataset:
    #     if 'N' in item[1]:
    #         counter1 += 1
    #         counter2 += item[1].count('N')

    ####

    print(f"Read {args.fasta_file} with {len(dataset)} sequences")

    # prepare logger
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_writer = None
    
    dist_misc.load_model(args=args, model_without_ddp=model, scheduler=training_scheduler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            # print(data_loader_train.batch_sampler)
            data_loader_train.batch_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            criterion,training_scheduler,  epoch,
            log_writer=None,
            args=args
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

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    args.distributed = False
    if 'contacts' in args.include:
        args.return_contacts = True
    else:
        args.return_contacts = False
    main(args)