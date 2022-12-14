import torch
import math
import sys
import time
import datetime
import os
import json

from esm import FastaBatchedDataset, pretrained
from data import MaskedBatchConverter
from args import create_parser
from criterion import MaskedPredictionLoss
from schedular import Scheduler, LinearScheduler
import dist_misc
from train import train_one_epoch


def main(args):
    # prepare model
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.98), eps=10e-8, weight_decay=0.01)
    criterion = MaskedPredictionLoss()
    training_scheduler = Scheduler(optimizer, torch.cuda.amp.GradScaler(), LinearScheduler(args))
    
    return_contacts = "contacts" in args.include
    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]

    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")

    # prepare dataset
    dataset = FastaBatchedDataset.from_file(args.fasta_file)
    ### change to dist sampler ###
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    data_loader_train = torch.utils.data.DataLoader(
        dataset, collate_fn=MaskedBatchConverter(alphabet, args.truncation_seq_length), batch_sampler=batches
    )
    print(f"Read {args.fasta_file} with {len(dataset)} sequences")

    # prepare logger
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_writer = None

    
    dist_misc.load_model(args=args, model_without_ddp=model, scheduler=training_scheduler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            criterion,  epoch, training_scheduler,
            log_writer=None,
            args=args
        )
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            dist_misc.save_model(
                args=args, model=model, model_without_ddp=model, scheuler=training_scheduler, epoch=epoch)

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
    main(args)