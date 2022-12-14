import torch

from esm import FastaBatchedDataset, pretrained
from data import MaskedBatchConverter
from args import create_parser
import math
import sys
from criterion import MaskedPredictionLoss
from schedular import Scheduler, LinearScheduler

### need transfer to distributed script ###
def main(args):
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.98), eps=10e-8, weight_decay=0.01)
    criterion = MaskedPredictionLoss()
    training_scheduler = Scheduler(optimizer, torch.cuda.amp.GradScaler(), LinearScheduler(args))

    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(args.fasta_file)
    
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=MaskedBatchConverter(alphabet, args.truncation_seq_length), batch_sampler=batches
    )
    
    print(f"Read {args.fasta_file} with {len(dataset)} sequences")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = "contacts" in args.include

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]
    
    for 

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)