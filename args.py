import json
import argparse
import pathlib

def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
    )

    parser.add_argument(
        "fasta_file",
        type=pathlib.Path,
        help="FASTA file on which to extract representations",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="output directory for extracted representations",
    )

    parser.add_argument(
        "--coden_size",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--embed_dim",
        type=int,
        default=480,
    )

    parser.add_argument("--toks_per_batch", type=int, default=1024, help="maximum batch size")
    parser.add_argument(
        "--truncation_seq_length",
        type=int,
        default=1022,
        help="truncate sequences longer than the given value",
    )

    parser.add_argument(
        "--repr_layers",
        type=int,
        default=[-1],
        nargs="+",
        help="layers indices from which to extract representations (0 to num_layers, inclusive)",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        choices=["mean", "per_tok", "bos", "contacts"],
        help="specify which representations to return",
        required=True,
    )

    ## training configs
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
    )
    
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        help='each epoch has 1k iters'
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help='each epoch has 1k iters'
    )

    parser.add_argument(
        "--accum_iter",
        type=int,
        default=32,
        help="accum grad to mimic large batch size",
    )

    parser.add_argument(
        "--lr",
        type=int,
        default=1e-3,
        help="accum grad to mimic large batch size",
    )
    
    parser.add_argument(
        "--min_lr",
        type=int,
        default=1e-6,
        help="accum grad to mimic large batch size",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="accum grad to mimic large batch size",
    )

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser