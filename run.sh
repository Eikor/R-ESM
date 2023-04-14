# torchrun \
#     --standalone \
#     --nnodes=1 \
#     --nproc_per_node=8 main.py \
#     /dros/common/working/jdxu/rnacentral_active_cleaned.fasta \
#     ncRNA_15B_coden3_epoch15/ \
#     --include contacts \
#     --num_layers 48 \
    # --num_heads 40 \
#     --embed_dim 5120 \
#     --coden_size 3 \
#     --warmup_epochs 1 \
#     --epochs 15 \
#     --lr 1e-3 \
#     --min_lr 1e-5 \
#     --accum_iter 384 \
#     --toks_per_batch 1026 \
#     --truncation_seq_length 1024

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 main.py \
    /dros/common/working/jdxu/rnacentral_active_cleaned.fasta \
    ncRNA_3B_coden3_epoch15/ \
    --include contacts \
    --num_layers 36 \
    --num_heads 40 \
    --embed_dim 2560 \
    --coden_size 3 \
    --warmup_epochs 1 \
    --epochs 15 \
    --lr 1e-3 \
    --min_lr 1e-5 \
    --accum_iter 384 \
    --toks_per_batch 1026 \
    --truncation_seq_length 1024
