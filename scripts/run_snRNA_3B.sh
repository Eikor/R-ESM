torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 main.py \
    /dros/common/working/jdxu/snRNA.fasta \
    snRNA_3B_coden3_epoch15/ \
    --include contacts \
    --save_interval 1 \
    --num_layers 36 \
    --num_heads 40 \
    --embed_dim 2560 \
    --coden_size 3 \
    --warmup_epochs 10 \
    --epochs 100 \
    --lr 1e-3 \
    --min_lr 1e-5 \
    --accum_iter 4 \
    --toks_per_batch 1024 \
    --truncation_seq_length 1024 \
    --fsdp \
    --bf16 \
    --static_batch