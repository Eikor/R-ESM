torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 main.py \
    /dros/common/working/jdxu/rnacentral_active_cleaned.fasta \
    ncRNA_650M_coden3_epoch100/ \
    --include contacts \
    --num_layers 33 \
    --embed_dim 1280 \
    --coden_size 3 \
    --epochs 50 \
    --accum_iter 16 \
    --toks_per_batch 4096 \
    --truncation_seq_length 4094
