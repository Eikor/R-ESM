OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main.py \
    /dros/common/working/jdxu/rnacentral_active_cleaned.fasta ncRNA_650M_coden3_epoch100/ \
    --include contacts \
    --num_layers 33 \
    --embed_dim 1280 \
    --coden_size 1