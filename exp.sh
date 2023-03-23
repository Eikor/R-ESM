# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main.py \
#  esm2_t12_35M_UR50D \
#  /root/esm/examples/data/some_proteins.fasta output/ \
#  --include contacts

# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     /junde/snRNA.fasta snRNA_150M_100epoch/ \
#     --include contacts \
#     --num_layers 30 \
#     --embed_dim 640

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    /junde/snRNA.fasta snRNA_35M_coden2_100epoch/ \
    --include contacts \
    --num_layers 12 \
    --embed_dim 480 \
    --coden_size 2