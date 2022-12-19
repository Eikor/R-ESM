# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main.py \
#  esm2_t12_35M_UR50D \
#  /root/esm/examples/data/some_proteins.fasta output/ \
#  --include contacts

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main.py \
 esm2_t12_35M_UR50D \
 /HOME/dataset/uniref50.fasta output/ \
 --include contacts