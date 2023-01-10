# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main.py \
#  esm2_t12_35M_UR50D \
#  /root/esm/examples/data/some_proteins.fasta output/ \
#  --include contacts

CUDA_VISIBLE_DEVICES=2,3,4,5 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main.py \
 esm2_t30_150M_UR50D \
 /root/ensembl.fasta output/ \
 --include contacts