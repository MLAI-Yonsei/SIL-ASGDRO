#!/bin/sh

CUDA=7
RHO=0.5
export CUDA_VISIBLE_DEVICES=$CUDA

for SEED in 0 1 2
do
python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.0001 --batch_size 16 --weight_decay 0.1 --model resnet50 --n_epochs 3 --gamma 0.1 --generalization_adjustment 1 --root_dir /data1/taero/data/celebA --log_dir /data1/taero/worst_sharp/subpopulation_shifts/logs/celeba_asam_best_${SEED}_no_robust --reweight_groups --asam
done
