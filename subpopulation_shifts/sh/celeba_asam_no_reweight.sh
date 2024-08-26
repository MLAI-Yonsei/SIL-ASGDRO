#!/bin/sh

CUDA=7
RHO=0.5
export CUDA_VISIBLE_DEVICES=$CUDA

for SEED in 1 2
do
python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.0001 --batch_size 16 --weight_decay 0.1 --model resnet50 --n_epochs 25 --root_dir /data1/taero/data/celebA --log_dir /data1/taero/worst_sharp/subpopulation_shifts/logs/celeba_asam_best_${SEED}_no_reweight_full_epoch --asam --save_best --save_last --seed ${SEED}
done
