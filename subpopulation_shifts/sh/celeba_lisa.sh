#!/bin/sh

CUDA=7
export CUDA_VISIBLE_DEVICES=$CUDA

for SEED in 0 1 2
do
python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.0001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 50 --root_dir /data1/taero/data/celebA --log_dir /data1/taero/worst_sharp/subpopulation_shifts/logs/celeba_lisa_best_${SEED} --gamma 0.1 --cut_mix --mix_ratio 0.5 --mix_alpha 2 --lisa_mix_up --save_best --save_last
done

