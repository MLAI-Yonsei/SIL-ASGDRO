#!/bin/sh

CUDA=7
LR=1e-3
WD=1e-4
export CUDA_VISIBLE_DEVICES=$CUDA

for SEED in 0 2 1
do
	python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --root_dir /data1/taero/data/cub --lr ${LR} --batch_size 16 --weight_decay ${WD} --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 0 --lisa_mix_up --mix_alpha 2 --mix_ratio 0.5 --save_best --save_last --log_dir /data1/taero/worst_sharp/subpopulation_shifts/logs/cub_lisa_lr_${LR}_wd_${WD}_seed_${SEED} --seed ${SEED}
done
