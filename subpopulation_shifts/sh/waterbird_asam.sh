#!/bin/sh

CUDA=6
RHO=0.5
LR=1e-3
WD=1e-4
SEED=0
export CUDA_VISIBLE_DEVICES=$CUDA

for RHO in 0.8 0.5
do
	python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --root_dir /data1/taero/data/cub --lr ${LR} --batch_size 16 --weight_decay ${WD} --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 2 --reweight_groups --log_dir /data1/taero/worst_sharp/subpopulation_shifts/logs/cub_asam_rho_${RHO}_balance_lr_${LR}_wd_${WD}_seed_${SEED} --rho ${RHO} --asam --save_best --save_last 
done
