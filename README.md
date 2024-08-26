# Sharpness-aware Worst-case Optimization for Sufficient Invariant Learning

The code implementation for the paper, Sharpness-aware Worst-case Optimization for Sufficient Invariant Learning.
The code is based on [LISA](https://github.com/huaxiuyao/LISA) which is also based on the code in [groupDRO](https://github.com/kohpangwei/group_DRO) and [fish](https://github.com/YugeTen/fish).

<!--Specify Citation Later -->

## Abstract

## Prerequisites
- python 3.8.16
- matplotlib 3.7.1
- numpy 1.24.4
- pandas 2.0.0
- pillow 9.5.0
- pytorch 1.13.1
- pytorch_transformers 1.2.0
- torchvision 0.14.1
- torchaudio 0.13.1
- tqdm 4.65.5
- wilds 2.0.0
- transformers 4.27.4
- ipdb 0.13.13 
- pexpect 4.9.0
- traitlets 5.14.3
- psutil 6.0.0

## Datasets and Scripts
<!--
### Subpopulation shifts and MetaShifts
To run the code, you need to first enter the directory: `cd subpopulation_shifts`. Then change the `root_dir` variable in `./data/data.py` if you need to put the dataset elsewhere other than `./data/`. 

For subpopulation shifts problems, the datasets are listed as follows:


#### MetaShifts
The dataset can be downloaded [[here]](https://drive.google.com/file/d/1Fr2HxUOL3_QUDHU5B3MMH7dgFu_u_gJ_/view?usp=sharing). You should put it under the directory `data`. The running scripts for 4 dataset with different distances are as follows:
```
python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300 --gamma 0.1 --dog_group 1 --lisa_mix_up --mix_alpha 2 --cut_mix --group_by_label
python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300 --gamma 0.1 --dog_group 2 --lisa_mix_up --mix_alpha 2 --cut_mix --group_by_label
python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300 --gamma 0.1 --dog_group 3 --lisa_mix_up --mix_alpha 2 --cut_mix --group_by_label
python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300 --gamma 0.1 --dog_group 4 --lisa_mix_up --mix_alpha 2 --cut_mix --group_by_label
```

#### CMNIST
This dataset is constructed from MNIST. It will be automatically downloaded when running the following script:
```
python run_expt.py -s confounder -d CMNIST -t 0-4 -c isred --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 0 --lisa_mix_up --mix_ratio 0.5`
```

#### CelebA
This dataset can be downloaded via the link in the repo [group_DRO](https://github.com/kohpangwei/group_DRO). 

The command to run LISA on CelebA is:
```
python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.0001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 50 --gamma 0.1 --generalization_adjustment 0 --lisa_mix_up --mix_alpha 2 --mix_ratio 0.5 --cut_mix`
```

#### Waterbirds
This dataset can be downloaded via the link in the repo [group_DRO](https://github.com/kohpangwei/group_DRO). 

The command to run LISA on Waterbirds is:
```
python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 0 --lisa_mix_up --mix_alpha 2 --mix_ratio 0.5`
```

#### MultiNLI
-->



### Domain Shifts
To run the code, you need to first enter the directory: `cd domain_shifts`.
When you run your command, please specify your own $DATA_PATH with "--data-dir".

The datasets will be automatically downloaded when running the scripts provided below. 

#### CivilComments

##### GroupDRO
```
python main.py --dataset civil --algorithm groupdro --data-dir /$DATA_PATH/CivilComments
```

##### ASGDRO
```
python main.py --dataset civil --algorithm asgdro --data-dir /$DATA_PATH/CivilComments 
```

###### MultiNLI
```
python main.py --dataset civil --algorithm asgdro --data-dir /$DATA_PATH/multinli
```

<!--
#### Camelyon17
```
python main.py --dataset camelyon --algorithm asgdro --data-dir /$DATA_PATH/Cameyon17 
```

#### FMoW
```
python main.py --dataset fmow --algorithm asgdro --data-dir /$DATA_PATH/FMoW 
```

#### RxRx1
```
python main.py --dataset rxrx --algorithm asgdro --data-dir /$DATA_PATH/RxRx1 
```

#### Amazon
```
python main.py --dataset amazon --algorithm asgdro --data-dir /$DATA_PATH/Amazon 
```

#### CivilComments
```
python main.py --dataset civil --algorithm asgdro --data-dir /$DATA_PATH/CivilComments
```
-->


