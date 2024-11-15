# Sufficient Invariant Learning for Distribution Shift (For Experiments in Table 2 )

The code is based on [LISA](https://github.com/huaxiuyao/LISA) which is also based on the code in [groupDRO](https://github.com/kohpangwei/group_DRO).

<!--Specify Citation Later -->

## Prerequisites
- python
- matplotlib
- numpy 
- pandas
- pillow
- pytorch
- pytorch_transformers
- torchvision 
- torchaudio 
- tqdm 
- wilds 
- transformers 
- ipdb 

## Datasets and Scripts

For subpopulation shifts problems, the part of datasets are listed as follows:

#### CMNIST
This dataset is constructed from MNIST. It will be automatically downloaded when running the following script:
```
python run_expt.py -s confounder -d CMNIST -t 0-4 -c isred --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 1 --gamma 0.1 --generalization_adjustment 0 --reweight_groups --asgdro --robust --log_dir [log_dir] --root_dir [root_dir] --rho 0.8
```

#### Waterbirds
According to [group_DRO](https://github.com/kohpangwei/group_DRO),                                            

This code expects the following files/folders in the `[root_dir]/cub` directory:

- `data/waterbird_complete95_forest2water2/`

You can download waterbrids dataset  [here](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz).

The command to run ASGDRO on Waterbirds is:
```
python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --root_dir [root_dir] --lr 1e-5 --batch_size 64  --weight_decay 0.1 --model resnet50 --n_epochs 300 --gamma 0.1 --generalization_adjustment 2 --reweight_groups --log_dir [log_dir] --rho 0.05 --asgdro
```

#### CelebA
According to [group_DRO](https://github.com/kohpangwei/group_DRO),

This code expects the following files/folders in the `[root_dir]/celebA` directory:

- `data/list_eval_partition.csv`
- `data/list_attr_celeba.csv`
- `data/img_align_celeba/`

You can download these dataset files from [this Kaggle link](https://www.kaggle.com/jessicali9530/celeba-dataset). The version of the CelebA dataset that we use in this paper (with the (hair, gender) groups) can also be accessed through the [WILDS package](https://github.com/p-lambda/wilds), which will automatically download the dataset.

The command to run ASGDRO on CelebA is:
```
python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.0001 --batch_size 16 --weight_decay 0.1 --model resnet50 --n_epochs 50 --gamma 0.1 --generalization_adjustment 1 --root_dir [root_dir] --log_dir [log_dir] --reweight_groups --robust --asgdro
```

