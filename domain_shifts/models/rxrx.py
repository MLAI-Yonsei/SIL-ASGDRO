import numpy as np
import os
from copy import deepcopy

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from wilds.common.data_loaders import get_eval_loader
from wilds.datasets.rxrx1_dataset import RxRx1Dataset

from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import GeneralWilds_Batched_Dataset, seed_worker

IMG_HEIGHT = 224
NUM_CLASSES = 1139
NUM_DOMAINS = 33 # in train

def initialize_rxrx1_transform(is_training):
    def standardize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2))
        std[std == 0.] = 1.
        return TF.normalize(x, mean, std)
    t_standardize = transforms.Lambda(lambda x: standardize(x))

    angles = [0, 90, 180, 270]
    def random_rotation(x: torch.Tensor) -> torch.Tensor:
        angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
        if angle > 0:
            x = TF.rotate(x, angle)
        return x
    t_random_rotation = transforms.Lambda(lambda x: random_rotation(x))

    if is_training:
        transforms_ls = [
            t_random_rotation,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            t_standardize,
        ]
    else:
        transforms_ls = [
            transforms.ToTensor(),
            t_standardize,
        ]
    transform = transforms.Compose(transforms_ls)
    return transform

class Model(nn.Module):
    def __init__(self, args, weights):
        super(Model, self).__init__()
        self.num_classes = NUM_CLASSES
        resnet = resnet50(pretrained=True)
        self.enc = nn.Sequential(*list(resnet.children())[:-1]) # remove fc layer
        self.fc = nn.Linear(2048, self.num_classes)
        if weights is not None:
            self.load_state_dict(deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))


    @staticmethod
    def getDataLoaders(args, device):
        dataset = RxRx1Dataset(root_dir=args.data_dir, download=True)

        # initialize transform
        train_transform = initialize_rxrx1_transform(is_training=True)
        eval_transform = initialize_rxrx1_transform(is_training=False)

        # get all train data
        train_data = dataset.get_subset('train', transform=train_transform)

        # separate into subsets by distribution
        train_sets = GeneralWilds_Batched_Dataset(args, train_data, args.batch_size, domain_idx=1)
        # take subset of test and validation, making sure that only labels appeared in train
        # are included
        datasets = {}
        for split in dataset.split_dict:
            if split != 'train':
                datasets[split] = dataset.get_subset(split, transform=eval_transform)

        # get the loaders
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': False, 'worker_init_fn': seed_worker} \
            if device.type == "cuda" else {}
        train_group_count = train_sets.domain_counts
        args.group_count = torch.Tensor(train_group_count)
        if args.balancing_sampling:
            # For GroupDRO and ASGDRO
            print("Sample instances of each group before balancing")
            assert train_sets.num_envs == 33
            group_weights = 1/torch.tensor(train_group_count)
            domain_idx = [train_sets.domain2idx[i.item()] for i in train_sets.domains] ## TODO: Caution
            weights = group_weights[domain_idx]
            for i in range(len(train_group_count)):
                print(f"Number of Train Instances Domain {i} before balancing: {train_group_count[i]}")
            print(f"Number of Total Instances: {sum(train_group_count)}")
            sampler = WeightedRandomSampler(weights, len(train_sets), replacement=True)
            train_loaders = DataLoader(train_sets, batch_size=args.batch_size, sampler=sampler, **kwargs)
        else:
            train_loaders = DataLoader(train_sets, batch_size=args.batch_size, shuffle=True, **kwargs)

        tv_loaders = {}
        for split, dataset in datasets.items():
            tv_loaders[split] = get_eval_loader('standard', dataset, batch_size=args.eval_batch_size, **kwargs)
        return train_loaders, tv_loaders

    def forward(self, x):
        # x = x.expand(-1, 3, -1, -1)  # reshape MNIST from 1x32x32 => 3x32x32
        if len(x.shape) == 3:
            x.unsqueeze_(0)
        e = self.enc(x)
        return self.fc(e.squeeze(-1).squeeze(-1))
