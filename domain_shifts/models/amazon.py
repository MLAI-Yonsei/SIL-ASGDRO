import os
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from transformers import BertTokenizerFast
from transformers import DistilBertTokenizerFast
from wilds.common.data_loaders import get_eval_loader
from wilds.datasets.amazon_dataset import AmazonDataset
from .bert.bert import BertFeaturizer
from .bert.distilbert import DistilBertFeaturizer

from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import GeneralWilds_Batched_Dataset, seed_worker

# logging.set_verbosity_error()

MAX_TOKEN_LENGTH = 512
NUM_CLASSES = 5
NUM_DOMAINS = 1252 # reviewers in triain_set

def initialize_bert_transform(args):
    """Adapted from the Wilds library, available at: https://github.com/p-lambda/wilds"""
    if args.model == 'bert':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    else:
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    def transform(text):
        tokens = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            return_tensors='pt')
        if args.model == 'bert':
            x = torch.stack(
                (tokens['input_ids'],
                 tokens['attention_mask'],
                 tokens['token_type_ids']),
                dim=2)
        else:
            x = torch.stack(
                (tokens['input_ids'],
                 tokens['attention_mask']),
                dim=2)
        x = torch.squeeze(x, dim=0) # First shape dim is always 1
        return x
    return transform


class BertClassifier(BertForSequenceClassification):
    """Adapted from the Wilds library, available at: https://github.com/p-lambda/wilds"""
    def __init__(self, args):
        super().__init__(args)
        self.d_out = 5

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        token_type_ids = x[:, :, 2]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[0]
        return outputs

class Model(nn.Module):
    def __init__(self, args, weights):
        super(Model, self).__init__()
        self.num_classes = NUM_CLASSES
        if args.model == 'bert':
            featurizer = BertFeaturizer.from_pretrained("bert-base-uncased")
        elif args.model == 'distilbert':
            featurizer = DistilBertFeaturizer.from_pretrained("distilbert-base-uncased")
        else:
            raise NotImplementedError
        classifier = nn.Linear(featurizer.d_out, 5)
        self.model = nn.Sequential(featurizer, classifier)
        if weights is not None:
            self.load_state_dict(deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))

    @staticmethod
    def getDataLoaders(args, device):
        dataset = AmazonDataset(root_dir=args.data_dir, download=True)
        # get all train data
        transform = initialize_bert_transform(args)
        train_data = dataset.get_subset('train', transform=transform)
        # separate into subsets by distribution
        train_sets = GeneralWilds_Batched_Dataset(args, train_data, args.batch_size, domain_idx=0)
        # take subset of test and validation, making sure that only labels appeared in train
        # are included
        datasets = {}
        for split in dataset.split_dict:
            if split != 'train':
                datasets[split] = dataset.get_subset(split, transform=transform)

        # get the loaders
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': False, 'worker_init_fn': seed_worker} \
            if device.type == "cuda" else {}
        train_group_count = train_sets.domain_counts
        args.group_count = torch.Tensor(train_group_count)
        if args.balancing_sampling:
            # For GroupDRO and ASGDRO
            print("Sample instances of each group before balancing")
            assert train_sets.num_envs == 1252
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
        return self.model(x)

