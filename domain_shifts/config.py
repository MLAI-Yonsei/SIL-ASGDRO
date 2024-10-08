
dataset_defaults = {
    'fmow': {
        'epochs': 5,
        'batch_size': 32,
        'optimiser': 'Adam',
        'optimiser_args': {
            'lr': 1e-4,
            'weight_decay': 0,
            'amsgrad': True,
            'betas': (0.9, 0.999),
        },
        'pretrain_iters': 24000,
        'meta_lr': 0.01,
        'meta_steps': 5,
        'selection_metric': 'acc_worst_region',
        'reload_inner_optim': True,
        'print_iters': 2000
    },
    'camelyon': {
        'epochs': 5,
        'batch_size': 32,
        'optimiser': 'SGD',
        'optimiser_args': {
            'momentum': 0.9,
            'lr': 1e-4,
            'weight_decay': 0,
        },
        'pretrain_iters': 10000,
        'meta_lr': 0.01,
        'meta_steps': 3,
        'selection_metric': 'acc_avg',
        'reload_inner_optim': True,
        'print_iters': 2000,
    },
    'amazon': {
        'epochs': 3,
        'batch_size': 8,
        'optimiser': 'Adam',
        'model': 'distilbert',
        'optimiser_args': {
            'lr': 2e-6,
            'weight_decay': 0,
            'amsgrad': True,
            'betas': (0.9, 0.999),
        },
        'pretrain_iters': 31000,
        'meta_lr': 0.01,
        'meta_steps': 5,
        'selection_metric': '10th_percentile_acc',
        'reload_inner_optim': True,
        'print_iters': 10000
    },
    'civil': {
        'epochs': 5,
        'batch_size': 8,
        'optimiser': 'Adam',
        'model': 'distilbert', # TODO
        'optimiser_args': {
            'lr': 1e-5,
            'weight_decay': 0.0,
            'amsgrad': True,
            'betas': (0.9, 0.999),
        },
        'pretrain_iters': 20000,
        'meta_lr': 0.05,
        'meta_steps': 5,
        'selection_metric': 'acc_wg',
        'reload_inner_optim': True,
        'print_iters': 3000
    },
    'rxrx': {
        'epochs': 90,
        'batch_size': 75,
        'optimiser': 'Adam',
        'optimiser_args': {
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'amsgrad': True,
            'betas': (0.9, 0.999),
        },
        'pretrain_iters': 15000,
        'meta_lr': 0.01,
        'meta_steps': 10,
        'selection_metric': 'acc_avg',
        'reload_inner_optim': True,
        'print_iters': 2000,
        'scheduler': 'cosine_schedule_with_warmup',
        'scheduler_kwargs': {'num_warmup_steps': 5420},
    },
}

