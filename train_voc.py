import argparse
import random
import collections
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from trainer import TrainerFactory

import models.model as module_arch
import utils.metric as module_metric
import utils.lr_scheduler as module_lr_scheduler
import data_loader.data_loaders as module_data
from utils.parse_config import ConfigParser
from logger.logger import Logger
from utils.memory import memory_sampling_balanced

def main(config):
    ngpus_per_node = torch.cuda.device_count()
    if config['multiprocessing_distributed']:
        # Single node, mutliple GPUs
        config.config['orld_size'] = ngpus_per_node * config['world_size']
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # Rather using distributed, use DataParallel
        main_worker(None, ngpus_per_node, config)

def main_worker(gpu, ngpus_per_node, config):
    if config['multiprocessing_distributed']:
        config.config['rank'] = config['rank'] * ngpus_per_node + gpu

    dist.init_process_group(
        backend=config['dist_backend'], init_method=config['dist_url'],
        world_size=config['world_size'], rank=config['rank']
    )
    
    # Set looging
    rank = dist.get_rank()
    logger = Logger(config.log_dir, rank=rank)
    logger.set_logger(f'train(rank{rank})', verbosity=2)

    # fix random seeds for reproduce
    SEED = config['seed']
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Task information
    task_step = config['data_loader']['args']['task']['step']
    task_name = config['data_loader']['args']['task']['name']
    task_setting = config['data_loader']['args']['task']['setting']

    # Create Dataloader
    dataset = config.init_obj('data_loader', module_data, **{"dataset": config['dataset'], "method": config['name'], "seed":config['seed']})

    # Create old Model
    if task_step > 0:
        model_old = config.init_obj('arch', module_arch, **{"classes": dataset.get_num_classes_per_task(task_step - 1), "method": config['name']})
        if config['multiprocessing_distributed'] and (config['arch']['args']['norm_act'] == 'bn_sync'):
            model_old = nn.SyncBatchNorm.convert_sync_batchnorm(model_old)
    else:
        model_old = None

    # Memory pre-processing
    if (task_step > 0) and (config['data_loader']['args']['memory']['mem_size'] > 0):
        # sampling current memory (M_t)
        memory_sampling_balanced(
            config=config,
            model=model_old,
            prev_train_loader=dataset.get_old_train_loader(),
            task_info=(config['dataset'], task_setting, task_name, task_step),
            logger=logger, gpu=gpu
        )
        
        # init M_t
        dataset.get_memory(config)
        en_mem = True
    else:
        en_mem = False
    
    logger.info(f"{str(dataset)}")
    logger.info(f"{dataset.dataset_info()}")

    train_loader = dataset.get_train_loader(DistributedSampler(dataset.train_set)) if config['multiprocessing_distributed'] else dataset.get_train_loader(None)
    val_loader = dataset.get_val_loader()
    test_loader = dataset.get_test_loader()
    
    if en_mem:
        mem_loader = dataset.get_memory_loader(DistributedSampler(dataset.memory)) if config['multiprocessing_distributed'] else dataset.get_memory_loader(None)
    else:
        mem_loader = None
    

    new_classes, old_classes = dataset.get_task_labels()
    logger.info(f"Old Classes: {old_classes}")
    logger.info(f"New Classes: {new_classes}")

    # Create Model
    model = config.init_obj('arch', module_arch, **{"classes": dataset.get_num_classes_per_task(), "method": config['name']})
    
    # (NOTE) This does not exist in MiB, PLOP / only exists in SSUL, DKD
    # It sets update of batchnorm statistics in ResNet backbone as momentum update (not cumulative update)
    model._set_bn_momentum(model.backbone, momentum=0.01)

    # Convert BN to SyncBN for DDP
    if config['multiprocessing_distributed'] and (config['arch']['args']['norm_act'] == 'bn_sync'):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    logger.info(model)

    # At step > 0
    if task_step > 0:
        old_path = config.save_dir.parent / f"step_{task_step - 1}" / f"checkpoint-{config['data_loader']['args']['task']['setting']}-{config['name']}-epoch{config['trainer']['epochs']}-seed{config['seed']}.pth"
        model._load_pretrained_model(f'{old_path}')
        logger.info(f"Load weights from a previous step:{old_path}")
 
        # Load old model to use KD
        if model_old is not None:
            model_old._load_pretrained_model(f'{old_path}')
        
        # initialize new class classifier
        if config['name'] == 'DKD':
            if config['hyperparameter']['ac'] > 0:
                logger.info('** Proposed Initialization Technique using an Auxiliary Classifier**')
                model.init_novel_classifier_DKD()
            else:
                logger.info('** Random Initialization **')
        elif config['name'] == 'MiB' or config['name'] == 'PLOP':
            logger.info('** Proposed Initialization Technique using Initialization strategy of MiB**')
            model.init_novel_classifier_MiB()
        else:
            raise NotImplementedError
            
        test_only = False
    # At step 0
    else:
        if config['basemodel_dir'] != "None":
            base_path = os.path.join(config['basemodel_dir'], f"checkpoint-{config['data_loader']['args']['task']['setting']}-{config['name']}-epoch{config['trainer']['epochs']}-seed{config['seed']}.pth")
            
            try:
                model._load_pretrained_model(base_path)
                test_only = True
                logger.info(f"[STEP 0] Model loaded from {base_path} => Skip base task training")
            except:
                logger.info(f"[STEP 0] Model load failure : from {base_path} => Train from scratch")
                test_only = False
                
        else:
            test_only = False
            logger.info('[STEP 0] Train from scratch')           

    # Build optimizer
    if task_step > 0:
        if config['name'] == 'DKD':
            optimizer = config.init_obj(
                'optimizer',
                torch.optim,
                [{"params": model.get_backbone_params(), "weight_decay": 0},
                {"params": model.get_aspp_params(), "lr": config["optimizer"]["args"]["lr"] * 10, "weight_decay": 0},
                {"params": model.get_old_classifier_params(), "lr": config["optimizer"]["args"]["lr"] * 10, "weight_decay": 0},
                {"params": model.get_new_classifier_params(), "lr": config["optimizer"]["args"]["lr"] * 10}]
            )
        elif config['name'] == 'MiB':
            optimizer = config.init_obj('optimizer', torch.optim,
                [{"params": model.get_backbone_params(), "weight_decay": 0},
                {"params": model.get_aspp_params(), "lr": config["optimizer"]["args"]["lr"] * 10, "weight_decay": 0},
                {"params": model.get_classifier_params(), "lr": config["optimizer"]["args"]["lr"] * 10}]
            )
            
        elif config['name'] == 'PLOP' or 'FT' in config['name']:
            optimizer = config.init_obj('optimizer', torch.optim,
                [{"params": model.get_backbone_params()},
                {"params": model.get_aspp_params()},
                {"params": model.get_classifier_params()}]
            )
        else:
            raise ValueError("Not implemented Yet")
    else:
        if config['name'] == 'DKD': # same for SSUL
            optimizer = config.init_obj(
                'optimizer',
                torch.optim,
                [{"params": model.get_backbone_params()},
                {"params": model.get_aspp_params(), "lr": config["optimizer"]["args"]["lr"] * 10},
                {"params": model.get_classifier_params(), "lr": config["optimizer"]["args"]["lr"] * 10}]
            )
        elif config['name'] == 'MiB' or config['name'] == 'PLOP' or 'FT' in config['name']:
            optimizer = config.init_obj('optimizer', torch.optim,
                [{"params": model.get_backbone_params()},
                {"params": model.get_aspp_params()},
                {"params": model.get_classifier_params()}]
            )
        else:
            raise ValueError("Not implemented Yet")

    lr_scheduler = config.init_obj(
        'lr_scheduler',
        module_lr_scheduler,
        **{"optimizer": optimizer, "max_iters": config["trainer"]['epochs'] * len(train_loader)}
    )
        
    evaluator_val = config.init_obj(
        'evaluator',
        module_metric,
        *[dataset.n_classes, [0], new_classes]
    )

    old_classes, _ = dataset.get_task_labels(step=0)
    new_classes = []
    for i in range(1, task_step + 1):
        c, _ = dataset.get_task_labels(step=i)
        new_classes += c
        
    evaluator_test = config.init_obj(
        'evaluator',
        module_metric,
        *[dataset.n_classes, list(set(old_classes + [0])), new_classes]
    )

    trainer = TrainerFactory.get_trainer(task_step=task_step, method=config['name'], 
                                         model=model, model_old=model_old,
                                         config=config,
                                         task_info=dataset.task_info(),
                                         logger=logger, gpu=gpu,
                                         train_loader=train_loader, mem_loader=mem_loader, val_loader=val_loader, test_loader=test_loader,
                                         optimizer = optimizer,
                                         lr_scheduler=lr_scheduler,
                                         evaluator_val=evaluator_val, evaluator_test=evaluator_test,
                                        )
                                         

    torch.distributed.barrier()

    if test_only:
        trainer.test()
    else:
        trainer.train()
        trainer.test()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Class incremental Semantic Segmentation')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type action target', defaults=(None, float, None, None))
    options = [
        CustomArgs(['--multiprocessing_distributed'], action='store_true', target='multiprocessing_distributed'),
        CustomArgs(['--dist_url'], type=str, target='dist_url'),

        CustomArgs(['--name'], type=str, target='name'),
        CustomArgs(['--save_dir'], type=str, target='trainer;save_dir'),
        CustomArgs(['--basemodel_dir'], type=str, target='basemodel_dir'),
        CustomArgs(['--test'], action='store_true', target='test'),

        # General training
        CustomArgs(['--seed'], type=int, target='seed'),
        CustomArgs(['--ep', '--epochs'], type=int, target='trainer;epochs'),
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;train;batch_size'),

        # CISS task type
        CustomArgs(['--task_name'], type=str, target='data_loader;args;task;name'),
        CustomArgs(['--task_step'], type=int, target='data_loader;args;task;step'),
        CustomArgs(['--task_setting'], type=str, target='data_loader;args;task;setting'),
        
        # MiB 
        CustomArgs(['--unkd'], type=float, target='hyperparameter;unkd'),
        CustomArgs(['--mem_loss'], type=str, target='hyperparameter;mem_loss'),
        
        # DKD
        CustomArgs(['--pos_weight'], type=float, target='hyperparameter;pos_weight'),
        CustomArgs(['--mbce'], type=float, target='hyperparameter;mbce'),
        CustomArgs(['--kd'], type=float, target='hyperparameter;kd'),
        CustomArgs(['--dkd_pos'], type=float, target='hyperparameter;dkd_pos'),
        CustomArgs(['--dkd_neg'], type=float, target='hyperparameter;dkd_neg'),
        CustomArgs(['--ac'], type=float, target='hyperparameter;ac'),
        
        # PLOP
        CustomArgs(['--pod_factor'], type=float, target='hyperparameter;pod_factor'),
        CustomArgs(['--pod_options'], type=str, target='hyperparameter;pod_options'),

        # MEMORY
        CustomArgs(['--mem_size'], type=int, target='data_loader;args;memory;mem_size'),
        CustomArgs(['--mem_dir'], type=str, target='data_loader;args;memory;mem_dir'),
        
        # Pseudo-labeling
        CustomArgs(['--pseudo_labeling'], type=str, target='hyperparameter;pseudo_labeling'),
        
        CustomArgs(['--freeze_bn'], action='store_true', target='arch;args;freeze_all_bn'),
        
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
