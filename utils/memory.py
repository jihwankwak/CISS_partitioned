"""
We modified the code from DKD
"""

import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch import distributed

from torch.nn.parallel import DistributedDataParallel as DDP

from data_loader.task import get_task_labels

def _prepare_device(n_gpu_use, logger):
    """
    setup GPU device if available, move model into configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        logger.warning("Warning: There\'s no GPU available on this machine,"
                       "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                       "on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids
    
def memory_sampling_balanced(config, model, prev_train_loader, task_info, logger, gpu):
    
    if gpu is None:
        # setup GPU device if available, move model into configured device
        device, device_ids = _prepare_device(config['n_gpu'], logger)
    else:
        device = gpu
        device_ids = None

    if not torch.cuda.is_available():
        logger.info("using CPU, this will be slow")
    elif config['multiprocessing_distributed']:
        if gpu is not None:
            torch.cuda.set_device(device)
            model.to(device)
            # When using a single GPU per process and per
            # DDP, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False
        else:
            model.to(device)
            # DDP will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = DDP(model)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = nn.DataParallel(model, device_ids=device_ids)

    task_dataset, task_setting, task_name, task_step = task_info
    # (NOTE) get_task_labels differ between DKD and other methods
    # DKD : new_classes : [16], old_classes: [1, 2, .., 15]
    # others : new_classes : [16], old_classes: [0, 1, 2, .., 15]
    new_cls_formem, old_cls_inmem = get_task_labels(name=task_name, method=config['name'], step=task_step-1)
    # prev_num_classes = len(old_classes)  # DKD: 15 / others: 16

    # memory_json = f'./data/{task_dataset}/{task_setting}_{task_name}_memory.json'
    memory_size = config['data_loader']['args']['memory']['mem_size']
    
    # pseudo_mem = False if prev_mem_loader is None else True
    
    # logger.info(f"[Step: {task_step}] Memory candidates Pseudo-labeling : {pseudo_mem}")
    # old_classes = []
    if task_step > 1:
        memory_json_old = config.save_dir.parent / f'step_{task_step-1}' / f'cls_bal_memory_size{memory_size}.json'
        memory_json = config.save_dir.parent / f'step_{task_step}' / f'cls_bal_memory_size{memory_size}.json'
        
        with open(memory_json_old, "r") as json_file:
            memory_old = json.load(json_file)
            
        running_stats = memory_old["running_stats"].copy()
        memory_stats = memory_old["memory_stats"].copy()
        filenames_wtask = memory_old["filenames_wtask"].copy()
        
        # remove bg(0) classes for memory considerations
        if 0 in old_cls_inmem:
            old_cls_inmem.remove(0)
        if 0 in new_cls_formem:
            new_cls_formem.remove(0)
            
        assert sorted(old_cls_inmem) == sorted([int(s) for s in list(running_stats['img'].keys())])
        
        cum_cls = old_cls_inmem + new_cls_formem
        num_sampled = len(sum(list(filenames_wtask.values()), []))
         
        # update
        running_stats['img'].update({f"{cls}": 0 for cls in new_cls_formem})
        running_stats['pixel'].update({f"{cls}": 0 for cls in new_cls_formem})

        memory_stats['stats'].update({f"{cls}": 0 for cls in new_cls_formem})
        memory_stats['filenames'].update({f"{cls}":[ ] for cls in new_cls_formem})
    else:
        # task 1
        num_sampled = 0
        old_cls_inmem = None
        
        # bg label 0 is not considered for memory
        if 0 in new_cls_formem:
            new_cls_formem.remove(0) 
        cum_cls = new_cls_formem
        
        logger.info('======== [Step: {}] MEMORY SAMPLING START ========'.format(task_step))
        logger.info('New classes for memory({} step classes): {}'.format(task_step-1, new_cls_formem))
        logger.info('There is no existing classes in memory')
        logger.info('Current memory usage: {}'.format(num_sampled))
        memory = {}
        memory_json = config.save_dir.parent / f'step_{task_step}' / f'cls_bal_memory_size{memory_size}.json'
        
        # 1. Running_stats
        #   - Description : saves class stats of memory candidates
        #   - arguments
        #   1) class : class stats w.r.t imgs
        #   2) pixel : class stats w.r.t pixels
        running_stats = {}
        running_stats['img'] = {f"{cls}": 0 for cls in new_cls_formem}
        running_stats['pixel'] = {f"{cls}": 0 for cls in new_cls_formem}
        
        # 2. Memory_stats
        #   - Description : saves class stats of imgs saved in memory
        #                   Namely, it saves filenames of imgs saved in memory for given class
        #   - arguments
        #   1) single_class : single image can only be assigned to singe class
        #   -> sampling : img_bal
        #   2) multi_class : single image can be assigned to multiple classes
        #   -> sampling : mul_img_bal, mul_px_bal
        memory_stats = {}
        memory_stats['filenames'] = {f"{cls}":[] for cls in cum_cls}
        # NOTE: stats for each class is needed for pixel-wise balanced appoarches
        memory_stats['stats'] = {f"{cls}": 0 for cls in cum_cls}
    
    
    ### STEP 1. Get memory candidates (pseudo-labeling enabled, following https://github.com/cvlab-yonsei/DKD)  
    memory_candidates = []
    
    for batch_idx, data in enumerate(prev_train_loader):
        if (batch_idx % 100 == 0) and (distributed.get_rank() == 0):
            print(f"Loading candidate info... [{batch_idx} / {len(prev_train_loader)}]")
        
        gt_targets = data['label']
        
        if task_step > 1:
            with torch.no_grad():
                images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']

                outputs, _ = model(images, ret_intermediate=False)
                if config['name'] == 'DKD':
                    logit = torch.sigmoid(outputs).detach()
                    pred_scores, pred_labels = torch.max(logit[:, 1:], dim=1)
                    pred_labels += 1
                elif config['name'] == 'PLOP' or config['name'] == 'MiB':
                    logit = torch.softmax(outputs, 1).detach()
                    pred_scores, pred_labels = torch.max(logit, dim=1)
                else:
                    raise NotImplementedError
                    
                """ pseudo labeling """
                targets = torch.where((targets == 0) & (pred_scores >= 0.7), pred_labels.long(), targets.long())
        else:
            images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']
    
        # save running stats & class info for each images in prev_train_loader & prev_mem_loader
        
        for b in range(images.size(0)):
            d, w, h = images[b].size()
            img_name = img_names[b]
            target = targets[b]
            gt_labels = gt_targets[b]
            
            unique_labels = torch.unique(target).detach().cpu().numpy().tolist()
            unique_gt_labels = torch.unique(gt_labels).detach().cpu().numpy().tolist()
            
            # remove bg(0), 255 class if it exists
            if 0 in unique_labels:
                unique_labels.remove(0)
            if 255 in unique_labels:
                unique_labels.remove(255)
            if 0 in unique_gt_labels:
                unique_gt_labels.remove(0)
            if 255 in unique_gt_labels:
                unique_gt_labels.remove(255)
            # save running stats
            for lb in unique_labels:
                if lb not in cum_cls:
                    continue
                
                running_stats['pixel'][str(lb)] += int((target==lb).sum())
                running_stats['img'][str(lb)] += 1
                
            # save memory stats
            img_num = len(unique_labels)
            # px_num = [sum(b==torch.reshape(target, (-1,))).item() for b in unique_labels]
            
            px_num = [int((target==lb).sum()) for lb in unique_labels]

            img_name = img_name.split(' ')[0].split('/')[-1]
            # memory_candidates['new'].append([img_name, img_num, unique_labels, unique_gt_labels])
            memory_candidates.append([img_name, img_num, px_num, unique_labels, unique_gt_labels])
            
    logger.info(f"[Step: {task_step}] Memory candidates from prev train data : {len(memory_candidates)}")
    
    ### STEP 2. Sample memory (Algorithm from https://arxiv.org/pdf/2108.03613)
    
    memory_list = []
    torch.distributed.barrier()
    np.random.shuffle(memory_candidates)
    
    max_per_cls = {f"{cls}": int(memory_size/len(cum_cls)) for cls in cum_cls}

    for idx, add_mem in enumerate(memory_candidates):
        img_name, objs_num, objs_ratio, add_labels, _ = add_mem
        
        # find cls of img that has the least number of saved cls in memory
        running_stats_per_image = {key: memory_stats['stats'][key] for key in memory_stats['stats'].keys() if int(key) in add_labels}
        
        # NOTE : some image may be saved in the memory without previous labels (due to data transformation)
        if running_stats_per_image == {}:
            continue
        
        c_min = int(min(running_stats_per_image, key=running_stats_per_image.get))
        
        if memory_stats['stats'][str(c_min)] < max_per_cls[str(c_min)] or num_sampled < memory_size:
            if num_sampled >= memory_size:
                c_max = int(max(memory_stats['stats'], key=memory_stats['stats'].get))
                
                _ = memory_stats['filenames'][f"{c_max}"].pop(random.randrange(len(memory_stats['filenames'][f"{c_max}"])))
                memory_stats['stats'][str(c_max)] -= 1
                num_sampled-=1
            
            memory_stats['filenames'][f'{c_min}'].append(add_mem)
            memory_stats['stats'][f'{c_min}'] += 1
            
            num_sampled+=1

    # double check
    memory_list = sum(list(memory_stats['filenames'].values()), [])
    
    if len(memory_list) != memory_size and (torch.distributed.get_rank() == 0):
        raise ValueError(f'temp memory list({len(memory_list)}) and memory size({memory_size}) should be same')

    logger.info("[STEP {}] memory stats after memory generation: {}".format(task_step, memory_stats['stats']))
    logger.info("[STEP {}] Sampled memory size: {}".format(task_step, len(memory_list)))
    if len(memory_list) != memory_size:
        raise AssertionError(f'Collected memory size({len(memory_list)}) should be equal to memory size constraint({memory_size})')

    ### STEP 3: save memory with its saved task 
    filenames_wtask = {f'step_{step}':[] for step in range(task_step)}
    task_info = get_task_labels(name=task_name, method=config['name'])
    
    for mem in memory_list:
        img_name,_,_,_, gt_labels = mem
        
        step = labels_to_step(gt_labels, task_info)
        filenames_wtask[f'step_{step}'].append(img_name)

    memory = {"filenames_wtask": filenames_wtask,
                                            "running_stats": running_stats,
                                            "memory_stats": memory_stats
                                            }
    
    with open(memory_json, "w") as json_file:
        json.dump(memory, json_file)
        
    torch.distributed.barrier()

def labels_to_step(labels, task_info):
    
    if 0 in labels or '0' in labels:
        raise ValueError("0 should not exist")
    
    for step, task_labels in task_info.items():
        fil = lambda c: any(x in task_labels for x in c)
        if fil(labels):
            return step    