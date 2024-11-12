import os
import json
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from data_loader.task import get_task_labels
from data_loader.dataset import VOCSegmentationIncremental, VOCSegmentationIncrementalMemory


class IncrementalDataLoader():
    def __init__(self, task, train, val, test, num_workers, pin_memory, memory=None, dataset=None, method=None, seed=None):
        self.task = task
        self.train = train
        
        self.seed = seed
        self.step = task['step']
        self.name = task['name']
        self.method = method
        self.dataset = dataset
        
        self.classes_idx_new, self.classes_idx_old = get_task_labels(name=self.name, method=self.method, step=self.step)
        self.setting = task['setting']
        if method == 'DKD':
            self.n_classes = len(list(set(self.classes_idx_new + self.classes_idx_old))) + 1 
        elif method == 'MiB' or method == 'PLOP':
            self.n_classes = len(list(set(self.classes_idx_new + self.classes_idx_old)))
        else:
            raise ValueError("Not implemented Yet")     
        
        tr_idxs_path = Path(task['idxs_path']) / self.dataset / f"{self.setting}_{task['name']}_train_seed{self.seed}.npy"
        val_idxs_path = Path(task['idxs_path']) / self.dataset / f"{self.setting}_{task['name']}_val_seed{self.seed}.npy"
        test_idxs_path = Path(task['idxs_path']) / self.dataset / f"{self.setting}_{task['name']}_test_seed{self.seed}.npy"

        ## DATASET for train dataset
        self.train_set = VOCSegmentationIncremental(
            setting=self.setting,
            step=self.step,
            classes_idx_new=self.classes_idx_new,
            classes_idx_old=self.classes_idx_old,
            task_info=get_task_labels(name=self.name, method=self.method, step=None),
            idxs_path=tr_idxs_path,
            **train['args'],
        )

        if val['cross_val'] is True:
            train_len = int(0.8 * len(self.train_set))
            val_len = len(self.train_set) - train_len   # select 20% of train dataset
            self.train_set, self.val_set = random_split(self.train_set, [train_len, val_len])
        else:
            # Validatoin using validation set.
            self.val_set = VOCSegmentationIncremental(
                val=True,
                setting=self.setting,
                step=self.step,
                classes_idx_new=self.classes_idx_new,
                classes_idx_old=self.classes_idx_old,
                task_info=get_task_labels(name=self.name, method=self.method, step=None),
                idxs_path=val_idxs_path,
                **val['args'],
            )

        self.test_set = VOCSegmentationIncremental(
            test=True,
            setting=self.setting,
            step=self.step,
            classes_idx_new=self.classes_idx_new,
            classes_idx_old=self.classes_idx_old,
            task_info=get_task_labels(name=self.name, method=self.method, step=None),
            idxs_path=test_idxs_path,
            **test['args'],
        )
        
        # init memory data
        self.memory = None
        
        # data loaded for memory sampling
        if self.step > 0 and (memory is not None) and memory['mem_size'] != 0:
            classes_idx_new, classes_idx_old = get_task_labels(name=self.name, method=self.method, step=self.step - 1)
            # NOTE : memory candidates should not be cropped when considered => Val:True
            self.prev_train_set = VOCSegmentationIncremental(
                cand=True,
                setting=self.setting,
                step=self.step-1,
                classes_idx_new=classes_idx_new,
                classes_idx_old=classes_idx_old,
                task_info=get_task_labels(name=self.name, method=self.method, step=None),
                idxs_path=Path(task['idxs_path']) / "voc" / f"{self.setting}_{task['name']}_train_seed{self.seed}.npy",
                **train['args'],
            )

        self.init_train_kwargs = {'num_workers': num_workers, "pin_memory": pin_memory, "batch_size": train["batch_size"]}
        self.init_val_kwargs = {'num_workers': num_workers, "pin_memory": pin_memory, "batch_size": val["batch_size"]}
        self.init_test_kwargs = {'num_workers': num_workers, "pin_memory": pin_memory, "batch_size": test["batch_size"]}


    def get_memory(self, config, concat=True):
        
        memory_size = config['data_loader']['args']['memory']['mem_size']
            
        if self.step > 0:               
            self.memory = VOCSegmentationIncrementalMemory(
                setting=self.setting,
                name=config['data_loader']['args']['task']['name'],
                method=config['name'],
                step=self.step,
                classes_idx_new=self.classes_idx_new,
                classes_idx_old=self.classes_idx_old,
                idxs_path=config.save_dir.parent / f'step_{self.step}' / f"cls_bal_memory_size{memory_size}.json",            
                **self.train['args'],
            )

    def get_train_loader(self, sampler=None):
        return DataLoader(self.train_set, **self.init_train_kwargs,
                          drop_last=True, sampler=sampler, shuffle=(sampler is None),)

    def get_val_loader(self, sampler=None):
        return DataLoader(self.val_set, **self.init_val_kwargs,
                          sampler=sampler, shuffle=False,)

    def get_test_loader(self, sampler=None):
        return DataLoader(self.test_set, **self.init_test_kwargs,
                          sampler=sampler, shuffle=False,)
    
    def get_memory_loader(self, sampler=None):
        return DataLoader(self.memory, **self.init_train_kwargs,
                          drop_last=True, sampler=sampler, shuffle=(sampler is None),)

    def get_old_train_loader(self, sampler=None):
        return DataLoader(self.prev_train_set, **self.init_train_kwargs,
                          drop_last=False, sampler=sampler, shuffle=False,)

    def __str__(self):
        return f"{self.setting} / {self.name} / step: {self.step}"

    def dataset_info(self):
        if self.memory is not None:
            return f"The number of datasets: {len(self.train_set)}+(M){len(self.memory)} / {len(self.val_set)} / {len(self.test_set)}"
        else:
            return f"The number of datasets: {len(self.train_set)} / {len(self.val_set)} / {len(self.test_set)}"

    def task_info(self):
        return {"setting": self.setting, "name": self.name, "step": self.step,
                "old_class": self.classes_idx_old, "new_class": self.classes_idx_new, "n_classes": self.n_classes, "seed":self.seed}

    def get_num_classes_per_task(self, step=None):
        task_dict = get_task_labels(name=self.name, method=self.method, step=None)
        if step == None:
            step = self.step
        assert step in task_dict.keys(), f"You should provide a valid step! [{step} is out of range]"
        
        classes = [len(task_dict[s]) for s in range(step + 1)]
        
        return classes
    
    def get_task_labels(self, step=None):
        if step is None:
            step = self.step
        return get_task_labels(name=self.name, method=self.method, step=step)