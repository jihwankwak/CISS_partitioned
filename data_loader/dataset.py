import os
import pathlib
import json
import numpy as np
import torchvision as tv
import json

from PIL import Image
from torch import distributed
from data_loader import DATASETS_IMG_DIRS
from data_loader import custom_transforms as tr
from base.base_dataset import BaseDataset, lbl_contains_any, lbl_contains_all
from data_loader.task import get_classes_per_task

class VOCSegmentationIncremental(BaseDataset):
    """
    PascalVoc dataset
    """
    def __init__(
        self,
        test=False, val=False, cand=False, setting='overlap', step=0,  classes_idx_new=[], classes_idx_old=[], task_info = None,
        idxs_path=None, seed=0, transform=True, transform_args={}, masking_value=0, 
    ):
        # Experiment setup
        if setting in ['disjoint', 'overlap', 'partitioned']:
            pass
        else:
            raise ValueError('Wrong setting entered! Please use one of disjoint, overlap, partitioned')

        super().__init__(
            transform_args=transform_args,
            base_dir=pathlib.Path(DATASETS_IMG_DIRS['voc']),
            transform=transform,
        )
        self.setting = setting
        self.step = step
        self.classes_idx_old = classes_idx_old
        self.classes_idx_new = classes_idx_new
        self.seed = seed

        self.test = test
        self.val = val
        self.train = not (self.test or self.val)
        # self.cand = True during memory sampling (to eliminate crop randomness)
        self.cand = cand
        
        self.masking_value = masking_value

        if self.train:
            self.split = 'train_aug'
        else:
            self.split = 'val'

        if 'aug' not in self.split:
            # val, test
            self._image_dir = self._base_dir / "JPEGImages"       # image
            self._cat_dir = self._base_dir / "SegmentationClass"  # target
        else:
            # train
            self._image_dir = self._base_dir
            self._cat_dir = self._base_dir
        _splits_dir = self._base_dir / "ImageSets" / "Segmentation"

        self.im_ids = []
        self.categories = []

        if (idxs_path is not None) and (os.path.exists(idxs_path)):
            # Use predefined dataset per tasks
            with open(idxs_path, 'r') as json_file:
                file_names = json.load(json_file)
        else:
            data_path = '/'.join(str(idxs_path).split('/')[:-1])
            
            if not os.path.exists(data_path):
                os.makedirs(data_path, exist_ok=True)
            
            # Define dataset per tasks & save 
            if distributed.get_rank() == 0:
                print(f"Building up incremental scenario")
                
            lines = (_splits_dir / f"{self.split}.txt").read_text().splitlines()
            
            if self.setting == 'partitioned':
                file_names = self.generate_cil_partitioned(lines, task_info)
            elif self.setting == 'overlap' or self.setting == 'disjoint':
                file_names = self.generate_cil_others(lines, task_info)
            else:
                raise NotImplementedError("Not implemented Yet")

            with open(idxs_path, "w") as json_file:
                json.dump(file_names, json_file)
                
        # Extract imgs used in current step
        self.im_ids = file_names['imgs_per_task'][str(self.step)]
        
        for x in self.im_ids:
            if 'aug' not in self.split:
                _image = self._image_dir / f"{x}.jpg"
                _cat = self._cat_dir / f"{x}.png"
            else:
                _image = self._image_dir / x.split()[0][1:]
                _cat = self._cat_dir / x.split()[1][1:]

            assert _image.is_file(), _image
            assert _image.is_file(), _cat
            
            self.images.append(_image)
            self.categories.append(_cat)

        assert len(self.images) == len(self.categories)
        assert len(self.im_ids) != 0

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        
        # print(_target)
        sample = {"image": _img, "label": _target}

        if self.transform:
            # train
            if self.split in ["trainval_aug", "trainval", "train_aug", "train"]:
                if not self.cand:
                    sample['image'], sample['label'] = self.transform_tr(sample)
                else:
                    # NOTE: for memory candidates, we disabled the data tranform (random crop. flip, ..)
                    #       (For implementation ease, we use center crop with 512)
                    sample['image'], sample['label'] = self.transform_val(sample)
            # val
            elif self.split in ["val_aug", "val"]:
                sample['image'], sample['label'] = self.transform_val(sample)
        else:
            # test
            sample['image'], sample['label'] = self.transform_test(sample)

        # Target masking
        sample['gt_label'] = self.transform_target_masking_all(sample['label'].clone()) 
        sample['label'] = self.transform_target_masking(sample['label'])
        
        # sample["image_name"] = str(self.images[index])
        sample["image_name"] = str(self.im_ids[index])
        return sample

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _target = Image.open(self.categories[index])
        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.RandomResizedCrop(
                    self.transform_args['crop_size'],
                    (0.5, 2.0)
                ),
                tr.RandomHorizontalFlip(),
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_val(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.Resize(size=self.transform_args['crop_size']),
                tr.CenterCrop(self.transform_args['crop_size']),
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_test(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_target_masking(self, target):
        if self.test:
            # Masking future class object 
            label_list = list(set(self.classes_idx_old + self.classes_idx_new + [0, 255]))
            target_transform = tv.transforms.Lambda(
                lambda t: t.apply_(lambda x: x if x in label_list else self.masking_value)
            )
            return target_transform(target)

        else:  # Train or Validation
            # Masking except current classes
            if self.masking_value is None:
                return target
            if self.setting in ['disjoint', 'overlap', 'partitioned']:
                label_list = list(set(self.classes_idx_new + [0, 255]))
                target_transform = tv.transforms.Lambda(
                    lambda t: t.apply_(lambda x: x if x in label_list else self.masking_value)
                )
                return target_transform(target)
            else:
                raise ValueError("Not implemented Yet")
            
    def transform_target_masking_all(self, target):
        label_list = list(set(self.classes_idx_old + self.classes_idx_new + [0, 255]))
        target_transform = tv.transforms.Lambda(
            lambda t: t.apply_(lambda x: x if x in label_list else self.masking_value)
        )
        return target_transform(target)
        
    def generate_cil_others(self, lines, task_info):
        
        file_names = {}
        file_names['imgs_per_task'] = {f"{step}": [] for step,_ in task_info.items()}
        
        for step, cls_list in task_info.items():
            
            classes_idx_new = cls_list.copy()
            classes_idx_old = sum([cls_list.copy() for step_it, cls_list in task_info.items() if step_it < step], [])
            
            # remove bg class if it exists
            if 0 in classes_idx_new:
                classes_idx_new.remove(0)
            if 0 in classes_idx_old:
                classes_idx_old.remove(0)
        
            for ii, line in enumerate(lines):
                if (ii % 1000 == 0) and (distributed.get_rank() == 0):
                    print(f"[{ii} / {len(lines)}]")
                    
                if 'aug' not in self.split: # val
                    _image = self._image_dir / f"{line}.jpg"
                    _cat = self._cat_dir / f"{line}.png"
                else: # train
                    _image = self._image_dir / line.split()[0][1:]
                    _cat = self._cat_dir / line.split()[1][1:]
                assert _image.is_file(), _image
                assert _cat.is_file(), _cat

                cat = Image.open(_cat)
                cat = np.array(cat, dtype=np.uint8)
                
                if (self.train or self.val):
                    # Remove the sample if the g.t mask does not contain new class
                    if not lbl_contains_any(cat, classes_idx_new):
                        continue
                    # Unique set
                    # : Remove the sample if the g.t mask contains any other labels that not appeared yet.
                    if self.train and self.setting == 'disjoint':
                        if not lbl_contains_all(cat, list(set(classes_idx_old + classes_idx_new + [0, 255]))):
                            continue
                else:  # Test
                    if not lbl_contains_any(cat, list(set(classes_idx_old + classes_idx_new))):
                        continue
                    
                file_names['imgs_per_task'][str(step)].append(line)

        return file_names
    
    def generate_cil_partitioned(self, lines, task_info):
        
        cls_list = sum([cls_list.copy() for _, cls_list in task_info.items()], [])
            
        # remove bg class if it exists
        if 0 in cls_list:
            cls_list.remove(0)
        
        file_names = {}
        file_names['imgs_per_task'] = {f"{step}": [] for step,_ in task_info.items()}
        file_names['imgs_per_cls'] = {f"{cls}": [] for cls in sorted(cls_list)}
        
        # Train, val
        if (self.train or self.val):
            # STEP 1. assign imgs to one of classes it contains
            for ii, line in enumerate(lines):
                if (ii % 1000 == 0) and (distributed.get_rank() == 0):
                    print(f"[{ii} / {len(lines)}]")
                    
                if 'aug' not in self.split: # val
                    _image = self._image_dir / f"{line}.jpg"
                    _cat = self._cat_dir / f"{line}.png"
                else: # train
                    _image = self._image_dir / line.split()[0][1:]
                    _cat = self._cat_dir / line.split()[1][1:]
                assert _image.is_file(), _image
                assert _cat.is_file(), _cat

                cat = Image.open(_cat)
                cat = np.array(cat, dtype=np.uint8)
        
                unique_lbs = np.unique(cat).tolist()
                
                if 0 in unique_lbs:
                    unique_lbs.remove(0)
                if 255 in unique_lbs:
                    unique_lbs.remove(255)
                
                if len(unique_lbs) == 0:
                    continue
                rand_idx = np.random.choice(len(unique_lbs))
                c_min = unique_lbs[rand_idx]

                file_names['imgs_per_cls'][str(c_min)].append(line) 
                
            # STEP 2. assign imgs to tasks using class_list for each step 
            #         and corresponding imgs for each class assigned in STEP 1
            for step, cls_list in task_info.items():
                step_file_names = []
                    
                for cls in cls_list:
                    # does not consider bg class
                    if cls == 0:
                        continue
                    step_file_names += file_names['imgs_per_cls'][str(cls)]
                
                file_names['imgs_per_task'][str(step)] = step_file_names  
        # Test
        else:
            for step, cls_list in task_info.items():
                
                classes_idx_new = cls_list.copy()
                classes_idx_old = sum([cls_list.copy() for step_it, cls_list in task_info.items() if step_it < step], [])
            
                # remove bg class if it exists
                if 0 in classes_idx_new:
                    classes_idx_new.remove(0)
                if 0 in classes_idx_old:
                    classes_idx_old.remove(0)
            
                for ii, line in enumerate(lines):                       
                    if 'aug' not in self.split: # val
                        _image = self._image_dir / f"{line}.jpg"
                        _cat = self._cat_dir / f"{line}.png"
                    else: # train
                        _image = self._image_dir / line.split()[0][1:]
                        _cat = self._cat_dir / line.split()[1][1:]
                    assert _image.is_file(), _image
                    assert _cat.is_file(), _cat

                    cat = Image.open(_cat)
                    cat = np.array(cat, dtype=np.uint8)
                    
                    if not lbl_contains_any(cat, list(set(classes_idx_old + classes_idx_new))):
                        continue
                        
                    file_names['imgs_per_task'][str(step)].append(line)
        
        return file_names
        
    def __str__(self):
        return f"VOC2012(split={self.split})"

    def __len__(self):
        return len(self.images)

class VOCSegmentationIncrementalMemory(BaseDataset):
    def __init__(
        self,
        setting='overlap', name=None, method=None, step=0, classes_idx_new=[], classes_idx_old=[], idxs_path=None,
        transform=True, transform_args={}, masking_value=0, 
    ):
        # Experiment setup
        if setting in ['disjoint', 'overlap', 'partitioned']:
            pass
        # ablation setup for pseudo
        elif 'overlap_pseudo' in setting:
            pass
        else:
            raise ValueError('Wrong setting entered! Please use one of sequential, disjoint, overlap')

        super().__init__(
            transform_args=transform_args,
            base_dir=pathlib.Path(DATASETS_IMG_DIRS['voc']),
            transform=transform,
        )
        self.setting = setting
        self.name = name
        self.method = method
        self.step = step
        
        # (NOTE) these classes are composed of current step
        self.classes_idx_old = classes_idx_old
        self.classes_idx_new = classes_idx_new

        self.masking_value = masking_value

        self._image_dir = self._base_dir / "JPEGImages"
        self._cat_dir = self._base_dir / "SegmentationClassAug"

        self.im_idxs = []
        self.im_ids = []
        self.images = []
        self.categories = []
        self.gt_tasks = []

        with open(idxs_path, "r") as json_file:
            memory_list = json.load(json_file)

        file_names_pertask = memory_list["filenames_wtask"].copy()
        
        file_names = sum(list(file_names_pertask.values()), [])
        gt_task_filenames = [int(step.split('_')[-1]) for step, filenames_perstep in file_names_pertask.items() for _ in range(len(filenames_perstep))]
        
        for idx, (x, gt_task) in enumerate(zip(file_names, gt_task_filenames)):

            _image = self._image_dir / x
            _cat = self._cat_dir / f"{x.split('.')[0]}.png"

            assert _image.is_file(), _image
            assert _image.is_file(), _cat
            
            self.im_ids.append(x)
            self.images.append(_image)
            self.categories.append(_cat)
            self.gt_tasks.append(gt_task)
            
            self.im_idxs.append([idx])
            
        assert (len(self.images) == len(self.categories)) and (len(self.categories) == len(self.gt_tasks))

    def __getitem__(self, index):
        # STEP 1: filename -> img, target data
        _img, _target, _gt_task, _name = self._make_img_gt_point_pair(index)

        sample = {"image": _img, "label": _target, "image_name": _name, "gt_task": _gt_task }
        # STEP 2: img, target transformation(crop, flip)
        if self.transform:
            
            sample['image'], sample['label'] = self.transform_tr(sample)
            # Target mask per task transform
            sample['label'] = self.transform_target_masking(sample['label'], _gt_task)
        else:
            # NOTE: In MiB/PLOP framework, center crop is used for test data also (so this is not used)
            raise AssertionError
            # sample['image'], sample['label'] = self.transform_test(sample)

        return sample

    def _make_img_gt_point_pair(self, index):
        
        idx_list = self.im_idxs[index]
        
        idx = idx_list[0]
    
        _img = Image.open(self.images[idx]).convert("RGB")
        _target = Image.open(self.categories[idx])
        _gt_task = self.gt_tasks[idx]
        _name = str(self.im_ids[idx]).split(' ')[0].split('/')[-1]
        return _img, _target, _gt_task, _name
        
    def transform_tr(self, sample):
        
        composed_transforms = tr.Compose(
            [
                tr.RandomResizedCrop(
                    self.transform_args['crop_size'],
                    (0.5, 2.0)
                ),
                tr.RandomHorizontalFlip(),
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        
        return composed_transforms(sample['image'], sample['label'])

    def transform_val(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.Resize(size=self.transform_args['crop_size']),
                tr.CenterCrop(self.transform_args['crop_size']),
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_test(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_target_masking(self, target, gt_task):
        # Train or Validation
        # Masking except gt class of mem data(gt classes for task when the data is sampled)
        gt_class_per_img = get_classes_per_task(name=self.name, method=self.method, step=gt_task)
        
        if self.masking_value is None:
            return target
        if self.setting in ['disjoint', 'overlap', 'partitioned']:
            label_list = list(set(gt_class_per_img + [0, 255]))
            target_transform = tv.transforms.Lambda(
                lambda t: t.apply_(lambda x: x if x in label_list else self.masking_value)
            )
            return target_transform(target)

    def __len__(self):
        return len(self.im_idxs)