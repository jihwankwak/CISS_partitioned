import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel

from torch.nn.parallel import DistributedDataParallel as DDP
from base import BaseTrainer
from utils import MetricTracker, MetricTracker_scalars
from models.loss import UnbiasedCrossEntropy, UnbiasedKnowledgeDistillationLoss, UnbiasedCrossEntropyMem
from data_loader import VOC, CLS_INFO



class Trainer_base(BaseTrainer):
    """
    Trainer class for a base step
    """
    def __init__(
        self, model, optimizer, evaluator, config, task_info,
        data_loader, lr_scheduler=None, logger=None, gpu=None
    ):
        super().__init__(config, logger, gpu)
        if not torch.cuda.is_available():
            logger.info("using CPU, this will be slow")
        elif config['multiprocessing_distributed']:
            if gpu is not None:
                torch.cuda.set_device(self.device)
                model.to(self.device)
                # When using a single GPU per process and per
                # DDP, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                self.model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False

            else:
                model.to(self.device)
                # DDP will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                self.model = DDP(model)

        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            self.model = nn.DataParallel(model, device_ids=self.device_ids)

        self.dataset = config['dataset']
        self.batch_size = config['data_loader']['args']['train']['batch_size']
        self.optimizer = optimizer
        self.evaluator_val = evaluator[0]
        self.evaluator_test = evaluator[1]

        self.task_info = task_info
        # (TODO) This can be handled by task_info rather than initialization
        self.n_classes = self.task_info['n_classes']
        # for step 2 in 15-1
        self.n_old_classes = len(self.task_info['old_class'])  # (MiB) 17 (DKD) 16 
        self.n_new_classes = len(self.task_info['new_class'])  # (MiB) 1 (DKD) 1
        self.new_classes = self.task_info['new_class']
        # In MiB, dismiss background label for n_old_classses & n_classes
        # self.batch_mem_ratio = int(self.batch_size * ((self.n_old_classes-1)/(self.n_classes-1)))
        
        # if self.rank == 0:
        #     self.logger.info(f"batch_size: {self.batch_size}")
        #     self.logger.info(f"batch_memratio: {self.batch_mem_ratio}")
        
        if config['hyperparameter']['pseudo_labeling']!='None':
            self.pseudo_labeling = float(config['hyperparameter']['pseudo_labeling'])
        else:  
            self.pseudo_labeling = None

        self.train_loader = data_loader[0]
        if self.train_loader is not None:
            self.len_epoch = len(self.train_loader)

        self.val_loader = data_loader[1]
        if self.val_loader is not None:
            self.do_validation = self.val_loader is not None

        self.test_loader = data_loader[2]
        if self.test_loader is not None:
            self.do_test = self.test_loader is not None
       
        self.lr_scheduler = lr_scheduler

        # For automatic mixed precision(AMP)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config['use_amp'])

        if self.evaluator_val is not None:
            self.metric_ftns_val = [getattr(self.evaluator_val, met) for met in config['metrics']]
        if self.evaluator_test is not None:
            self.metric_ftns_test = [getattr(self.evaluator_test, met) for met in config['metrics']]
        

        # (NOTE) This should be set per method
        self.train_metrics = MetricTracker(
            'loss', 'loss_unce', 'loss_unkd',
            writer=self.writer,
            colums=['total', 'counts', 'average'],
        )
        self.valid_metrics = MetricTracker_scalars(writer=self.writer)
        self.test_metrics = MetricTracker_scalars(writer=self.writer)
        
        # self.ret_mode = config['data_loader']['args']['memory']['retrieval']['ret_mode']

        if config.resume is not None:
            self._resume_checkpoint(config.resume, config['test'])

        self.unbiasedCELoss = UnbiasedCrossEntropy(old_cl=self.n_old_classes, ignore_index=255, reduction="none")

        self._print_train_info()

    def _print_train_info(self):
        self.logger.info(f"Total loss = L_unce")

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        torch.distributed.barrier()

        self.model.train()
        # (NOTE) This does not exists in MiB, PLOP code but it should exist
        if isinstance(self.model, (nn.DataParallel, DDP)):
            self.model.module.freeze_bn(affine_freeze=False)
        else:
            self.model.freeze_bn(affine_freeze=False)

        self.train_metrics.reset()
        self.logger.info(f'Epoch - {epoch}')

        # Random shuffling
        if not isinstance(self.train_loader.sampler, torch.utils.data.RandomSampler):
            self.train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, data in enumerate(self.train_loader):
            data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
            with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                logit, _ = self.model(data['image'], ret_intermediate=False)
                
                loss = self.unbiasedCELoss(logit, data['label'])
                loss = loss.mean()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad(set_to_none=True)
            
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            # Get First lr
            if batch_idx == 0:
                self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch - 1)
                self.logger.info(f"lr[0]: {self.optimizer.param_groups[0]['lr']:.6f} / lr[1]: {self.optimizer.param_groups[1]['lr']:.6f} / lr[2]: {self.optimizer.param_groups[2]['lr']:.6f}")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.progress(self.logger, batch_idx, len(self.train_loader))

            if batch_idx == self.len_epoch:
                break

        # average train loss per epoch
        log = self.train_metrics.result()

        val_flag = False
        if self.do_validation and (epoch % self.validation_period) == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if self.rank == 0:
                val_flag = True

        return log, val_flag

    def _valid_epoch(self, epoch):
        torch.distributed.barrier()
        
        log = {}
        self.evaluator_val.reset()
        self.logger.info(f"Number of val loader: {len(self.val_loader)}")

        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.val_loader):
                data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
                target = data['label'].cpu().numpy()

                logit, _ = self.model(data['image'])

                # logit = torch.softmax(logit)
                pred = logit.argmax(dim=1)  # pred: [N. H, W]
                pred = pred.cpu().numpy()
                self.evaluator_val.add_batch(target, pred)

            if self.rank == 0:
                self.writer.set_step((epoch), 'valid')

            for met in self.metric_ftns_val:
                if 'confusion' in met().keys():
                    pass
                else:
                    if len(met().keys()) > 2:
                        self.valid_metrics.update(met.__name__, [met()['old'], met()['new'], met()['harmonic']], 'old', 'new', 'harmonic', n=1)
                    else:
                        self.valid_metrics.update(met.__name__, [met()['overall']], 'overall', n=1)

                if 'old' in met().keys():
                    log.update({met.__name__ + '_old': f"{met()['old']:.2f}"})
                if 'new' in met().keys():
                    log.update({met.__name__ + '_new': f"{met()['new']:.2f}"})
                if 'harmonic' in met().keys():
                    log.update({met.__name__ + '_harmonic': f"{met()['harmonic']:.2f}"})
                if 'overall' in met().keys():
                    log.update({met.__name__ + '_overall': f"{met()['overall']:.2f}"})
                if 'by_class' in met().keys():
                    by_class_str = '\n'
                    for i in range(len(met()['by_class'])):
                        if i in self.evaluator_val.new_classes_idx:
                            by_class_str = by_class_str + f"{i:2d} *{list(CLS_INFO[self.dataset].values())[i]} {met()['by_class'][i]:.2f}\n"
                        elif i in self.evaluator_val.old_classes_idx:
                            by_class_str = by_class_str + f"{i:2d}  {list(CLS_INFO[self.dataset].values())[i]} {met()['by_class'][i]:.2f}\n"
                    log.update({met.__name__ + '_by_class': by_class_str})
        return log

    def _test(self, epoch=None):
        torch.distributed.barrier()

        log = {}
        self.evaluator_test.reset()
        self.logger.info(f"Number of test loader: {len(self.test_loader)}")

        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
                target = data['label'].cpu().numpy()

                logit, _ = self.model(data['image'])

                pred = logit.argmax(dim=1)  # pred: [N. H, W]
                pred = pred.cpu().numpy()
                
                self.evaluator_test.add_batch(target, pred)

            if epoch is not None:
                if self.rank == 0:
                    self.writer.set_step((epoch), 'test')

            for met in self.metric_ftns_test:
                if epoch is not None:
                    if 'confusion' in met().keys():
                        pass
                    else:
                        if len(met().keys()) > 2:
                            self.test_metrics.update(met.__name__, [met()['old'], met()['new'], met()['harmonic']], 'old', 'new', 'harmonic', n=1)
                        else:
                            self.test_metrics.update(met.__name__, [met()['overall']], 'overall', n=1)

                if 'old' in met().keys():
                    log.update({met.__name__ + '_old': f"{met()['old']:.2f}"})
                if 'new' in met().keys():
                    log.update({met.__name__ + '_new': f"{met()['new']:.2f}"})
                if 'harmonic' in met().keys():
                    log.update({met.__name__ + '_harmonic': f"{met()['harmonic']:.2f}"})
                if 'overall' in met().keys():
                    log.update({met.__name__ + '_overall': f"{met()['overall']:.2f}"})
                if 'by_class' in met().keys():
                    by_class_str = '\n'
                    for i in range(len(met()['by_class'])):
                        if i in self.evaluator_test.new_classes_idx:
                            by_class_str = by_class_str + f"{i:2d} *{list(CLS_INFO[self.dataset].values())[i]} {met()['by_class'][i]:.2f}\n"
                        else:
                            by_class_str = by_class_str + f"{i:2d}  {list(CLS_INFO[self.dataset].values())[i]} {met()['by_class'][i]:.2f}\n"
                    log.update({met.__name__ + '_by_class': by_class_str})
                    
        return log        
    
class Trainer_incremental(Trainer_base):
    """
    Trainer class for incremental steps
    """
    def __init__(
        self, model, model_old, optimizer, evaluator, config, task_info,
        data_loader, lr_scheduler=None, logger=None, gpu=None
    ):
        super().__init__(
            model=model, optimizer=optimizer, evaluator=evaluator, config=config, task_info=task_info,
            data_loader=data_loader, lr_scheduler=lr_scheduler, logger=logger, gpu=gpu)
    
        self.mem_loader = data_loader[3]
        self.en_mem = True if self.mem_loader is not None else False
        
        if config['multiprocessing_distributed']:
            if gpu is not None:
                if model_old is not None:
                    model_old.to(self.device)
                    self.model_old = DDP(model_old, device_ids=[gpu])
            else:
                if model_old is not None:
                    model_old.to(self.device)
                    self.model_old = DDP(model_old)
        else:
            if model_old is not None:
                self.model_old = nn.DataParallel(model_old, device_ids=self.device_ids)

        # (NOTE) This should be set per method
        self.train_metrics = MetricTracker(
            'loss', 'loss_unce', 'loss_unkd', 'loss_uncemem', 
            writer=self.writer, colums=['total', 'counts', 'average'],
        )        
        if config.resume is not None:
            self._resume_checkpoint(config.resume, config['test'])
        
        if self.config['hyperparameter']['mem_loss'] == 'AugM':
            self.memloss = UnbiasedCrossEntropyMem(old_cl=self.n_old_classes, ignore_index=255, reduction="none")
        elif self.config['hyperparameter']['mem_loss'] == 'CE':
            self.memloss =  nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        else:
            raise NotImplementedError
        self.unbiasedKDloss = UnbiasedKnowledgeDistillationLoss(reduction="none", alpha=config['hyperparameter']['alpha'])

        if self.en_mem:
            self.logger.info(f"Current batch length : {len(self.train_loader)} | Memory batch length : {len(self.mem_loader)}")
            
    def _print_train_info(self):
        self.logger.info(f"Total loss = 0.5 * L_unce + 0.5 * L_unmem + {self.config['hyperparameter']['unkd']} * L_unkd ")

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        torch.distributed.barrier()

        self.model.train()
        if isinstance(self.model, (nn.DataParallel, DDP)):
            self.model.module.freeze_bn(affine_freeze=False)
            self.model.module.freeze_dropout()
        else:
            self.model.freeze_bn(affine_freeze=False)
            self.model.freeze_dropout()
        self.model_old.eval()

        self.train_metrics.reset()
        self.logger.info(f'Epoch - {epoch}')
        self.logger.info(f'Pseudo labeling threshold - {str(self.pseudo_labeling)}')

        # Random shuffling
        if not isinstance(self.train_loader.sampler, torch.utils.data.RandomSampler):
            self.train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, data in enumerate(self.train_loader):
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.en_mem:
                self.mem_loader.sampler.set_epoch(epoch)
                try:
                    mem_data = next(mem_iter)
                except:
                    mem_iter = iter(self.mem_loader)
                    mem_data = next(mem_iter)
                 
                rand_index = torch.randperm(self.batch_size)[:self.batch_size // 2]
                
                mem_data_idx = rand_index
                cur_data_idx = torch.tensor(list(set([i for i in range(self.batch_size)])-set(rand_index.tolist())))
                
                cur_img = data['image'][cur_data_idx, ...]
                cur_label = data['label'][cur_data_idx, ...]
                
                mem_img = mem_data['image'][mem_data_idx, ...]
                mem_label = mem_data['label'][mem_data_idx, ...]
                
                mem_img, mem_label = mem_img.to(self.device), mem_label.to(self.device).long()
            else:
                cur_img, cur_label, cur_names = data['image'], data['label'], data['image_name']
                
            # data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
            cur_img, cur_label = cur_img.to(self.device), cur_label.to(self.device).long()
            
            with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                cur_logit, cur_feat = self.model(cur_img, ret_intermediate=False)
                
                if self.en_mem:
                    mem_logit, mem_feat = self.model(mem_img, ret_intermediate=False)

                if self.model_old is not None:
                    with torch.no_grad():
                        cur_logit_old, _ = self.model_old(cur_img, ret_intermediate=False)
                        if self.en_mem:
                            mem_logit_old, _ = self.model_old(mem_img, ret_intermediate=False)
                        
                        if self.pseudo_labeling != None:
                            threshold = self.pseudo_labeling
                            
                            mask_background = cur_label < self.n_old_classes
                            probs = torch.softmax(cur_logit_old, dim=1)
                            mask_confidence = probs.max(dim=1)[0] > threshold
                            mask_total = mask_background & mask_confidence
                            cur_p_label = probs.argmax(dim=1)
                            # cur_p_label[probs.max(dim=1)[0] < threshold] = 255
                            cur_label[mask_total] = cur_p_label[mask_total]
                        else:
                            pass
                        
                # Unbiased Crossentropy
                if self.en_mem:
                    loss_unce = self.unbiasedCELoss(cur_logit, cur_label)
                    loss_uncemem = self.memloss(mem_logit, mem_label)
                    loss_ce = 0.5*loss_unce.mean() + 0.5*loss_uncemem.mean()
                else:
                    loss_unce = self.unbiasedCELoss(cur_logit, cur_label).mean()
                    loss_uncemem = torch.tensor(0.)
                    loss_ce = loss_unce
                
                # Unbiased Knowledge Distillation
                cur_loss_unkd = self.unbiasedKDloss(cur_logit, cur_logit_old, mask=None)
                
                if self.en_mem:
                    mem_loss_unkd = self.unbiasedKDloss(mem_logit, mem_logit_old, mask=None)
                    loss_unkd = 0.5*cur_loss_unkd +0.5*mem_loss_unkd
                else:
                    mem_loss_unkd = torch.tensor(0.)
                    loss_unkd = cur_loss_unkd
    
                #  feat : (N, 256, H/16, W/16)
    
                loss = loss_ce + self.config['hyperparameter']['unkd'] * loss_unkd.mean()
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('loss_unce', loss_unce.mean().item())
            self.train_metrics.update('loss_uncemem', loss_uncemem.mean().item())
            self.train_metrics.update('loss_unkd', loss_unkd.mean().item())

            # Get First lr
            if batch_idx == 0:
                self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch - 1)
                self.logger.info(f"lr[0]: {self.optimizer.param_groups[0]['lr']:.6f} / lr[1]: {self.optimizer.param_groups[1]['lr']:.6f} / lr[2]: {self.optimizer.param_groups[2]['lr']:.6f}")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.progress(self.logger, batch_idx, len(self.train_loader))

            if batch_idx == self.len_epoch:
                break

        # average train loss per epoch
        log = self.train_metrics.result()

        val_flag = False
        if self.do_validation and (epoch % self.validation_period) == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if self.rank == 0:
                val_flag = True

        return log, val_flag
