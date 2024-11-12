class TrainerFactory():
    def __init__(self):
        pass
    
    @staticmethod
    def get_trainer(task_step, method, **kwargs):
        if method == 'MiB':
            from trainer.mib_trainer_voc import Trainer_base, Trainer_incremental
        elif method == 'PLOP':
            pass
        elif method == 'DKD':
            from trainer.dkd_trainer_voc import Trainer_base, Trainer_incremental
        else:
            # TODO: PLOP
            raise ValueError("Not implemented Yet")
        
        if task_step > 0:
            return Trainer_incremental(model=kwargs['model'], model_old=kwargs['model_old'],
                                    optimizer=kwargs['optimizer'],
                                    evaluator=(kwargs['evaluator_val'], kwargs['evaluator_test']),
                                    config=kwargs['config'],
                                    task_info=kwargs['task_info'],
                                    data_loader=(kwargs['train_loader'], kwargs['val_loader'], kwargs['test_loader'], kwargs['mem_loader']),
                                    lr_scheduler=kwargs['lr_scheduler'],
                                    logger=kwargs['logger'], gpu=kwargs['gpu'])
        else:
            return Trainer_base(model=kwargs['model'], 
                                    optimizer=kwargs['optimizer'],
                                    evaluator=(kwargs['evaluator_val'], kwargs['evaluator_test']),
                                    config=kwargs['config'],
                                    task_info=kwargs['task_info'],
                                    data_loader=(kwargs['train_loader'], kwargs['val_loader'], kwargs['test_loader']),
                                    lr_scheduler=kwargs['lr_scheduler'],
                                    logger=kwargs['logger'], gpu=kwargs['gpu'])
        