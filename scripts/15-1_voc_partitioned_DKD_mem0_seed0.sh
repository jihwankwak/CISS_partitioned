GPU=0,1,2,3
BS=8  # Total 24
SAVEDIR=./results/15-1_voc_partitioned_DKD_mem100_seed0
SEED=0

TASKSETTING=partitioned  # or 'disjoint'
TASKNAME='15-1' # or ['15-1', '19-1', '10-1', '5-3']

INIT_LR=0.001
LR=0.0001
INIT_EPOCH=60
EPOCH=60
INIT_POSWEIGHT=2
MEMORY_SIZE=0  # 100 for DKD-M

NAME='DKD'
BASEMODEL_DIR='./'

python train_voc.py -c configs/DKD/config_voc.json \
	-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
	--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} --epochs ${INIT_EPOCH} --basemodel_dir ${BASEMODEL_DIR} --pos_weight ${INIT_POSWEIGHT} --seed ${SEED}

python train_voc.py -c configs/DKD/config_voc.json \
	-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
	--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --epochs ${EPOCH} --mem_size ${MEMORY_SIZE} --seed ${SEED}

python train_voc.py -c configs/DKD/config_voc.json \
	-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
	--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --freeze_bn --epochs ${EPOCH} --mem_size ${MEMORY_SIZE} --seed ${SEED}

python train_voc.py -c configs/DKD/config_voc.json \
	-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
	--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} --freeze_bn --epochs ${EPOCH} --mem_size ${MEMORY_SIZE} --seed ${SEED}

python train_voc.py -c configs/DKD/config_voc.json \
	-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
	--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} --freeze_bn --epochs ${EPOCH} --mem_size ${MEMORY_SIZE} --seed ${SEED}

python train_voc.py -c configs/DKD/config_voc.json \
	-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
	--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --freeze_bn --epochs ${EPOCH} --mem_size ${MEMORY_SIZE} --seed ${SEED}