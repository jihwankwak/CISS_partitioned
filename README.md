## 1. Environment
> - OS: Ubuntu 18.04.5 LTS
> - CUDA: 11.2
> - GPU: Titan XP
> - CPU: Intel(R) Xeon(R) Silver 4110 CPU @ 2.10GHz 

## 2. Conda Environment
1. Manual installation
```
conda create --name ciss_mem python=3.7
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -c nvidia
```

2. Download Environment
[environment.yml](https://drive.google.com/drive/folders/1FJlHERP47lQci5t0a-TDXOBJ6ctcLbBk?usp=share_link)


## 3. Dataset

Following [Baek et al](https://github.com/cvlab-yonsei/DKD), we use augmented 10,582 training samples and 1,449 validation samples for PASCAL VOC 2012. You can download the 1) original dataset 2) labels of augmentend samples ('SegmentationClassAug') 3) file names ('train_aug.txt') following the instructions of above repo. The structure of data path should be organized as follows: 

```
...
├── data_loader
├── dataset
│   └── VOC2012
│       ├── Annotations
│       ├── ImageSets
│       ├── JPEGImages
│       ├── list
│       ├── saliency_map
│       ├── SegmentationClass
│       ├── SegmentationClassAug
│       │   ├── train_aug.txt
│       │   └── val.txt 
│       ├── SegmentationObject
│       └── splits
├── logger
...
```
