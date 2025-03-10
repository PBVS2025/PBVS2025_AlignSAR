# AdaptFormer for SAR Image Classification

## Overview
This repository contains the implementation of AdaptFormer for SAR image classification. AdaptFormer is a parameter-efficient fine-tuning method that adapts pre-trained vision transformers to downstream tasks by introducing lightweight adapter modules.

## Key Features
- Parameter-efficient fine-tuning of Vision Transformers
- FFN adaptation for improved performance
- Support for SAR image classification
- Pre-trained model integration

## Environment Setup

### Option 1: Using conda (recommended)
```bash
# Create conda environment from environment.yml
conda env create -f environment.yml
conda activate videoinr
```

### Option 2: Manual installation
```bash
# Create a new conda environment
conda create -n videoinr python=3.8
conda activate videoinr

# Install PyTorch and other dependencies
conda install pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.3 -c pytorch
pip install timm==0.3.2 tensorboard matplotlib easydict
```

## Dataset Structure
The dataset should be organized as follows:
```
/path/to/dataset/
├── train/
│   ├── sar/
│   │   ├── class_0/
│   │   ├── class_1/
│   │   └── ...
│   └── eo/
│       ├── class_0/
│       ├── class_1/
│       └── ...
└── test/
```

## Pre-trained Models
Pre-trained models should be placed in the `pretrained/` directory:
- `mae_pretrain_vit_b_edit.pth`: MAE pre-trained ViT-B model
- `checkpoint-XX.pth`: Fine-tuned checkpoints (where XX is the epoch number)

## Training

### Basic Training
You can use the provided training script:
```bash
# Run training
./train9.sh
```

The `train9.sh` script contains:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2

python main_image.py \
    --batch_size 128 \
    --cls_token \
    --finetune pretrained/mae_pretrain_vit_b_edit.pth \
    --data_path /path/to/dataset \
    --drop_path 0.0 \
    --blr 5e-4 \
    --epochs 50 \
    --warmup_epochs 5 \
    --ffn_adapt \
    --output_dir experiments/exp09
```

### Training Parameters
- `--batch_size`: Batch size for training
- `--cls_token`: Use classification token
- `--finetune`: Path to pre-trained model
- `--data_path`: Path to dataset
- `--drop_path`: Drop path rate
- `--blr`: Base learning rate
- `--epochs`: Number of training epochs
- `--warmup_epochs`: Number of warmup epochs
- `--ffn_adapt`: Use FFN adaptation
- `--output_dir`: Directory to save outputs


## Inference

## Pre-trained Models

Due to file size limitations, pre-trained models are not included in this repository. 
You can download them from the following Google Drive link:

[Download Pre-trained Models](https://drive.google.com/drive/folders/1AJPyluRsgQHLjaW28ALIDaHHSEBbV67v?usp=sharing))

After downloading, place the model files in the `pretrained/` directory:
- `mae_pretrain_vit_b_edit.pth`: MAE pre-trained ViT-B model
- `checkpoint-best.pth`: Fine-tuned checkpoints

The pre-trained models include:
1. MAE pre-trained ViT-B model: Base Vision Transformer model pre-trained using Masked Autoencoder approach
2. Fine-tuned checkpoints: Models fine-tuned on SAR image classification task with AdaptFormer approach


### Testing 
You can use the provided test script:
```bash
# Test a specific checkpoint
./test3.sh
```

The `test3.sh` script contains:
```bash
export CUDA_VISIBLE_DEVICES=0

python main_image_test2.py \
    --batch_size 128 \
    --cls_token \
    --finetune pretrained/mae_pretrain_vit_b_edit.pth \
    --data_path /path/to/test/data \
    --resume pretrained/checkpoint-24.pth \
    --drop_path 0.0 \
    --blr 0.1 \
    --dataset cifar100 \
    --nb_classes 10 \
    --ffn_adapt \
    --output_dir test/exp09/epoch24 \
    --epochs 50 \
    --eval
```


## Citation
If you use this code, please cite:
```
@inproceedings{chen2022adaptformer,
  title={AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition},
  author={Chen, Shoufa and Ge, Chongjian and Tong, Zhan and Wang, Jie and Song, Yale and Wang, Jianfei and Luo, Ping},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```

## License
MIT License
