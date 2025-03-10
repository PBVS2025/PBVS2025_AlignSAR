# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from easydict import EasyDict
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_

import util.misc as misc
from util.pos_embed import interpolate_pos_embed_ori as interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

# from datasets.image_datasets import build_image_dataset
from eo_sar_data import SarEODataset, PairedTransform
from utils import compute_adjustment
from torchsampler import ImbalancedDatasetSampler


from engine_finetune import train_one_epoch, evaluate
import models.vit_image_second as vit_image

from collections import defaultdict
import csv
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IDX2IDX = {
    0: 1,  # 'SUV' -> 'suv'
    1: 4,  # 'box_truck' -> 'box truck'
    2: 7,  # 'bus' -> 'bus'
    3: 6,  # 'flatbed_truck' -> 'flatbed truck'
    4: 5,  # 'motorcycle' -> 'motorcycle'
    5: 2,  # 'pickup_truck' -> 'pickup truck'
    6: 8,  # 'pickup_truck_w_trailer' -> 'pickup truck with trailer'
    7: 0,  # 'sedan' -> 'sedan'
    8: 9,  # 'semi_w_trailer' -> 'semi truck with trailer'
    9: 3,  # 'van' -> 'van'
}
# Challenge test 데이터셋용 클래스
class SarTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sar_paths = []
        self.image_names = []  # 파일 이름 저장용
        
        # SAR 이미지 파일 리스트 수집
        for file_name in sorted(os.listdir(root_dir)):
            if file_name.endswith(('.jpg', '.png')):  # 적절한 파일 확장자 추가
                self.sar_paths.append(os.path.join(root_dir, file_name))
                self.image_names.append(file_name)

    def __len__(self):
        return len(self.sar_paths)

    def __getitem__(self, idx):
        sar_path = self.sar_paths[idx]
        sar_image = Image.open(sar_path).convert('RGB')
        
        if self.transform:
            sar_image = self.transform(sar_image)
        else:
            resize = transforms.Resize((224, 224))  # Resize 추가
            to_tensor = transforms.ToTensor()
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            sar_image = normalize(to_tensor(resize(sar_image)))
            
        return sar_image, self.image_names[idx]

def get_args_parser():
    parser = argparse.ArgumentParser('AdaptFormer fine-tuning for action recognition for image classification', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # custom configs
    parser.add_argument('--dataset', default='imagenet', choices=['imagenet', 'cifar100', 'flowers102', 'svhn', 'food101'])
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')

    parser.add_argument('--inception', default=False, action='store_true', help='whether use INCPETION mean and std'
                                                                                '(for Jx provided IN-21K pretrain')
    # AdaptFormer related parameters
    parser.add_argument('--ffn_adapt', default=False, action='store_true', help='whether activate AdaptFormer')
    parser.add_argument('--ffn_num', default=64, type=int, help='bottleneck middle dimension')
    parser.add_argument('--vpt', default=False, action='store_true', help='whether activate VPT')
    parser.add_argument('--vpt_num', default=1, type=int, help='number of VPT prompts')
    parser.add_argument('--fulltune', default=False, action='store_true', help='full finetune model')

    return parser

def save_adjustment(adjustment, file_path):
        
        torch.save(adjustment, file_path)
        print(f"Logit adjustment is saved to {file_path}.")

def load_adjustment(file_path, device):
    if os.path.exists(file_path):
        adjustment = torch.load(file_path, map_location=device)
        print(f"Logit adjustment is loaded from {file_path}.")
        return adjustment
    else:
        print(f"File path:  {file_path} doesn't exist.")
        return None

def main(args):
    if args.log_dir is None:
        args.log_dir = args.output_dir
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    
    test_dataset = SarTestDataset(root_dir=args.data_path)
        
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    
    data_loader_val = None

    # fine-tuning configs
    tuning_config = EasyDict(
        # AdaptFormer
        ffn_adapt=args.ffn_adapt,
        ffn_option="parallel",
        ffn_adapter_layernorm_option="none",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",
        ffn_num=args.ffn_num,
        d_model=768,
        # VPT related
        vpt_on=args.vpt,
        vpt_num=args.vpt_num,
    )

    if args.model.startswith('vit'):
        model = vit_image.__dict__[args.model](
        num_classes=args.nb_classes,
        global_pool=args.global_pool,
        drop_path_rate=args.drop_path,
        tuning_config=tuning_config,
        distilled=True,
        )
    else:
        raise NotImplementedError(args.model)

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        # interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        
        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)
        trunc_normal_(model.head_dist.weight, std=0.01)

    # for linear prob only
    # hack: revise model's head with BN
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    
    model.head_dist = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head_dist.in_features, affine=False, eps=1e-6), model.head_dist)
    
    model.to(device)

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=None, loss_scaler=None)

    if args.eval:
        model.eval()
        
        network_output = np.array([])
        scores = np.array([])
        names = []
        predictions = []
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for sar_img, img_name in test_loader:
                    sar_img = sar_img.to(device)
                    
                    outputs, _ = model(sar_img)
                    
                    score, predicted = torch.max(outputs.data, 1)
                    network_output = np.concatenate((network_output, predicted.detach().cpu().numpy()))
                    scores = np.concatenate((scores, score.detach().cpu().numpy()))
                    names.extend(img_name)
                    
                    if len(names) % 100 == 0:
                        print(f'Processed {len(names)}/{len(test_loader)} images')
                        
        with open(os.path.join(args.output_dir, 'results.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['image_id', 'class_id', 'score'])
            
            for idx, name in enumerate(names):
                image_id = name.replace('Gotcha', '').replace('.png', '')
                
                writer.writerow([
                    image_id, 
                    IDX2IDX[int(network_output[idx])],
                    scores[idx]
                ])

        print(f'\nPredictions saved to results.csv')
        exit(0)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
