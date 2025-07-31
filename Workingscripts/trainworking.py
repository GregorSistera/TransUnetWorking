import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import multiprocessing

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_myepithelial  # Your custom trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='C:/Users/Georges/Desktop/IHCEPITHELIALSEGDATASET/npz_data/train_npz')
    parser.add_argument('--dataset', type=str, default='MyEpithelialDataset')
    parser.add_argument('--list_dir', type=str,
                        default='C:/Users/Georges/Desktop/IHCEPITHELIALSEGDATASET/lists/lists_MyEpithelialDataset')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--max_iterations', type=int, default=30000)
    parser.add_argument('--max_epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--deterministic', type=int, default=1)
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--n_skip', type=int, default=3)
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16')
    parser.add_argument('--vit_patches_size', type=int, default=16)
    return parser.parse_args()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)  # must be in main guard on Windows
    args = parse_args()

    dataset_config = {
        'MyEpithelialDataset': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'num_classes': 2,
        }
    }

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    assert dataset_name in dataset_config, f"Dataset '{dataset_name}' not found in config."
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    args.is_pretrain = True
    args.exp = f'TU_{dataset_name}_{args.img_size}'
    snapshot_path = os.path.join("model", args.exp, 'TU')
    if args.is_pretrain:
        snapshot_path += '_pretrain'
    snapshot_path += f'_{args.vit_name}_skip{args.n_skip}'
    if args.vit_patches_size != 16:
        snapshot_path += f'_vitpatch{args.vit_patches_size}'
    if args.max_iterations != 30000:
        snapshot_path += f'_{str(args.max_iterations)[:2]}k'
    if args.max_epochs != 30:
        snapshot_path += f'_epo{args.max_epochs}'
    snapshot_path += f'_bs{args.batch_size}'
    if args.base_lr != 0.01:
        snapshot_path += f'_lr{args.base_lr}'
    snapshot_path += f'_{args.img_size}'
    if args.seed != 1234:
        snapshot_path += f'_s{args.seed}'

    os.makedirs(snapshot_path, exist_ok=True)

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if 'R50' in args.vit_name:
        grid_size = int(args.img_size / args.vit_patches_size)
        config_vit.patches.grid = (grid_size, grid_size)

    config_vit.pretrained_path = "model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz"
    assert os.path.exists(config_vit.pretrained_path), f"Pretrained file not found at {config_vit.pretrained_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(device)
    net.load_from(weights=np.load(config_vit.pretrained_path))

    trainer = {
        'MyEpithelialDataset': trainer_myepithelial
    }
    trainer[dataset_name](args, net, snapshot_path)
