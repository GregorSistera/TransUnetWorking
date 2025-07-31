import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import logging

from datasets.dataset_myepithelial import EpithelialNPZDataset
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


def get_args():
    parser = argparse.ArgumentParser(description="TransUNet Inference")
    parser.add_argument('--dataset', type=str, default='MyEpithelialDataset')
    parser.add_argument('--npz_dir', type=str, default='./inference_data')
    parser.add_argument('--list_file', type=str, default='./lists/lists_MyEpithelialDataset/infer.txt')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16')
    parser.add_argument('--n_skip', type=int, default=3)
    parser.add_argument('--vit_patches_size', type=int, default=16)
    parser.add_argument('--is_savenii', action='store_true')
    parser.add_argument('--test_save_dir', type=str, default='./predictions')
    parser.add_argument('--deterministic', type=int, default=1)
    parser.add_argument('--base_lr', type=float, default=0.0075)
    parser.add_argument('--seed', type=int, default=1234)
    return parser.parse_args()


def setup_environment(args):
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


def update_infer_txt(npz_dir, list_file):
    if not os.path.exists(npz_dir):
        raise FileNotFoundError(f"❌ NPZ directory does not exist: {npz_dir}")

    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    base_names = sorted([os.path.splitext(f)[0] for f in npz_files])

    os.makedirs(os.path.dirname(list_file), exist_ok=True)
    with open(list_file, 'w') as f:
        for name in base_names:
            f.write(name + '\n')

    print(f"📄 infer.txt updated with {len(base_names)} entries from {npz_dir}")


def load_model(args):
    exp_name = f'TU_{args.dataset}_{args.img_size}'
    snapshot_path = os.path.join("model", exp_name,
                                 f"TU_pretrain_{args.vit_name}_skip{args.n_skip}_epo{args.max_epochs}_bs{args.batch_size}_{args.img_size}")
    model_path = os.path.join(snapshot_path, 'best_model.pth')

    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        sys.exit(1)

    config = CONFIGS_ViT_seg[args.vit_name]
    config.n_classes = args.num_classes
    config.n_skip = args.n_skip
    config.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if 'R50' in args.vit_name:
        config.patches.grid = (args.img_size // args.vit_patches_size,
                               args.img_size // args.vit_patches_size)

    model = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    model.load_state_dict(torch.load(model_path))
    print(f"✅ Loaded model from: {model_path}")
    return model, exp_name


def inference(args, model, save_path=None):
    dataset = EpithelialNPZDataset(npz_dir=args.npz_dir, list_file=args.list_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    model.eval()

    print(f"🧪 Running inference on {len(dataset)} samples...")
    with torch.no_grad():
        for idx, (image, _) in enumerate(tqdm(dataloader)):
            image = image.cuda()
            pred = model(image)
            pred_soft = torch.softmax(pred, dim=1)
            pred_label = torch.argmax(pred_soft, dim=1).squeeze().cpu().numpy().astype(np.uint8)

            if save_path:
                mask_img = (pred_label * (255 // max(1, args.num_classes - 1))).astype(np.uint8)
                img_out = Image.fromarray(mask_img)
                out_path = os.path.join(save_path, f"pred_{idx:03d}.png")
                img_out.save(out_path)


def main():
    args = get_args()
    setup_environment(args)

    # Update infer.txt before running inference
    update_infer_txt(args.npz_dir, args.list_file)

    model, exp_name = load_model(args)

    if args.is_savenii:
        pred_dir = os.path.join(args.test_save_dir, exp_name)
        os.makedirs(pred_dir, exist_ok=True)
    else:
        pred_dir = os.path.join(args.test_save_dir, exp_name)
        os.makedirs(pred_dir, exist_ok=True)

    # Logging
    os.makedirs('./test_log', exist_ok=True)
    log_file = os.path.join('./test_log', f'test_log_{exp_name}.txt')
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(f"🚀 Starting inference...")
    inference(args, model, save_path=pred_dir)
    logging.info("✅ Inference completed.")


if __name__ == "__main__":
    main()
