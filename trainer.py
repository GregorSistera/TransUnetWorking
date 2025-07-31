# ✅ Refactored Training Script for TransUNet with Improvements

import multiprocessing

multiprocessing.set_start_method('spawn', force=True)

import dill
import multiprocessing.reduction

multiprocessing.reduction.ForkingPickler = dill._dill.Pickler


def dumps(obj):
    return dill.dumps(obj)


multiprocessing.reduction.ForkingPickler.dumps = staticmethod(dumps)
multiprocessing.reduction.ForkingPickler.loads = staticmethod(dill.loads)

import os
import sys
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torchvision.utils as vutils
from utils import DiceLoss
from datasets.dataset_myepithelial import EpithelialNPZDataset


def worker_init_fn(worker_id):
    random.seed(1234 + worker_id)


def post_process(pred):
    import cv2
    pred = pred.squeeze().cpu().numpy().astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)
    pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)
    return torch.tensor(pred).unsqueeze(0)


class ComboLoss(nn.Module):
    def __init__(self, alpha=0.6, gamma=2.0, num_classes=1):
        super().__init__()
        self.dice = DiceLoss(num_classes)
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = self.ce(inputs, targets.long())
        pt = torch.exp(-ce)
        focal_loss = ((1 - pt) ** self.gamma * ce).mean()
        dice_loss = self.dice(inputs, targets, softmax=True)
        return self.alpha * dice_loss + (1 - self.alpha) * focal_loss


def get_transform(img_size):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        A.ColorJitter(p=0.3),
        A.Normalize(),
        ToTensorV2(),
    ])


def validate(model, dataloader, loss_fn, device, save_pred_path=None, epoch=None, num_classes=1):
    model.eval()
    val_loss_total = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            val_loss_total += loss.item() * images.size(0)
            num_samples += images.size(0)

            if save_pred_path and batch_idx == 0 and epoch is not None:
                preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                preds = torch.stack([post_process(p) for p in preds])

                imgs_norm = (images - images.min()) / (images.max() - images.min())
                img_grid = vutils.make_grid(imgs_norm[:4].cpu())
                pred_grid = vutils.make_grid(preds[:4].float().cpu() * (255 / (num_classes - 1)))
                label_grid = vutils.make_grid(labels[:4].unsqueeze(1).float().cpu() * (255 / (num_classes - 1)))

                os.makedirs(save_pred_path, exist_ok=True)
                vutils.save_image(img_grid, os.path.join(save_pred_path, f'epoch_{epoch}_images.png'))
                vutils.save_image(pred_grid, os.path.join(save_pred_path, f'epoch_{epoch}_preds.png'))
                vutils.save_image(label_grid, os.path.join(save_pred_path, f'epoch_{epoch}_gts.png'))

    model.train()
    return val_loss_total / num_samples


def trainer_myepithelial(args, model, snapshot_path):
    os.makedirs(snapshot_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(snapshot_path, "log.txt"),
                        level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    transform = get_transform(args.img_size)

    db_train = EpithelialNPZDataset(args.root_path, os.path.join(args.list_dir, 'train.txt'), transform)
    db_val = EpithelialNPZDataset(args.root_path, os.path.join(args.list_dir, 'val.txt'), transform)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=True)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.to(device).train()

    loss_fn = ComboLoss(alpha=0.6, gamma=2.0, num_classes=num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))

    iter_num = 0
    best_val_loss = float('inf')

    for epoch_num in range(args.max_epochs):
        train_loss_total = 0.0
        epoch_iter = tqdm(trainloader, ncols=70, desc=f"Epoch {epoch_num} Training")

        for images, labels in epoch_iter:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            train_loss_total += loss.item() * images.size(0)

            writer.add_scalar('train/total_loss', loss.item(), iter_num)

            if iter_num % 20 == 0:
                imgs_norm = (images[:4] - images[:4].min()) / (images[:4].max() - images[:4].min())
                img_grid = vutils.make_grid(imgs_norm.cpu())
                preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)[:4]
                preds_grid = vutils.make_grid(preds.float().cpu() * (255 / (num_classes - 1)))
                gt_grid = vutils.make_grid(labels[:4].unsqueeze(1).float().cpu() * (255 / (num_classes - 1)))

                writer.add_image('train/Images', img_grid, iter_num)
                writer.add_image('train/Predictions', preds_grid, iter_num)
                writer.add_image('train/GroundTruth', gt_grid, iter_num)

            epoch_iter.set_postfix(loss=loss.item())

        scheduler.step()
        avg_train_loss = train_loss_total / len(db_train)
        logging.info(f"Epoch {epoch_num} Avg Training Loss: {avg_train_loss:.6f}")

        val_loss = validate(model, valloader, loss_fn, device,
                            save_pred_path=os.path.join(snapshot_path, 'val_preds'),
                            epoch=epoch_num, num_classes=num_classes)
        logging.info(f"Epoch {epoch_num} Val Loss: {val_loss:.6f}")
        writer.add_scalar('val/total_loss', val_loss, epoch_num)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(snapshot_path, 'best_model.pth'))
            logging.info("Saved best model")

        if epoch_num == args.max_epochs - 1:
            torch.save(model.state_dict(), os.path.join(snapshot_path, f'epoch_{epoch_num}.pth'))
            logging.info("Saved final model")

    writer.close()
    return "Training Finished!"
