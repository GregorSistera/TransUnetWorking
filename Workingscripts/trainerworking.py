import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import dill
import multiprocessing.reduction
multiprocessing.reduction.ForkingPickler = dill._dill.Pickler

def dumps(obj):
    return dill.dumps(obj)

multiprocessing.reduction.ForkingPickler.dumps = staticmethod(dumps)
multiprocessing.reduction.ForkingPickler.loads = staticmethod(dill.loads)

import random
import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import DiceLoss
import torchvision.utils as vutils


def worker_init_fn(worker_id):
    print(f"[worker_init_fn] Worker {worker_id} initializing.")
    random.seed(1234 + worker_id)

def validate(model, dataloader, ce_loss_fn, dice_loss_fn, device, save_pred_path=None, epoch=None, num_classes=1):
    model.eval()
    val_loss_total = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss_ce = ce_loss_fn(outputs, labels.long())
            loss_dice = dice_loss_fn(outputs, labels, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            val_loss_total += loss.item() * images.size(0)
            num_samples += images.size(0)

            if save_pred_path and batch_idx == 0 and epoch is not None:
                preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                imgs_norm = (images - images.min()) / (images.max() - images.min())
                img_grid = vutils.make_grid(imgs_norm[:4].cpu())
                pred_grid = vutils.make_grid(preds[:4].float().cpu() * (255/(num_classes-1)))
                label_grid = vutils.make_grid(labels[:4].unsqueeze(1).float().cpu() * (255/(num_classes-1)))

                os.makedirs(save_pred_path, exist_ok=True)
                vutils.save_image(img_grid, os.path.join(save_pred_path, f'epoch_{epoch}_images.png'))
                vutils.save_image(pred_grid, os.path.join(save_pred_path, f'epoch_{epoch}_preds.png'))
                vutils.save_image(label_grid, os.path.join(save_pred_path, f'epoch_{epoch}_gts.png'))

    model.train()
    return val_loss_total / num_samples

def trainer_myepithelial(args, model, snapshot_path):
    from datasets.dataset_myepithelial import EpithelialNPZDataset

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

    transform = A.Compose([
        A.Resize(height=args.img_size, width=args.img_size),
        ToTensorV2(),
    ])

    train_list_path = os.path.join(args.list_dir, 'train.txt')
    val_list_path = os.path.join(args.list_dir, 'val.txt')

    db_train = EpithelialNPZDataset(npz_dir=args.root_path, list_file=train_list_path, transform=transform)
    db_val = EpithelialNPZDataset(npz_dir=args.root_path, list_file=val_list_path, transform=transform)

    logging.info(f"Train set size: {len(db_train)}")
    logging.info(f"Val set size: {len(db_val)}")

    trainloader = DataLoader(db_train,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0,
                             pin_memory=True,
                             worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=0,
                           pin_memory=True)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    model.train()

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = max_epoch * len(trainloader)
    logging.info(f"{len(trainloader)} iterations per epoch. {max_iterations} max iterations")

    best_val_loss = float('inf')

    for epoch_num in range(max_epoch):
        train_loss_sum = 0.0
        train_loss_ce_sum = 0.0

        epoch_iter = tqdm(trainloader, ncols=70, desc=f"Epoch {epoch_num} Training")
        for i_batch, (image_batch, label_batch) in enumerate(epoch_iter):
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            outputs = model(image_batch)

            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            train_loss_sum += loss.item() * image_batch.size(0)
            train_loss_ce_sum += loss_ce.item() * image_batch.size(0)

            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            writer.add_scalar('info/loss_ce', loss_ce.item(), iter_num)

            if iter_num % 20 == 0:
                imgs_norm = (image_batch[:4] - image_batch[:4].min()) / (image_batch[:4].max() - image_batch[:4].min())
                img_grid = vutils.make_grid(imgs_norm.cpu())
                preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)[:4]
                preds_grid = vutils.make_grid(preds.float().cpu() * (255/(num_classes-1)))
                gt_grid = vutils.make_grid(label_batch[:4].unsqueeze(1).float().cpu() * (255/(num_classes-1)))

                writer.add_image('train/Images', img_grid, iter_num)
                writer.add_image('train/Predictions', preds_grid, iter_num)
                writer.add_image('train/GroundTruth', gt_grid, iter_num)

            epoch_iter.set_postfix(loss=loss.item(), loss_ce=loss_ce.item())

        avg_train_loss = train_loss_sum / len(db_train)
        avg_train_ce = train_loss_ce_sum / len(db_train)
        logging.info(f"Epoch {epoch_num} training: Avg Loss: {avg_train_loss:.6f}, Avg CE Loss: {avg_train_ce:.6f}")

        val_loss = validate(model, valloader, ce_loss, dice_loss, device,
                            save_pred_path=os.path.join(snapshot_path, 'val_preds'),
                            epoch=epoch_num, num_classes=num_classes)
        logging.info(f"Epoch {epoch_num} validation loss: {val_loss:.6f}")
        writer.add_scalar('val/total_loss', val_loss, epoch_num)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(snapshot_path, 'best_model.pth')
            torch.save(model.state_dict(), best_path)
            logging.info(f"Saved best model checkpoint to {best_path}")

        if epoch_num > max_epoch // 2 and (epoch_num + 1) % 50 == 0:
            ckpt_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"Saved model checkpoint to {ckpt_path}")

        if epoch_num == max_epoch - 1:
            final_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            torch.save(model.state_dict(), final_path)
            logging.info(f"Saved final model checkpoint to {final_path}")

    writer.close()
    return "Training Finished!"
