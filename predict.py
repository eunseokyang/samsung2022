import argparse
import logging
import os
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.io import write_png

from dataloader import ImageDataset, MultiImageDataset
from unet import UNet, UNetPlus

# python predict.py --exp 220610_162631 --validation --epochs 3
# python predict.py --exp 220607_163752 --single_image --validation --epochs 30

IMGSIZE = (66, 45)

checkpoint_root = Path('./checkpoints')

def save_result(pred, save_dir, fname):
    path = save_dir / (fname + '.png')
    # compression level?
    write_png(pred, str(path))


def predict(model, device, save_dir, single_image=False, is_validation=False):
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset_cl = ImageDataset if single_image else MultiImageDataset
    batch_size = 4 if single_image else 1

    if not is_validation:
        test_dataset = dataset_cl(option='Test')
    else:
        test_dataset = dataset_cl(option='Validation', is_test=True)
    fnames = sorted(list(set(['_'.join(s.split('_')[:-1]) for s in test_dataset.sem_list])))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # assert len(test_loader) == len(fnames)

    model.eval()
    for batch, org in enumerate(test_loader):
        org = org.to(device=device)
        with torch.no_grad():
            pred = model(org)
            pred = torch.sigmoid(pred)

        pred = pred.float().cpu()
        pred = pred.mean(axis=0)
        pred = (pred * 255).type(torch.uint8)

        save_result(pred, save_dir, fnames[batch])


def get_args():
    parser = argparse.ArgumentParser(description='Depth estimation for SEM')

    parser.add_argument('--epochs', '-e', type=int, default=3, help='Load model from a .pth file')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--exp', help='experiment date', default='first')
    parser.add_argument('--validation', action='store_true', default=False)
    parser.add_argument('--single_image', action='store_true', default=False, help='single imgae input')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if args.single_image:
        model = UNet(n_channels=1, n_classes=1, bilinear=args.bilinear)
    else:
        model = UNetPlus(n_channels=1, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.validation:
        save_root = Path('./results_validation/')
    else:
        save_root = Path('./results')

    checkpoint_dir = checkpoint_root / args.exp
    save_dir = save_root / args.exp
    model_path = checkpoint_dir / f'checkpoint_epoch{args.epochs}.pth'

    logging.info(f'Loading model {model_path}')
    logging.info(f'Using device {device}')

    model.to(device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    logging.info('Model loaded')

    predict(model=model,
            device=device,
            save_dir=save_dir,
            single_image=args.single_image,
            is_validation=args.validation)

