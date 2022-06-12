import argparse
import logging
from datetime import datetime
from pathlib import Path
import secrets

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb

from utils import set_seed
from dataloader import ImageDataset, MultiImageDataset
from unet import UNet, UNetPlus
from simple_vit import SimpleViT

now = datetime.now().strftime('%y%m%d_%H%M%S')
checkpoint_dir = Path(f'./checkpoints/{now}/')

IMGSIZE = (66, 45)

def evaluate(model, val_loader, device):
    loss_fn = nn.MSELoss()
    n_eval_batches = len(val_loader)
    losses = 0

    model.eval()
    for batch, (org, targ) in enumerate(val_loader):
        org = org.to(device=device, dtype=torch.float32)
        targ = targ.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            pred = model(org)
            pred = torch.sigmoid(pred)
            loss = loss_fn(pred, targ)
            losses += loss

    model.train()
    return losses / n_eval_batches


def train(model, device='cuda', epochs=5, batch_size=16, learning_rate=1e-3, every_eval_steps=200,
          is_transform=False, single_image=False, save_checkpoint=True):
    
    dataset_cl = ImageDataset if single_image else MultiImageDataset

    train_dataset = dataset_cl(option='Train', transform=is_transform)
    val_dataset = dataset_cl(option='Validation')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    experiment = wandb.init(project='samsung', reinit=True)
    # , mode="disabled"
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  save_checkpoint=save_checkpoint))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), pct_start=0.1, epochs=epochs)
    loss_fn = nn.MSELoss()

    global_step = 0
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0

        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch, (org, targ) in enumerate(train_loader):
                org = org.to(device=device, dtype=torch.float32)
                targ = targ.to(device=device, dtype=torch.float32)

                pred = model(org)
                pred = torch.sigmoid(pred) # ..
                loss = loss_fn(pred, targ)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # scheduler.step()

                pbar.update(org.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss': loss.item()})

                if global_step % every_eval_steps == 0:
                    histograms = {}
                    for tag, value in model.named_parameters():
                        tag = tag.replace('/', '.')
                        histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                    # eval
                    val_loss = evaluate(model, val_loader, device)  

                    if single_image:
                        img_org, img_targ, img_pred = org[0], targ[0], pred[0]
                    else:
                        img_org, img_targ, img_pred = org[0, 0], targ[0, 0], pred[0, 0]

                    logging.info(f'Validation loss: {val_loss}')
                    experiment.log({
                        'learning rate': optimizer.param_groups[0]['lr'],
                        'validation loss': val_loss,
                        'images': wandb.Image(img_org.cpu()),
                        'depth': {
                            'true': wandb.Image(img_targ.float().cpu()),
                            'pred': wandb.Image(img_pred.float().cpu())
                        },
                        'step': global_step,
                        'epoch': epoch,
                        **histograms
                    })

            # scheduler.step()

            if save_checkpoint:
                Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), str(checkpoint_dir / 'checkpoint_epoch{}.pth'.format(epoch)))
                # logging.info(f'Checkpoint {epoch} saved!')

def get_args():
    parser = argparse.ArgumentParser(description='Depth estimation for SEM images')
    parser.add_argument('--seed', type=int, default=0, help='Seed')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', '-l', metavar='LR', type=float, default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--model', type=str, default='unet', help='Model to use')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--is_transform', action='store_true', default=False, help='transform or not')
    parser.add_argument('--single_image', action='store_true', default=False, help='single imgae input')

    return parser.parse_args()

if __name__=='__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    set_seed(seed=args.seed)
    logging.info(f'Seed number {args.seed}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if args.model == 'unet':
        if args.single_image:
            model = UNet(n_channels=1, n_classes=1, bilinear=args.bilinear)
        else:
            model = UNetPlus(n_channels=1, bilinear=args.bilinear)

        logging.info(f'Network: {args.model}\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{"Bilinear" if model.bilinear else "Deconv"} upscaling\n'
                 f'\tUsing single image input: {args.single_image}'
                 )

    elif args.model == 'vit':
        model = SimpleViT(
            image_size=IMGSIZE,
            patch_size=(6, 5),
            dim=1024,
            depth=3,
            heads=8,
            mlp_dim=2048
        )

        logging.info(f'Network: {args.model}\n'
                #  f'\tpatch_size: {model.patch_size}\n'
                #  f'\tdim: {model.dim}'
                #  f'\tdepth: 3'
                #  f'\thead: 8'
                 )

    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}') 

    model.to(device=device)

    train(model=model,
          epochs=args.epochs,
          batch_size=args.batch_size,
          learning_rate=args.lr,
          is_transform=args.is_transform,
          single_image=args.single_image,
          device=device)

    # try:
    #     train(model=model,
    #         epochs=args.epochs,
    #         batch_size=args.batch_size,
    #         learning_rate=args.lr,
    #         device=device)

    # except KeyboardInterrupt:
    #     torch.save(net.state_dict(), 'INTERRUPTED.pth')
    #     logging.info('Saved interrupt')
    #     raise