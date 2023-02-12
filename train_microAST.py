import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
import wandb
from torchvision import transforms

import net_microAST as net
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images',
                    default='dataset/ms-coco') 
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images',
                    default='dataset/wikiart') 

parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--sample_path', type=str, default='samples', 
                    help='Derectory to save the intermediate samples')

# training options
parser.add_argument('--save_dir', default='./exp',
                    help='Directory to save the models')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=120000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=3.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--SSC_weight', type=float, default=3.0)
parser.add_argument('--TV_weight', type=float, default=1e-5)
parser.add_argument('--n_threads', type=int, default=12)
parser.add_argument('--log_steps', type=int, default=40)
parser.add_argument('--save_model_interval', type=int, default=5000)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--resume', action='store_true', help='train the model from the checkpoint')
parser.add_argument('--dec_tuned', action='store_true', help='use the modified decoder')
parser.add_argument('--checkpoints', default='./checkpoints',
                    help='Directory to save the checkpoint')
parser.add_argument('--loss_func', default='mse',
                    help='Loss function along the VGG. "mse", "cos" or "l1')
parser.add_argument('--ckpt_path', type=str, default='./checkpoints/last.ckpt', 
                    help='Path to the checkpoint to resume training')
args = parser.parse_args()


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*.jpg')) + list(Path(self.root).glob('*.png'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

class ContentStyleDataset(data.Dataset):
  def __init__(self, content_dataset, style_dataset, transform):
    super(ContentStyleDataset, self).__init__()
    self.content_dataset = list(Path(content_dataset).glob('*.jpg')) + list(Path(content_dataset).glob('*.png'))
    self.style_dataset = list(Path(style_dataset).glob('*.jpg')) + list(Path(style_dataset).glob('*.png'))
    self.transform = transform

  def __getitem__(self, index):
    content_img = self.content_dataset[index % len(self.content_dataset)]
    style_img = self.style_dataset[index % len(self.style_dataset)]

    content_img = Image.open(str(content_img)).convert('RGB')
    content_img = self.transform(content_img)

    style_img = Image.open(str(style_img)).convert('RGB')
    style_img = self.transform(style_img)
    
    return content_img, style_img

  def __len__(self):
    return max(len(self.content_dataset), len(self.style_dataset))

  def name(self):
    return 'ContentStyleDataset'

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
  from pytorch_lightning.callbacks import ModelCheckpoint
  from net_microAST import LogPredictionsCallback
  from pytorch_lightning.tuner.tuning import Tuner

  save_dir = Path(args.save_dir)
  save_dir.mkdir(exist_ok=True, parents=True)
  log_dir = Path(args.log_dir)
  log_dir.mkdir(exist_ok=True, parents=True)
  checkpoints_dir = Path(args.checkpoints)
  checkpoints_dir.mkdir(exist_ok=True, parents=True)

  if(args.resume):
    checkpoint_path = Path(args.ckpt_path)
  
  wandb_logger = WandbLogger(project='microAST', name='dec-tuned', log_model="all", save_dir=log_dir, save_code=False)

  wandb.login()

  wandb.run.log_code(".") # log code

  vgg = net.vgg
  vgg.load_state_dict(torch.load(args.vgg))
  vgg = nn.Sequential(*list(vgg.children())[:31])

  content_encoder = net.Encoder()
  style_encoder = net.Encoder()
  modulator = net.Modulator()
  if(args.dec_tuned):
    decoder = net.DecoderTuned()
  else:
    decoder = net.Decoder()

  transform = train_transform()

  dataset = ContentStyleDataset(args.content_dir, args.style_dir, transform)
  
  network = net.Net(vgg, content_encoder, style_encoder, modulator, decoder, args.lr, args.lr_decay,\
    style_weight=args.style_weight, \
    content_weight=args.content_weight, \
    SSC_weight=args.SSC_weight, \
    TV_weight=args.TV_weight, \
    train_dataset=dataset, n_workers=args.n_threads, log_steps=args.log_steps, loss_func=args.loss_func)

  wandb_logger.watch(network)

  trainer = pl.Trainer(devices=1, limit_train_batches=args.max_iter, max_epochs=1, precision=16, accelerator="gpu", callbacks = \
      [LogPredictionsCallback(), ModelCheckpoint(dirpath=checkpoints_dir, save_top_k=-1, every_n_train_steps=50, verbose=True), net.CustomModelCheckpoint(dirpath=checkpoints_dir,  every_n_train_steps=5000)], \
        logger=wandb_logger)

  # tune hyperparameters
  tuner = Tuner(trainer)
  tuner.scale_batch_size(network, mode='power', init_val=8, max_trials=0)

  # training
  if(args.resume):
    trainer.fit(network, ckpt_path=checkpoint_path)
  else:
    trainer.fit(network)

  wandb.finish()

if __name__ == '__main__':
    main()