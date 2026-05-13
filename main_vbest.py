"""V_best main: V8 architecture with UNIFORM loss weights (AirfRANS protocol).

Default pointnet_scales = ((0.15, 32), (0.5, 64)) — same as V8.
The only V_best change vs V8 is in train_vbest.py: weights [1,1,1,1] not [1,1,1.5,0.3].
"""
import argparse, yaml, json, os
import torch
import train_vbest as train  # V_best: uniform weights + auto-eval at end
# Cached loader: reads preprocess.py output if AIRFRANS_CACHE_DIR env var is set,
# otherwise falls back to the original VTU-parsing Dataset.
from dataset.dataset_cached import Dataset
import os.path as osp
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='UrbanWindViT')
parser.add_argument('-n', '--nmodel', default=1, type=int)
parser.add_argument('-w', '--weight', default=1, type=float)
parser.add_argument('-t', '--task', default='full', type=str)
parser.add_argument('-s', '--score', default=0, type=int)
parser.add_argument('--my_path', default='/data/path', type=str)
parser.add_argument('--save_path', default='metrics', type=str)
args = parser.parse_args()

with open(args.my_path + '/manifest.json', 'r') as f:
    manifest = json.load(f)

manifest_train = manifest[args.task + '_train']
n = int(.1 * len(manifest_train))
train_names = manifest_train[:-n]
val_names = manifest_train[-n:]
print("start load data")
train_dataset, coef_norm = Dataset(train_names, norm=True, sample=None,
                                   my_path=args.my_path, task=args.task)
val_dataset = Dataset(val_names, sample=None, coef_norm=coef_norm,
                      my_path=args.my_path, task=args.task)
print("load data finish")

use_cuda = torch.cuda.is_available()
device = 'cuda:0' if use_cuda else 'cpu'
print('Using ' + ('GPU' if use_cuda else 'CPU'))

with open('params.yaml', 'r') as f:
    hparams = yaml.safe_load(f)[args.model]

override_lr = os.environ.get('OVERRIDE_LR')
if override_lr is not None:
    hparams['lr'] = float(override_lr)
    print(f"OVERRIDE_LR: hparams['lr'] = {hparams['lr']}")

models = []
for i in range(args.nmodel):
    if args.model == 'UrbanWindViT':
        from models.UrbanWindViT_vbest import UrbanWindViT
        model = UrbanWindViT(
            grid_size=64,
            grid_x_range=(-2.0, 4.0),
            grid_y_range=(-1.5, 1.5),
            pointnet_scales=((0.15, 32), (0.5, 64)),   # V_best default radii
            latent_dim=256,
            patch_size=2,
            num_layers=5,
            num_heads=8,
            ffn_hidden=1024,
            fourier_freqs=10,
            out_dim=4,
            dropout=0.1,
        ).cuda()
    else:
        raise ValueError(f"main_vbest only supports UrbanWindViT, got {args.model}")

    log_path = osp.join(args.save_path, args.task, args.model)
    print('start training')
    model = train.main(
        device, train_dataset, val_dataset, model, hparams, log_path,
        criterion='MSE_weighted', val_iter=10, reg=args.weight,
        name_mod=args.model, val_sample=True,
        # AUTO-EVAL hookup
        auto_eval=True,
        my_path=args.my_path,
        task=args.task,
        coef_norm=coef_norm,
        DatasetClass=Dataset,
        save_path=args.save_path,
    )
    print('end training')
    models.append(model)

torch.save(models, osp.join(args.save_path, args.task, args.model, args.model))
