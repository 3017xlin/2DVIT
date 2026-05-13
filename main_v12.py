"""V12 main: V8 + per-point exact wall-gradient feature.

Single change vs V8: import dataset.dataset_v12.Dataset (which appends
exact KDTree-computed wall-grad to data.x) and import models.UrbanWindViT_v12
(decoder identity has 11 channels instead of 9). Loss weights and training
loop are unchanged from V8.
"""
import argparse, yaml, json, os
import torch
import train_v8 as train  # V12 reuses V8 training (no loss change)
import utils.metrics as metrics
from dataset.dataset_v12 import Dataset
import os.path as osp
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
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
train_dataset, coef_norm = Dataset(train_names, norm=True, sample=None, my_path=args.my_path)
val_dataset = Dataset(val_names, sample=None, coef_norm=coef_norm, my_path=args.my_path)
print("load data finish")

use_cuda = torch.cuda.is_available()
device = 'cuda:0' if use_cuda else 'cpu'
if use_cuda:
    print('Using GPU')
else:
    print('Using CPU')

with open('params.yaml', 'r') as f:
    hparams = yaml.safe_load(f)[args.model]

override_lr = os.environ.get('OVERRIDE_LR')
if override_lr is not None:
    hparams['lr'] = float(override_lr)
    print(f"OVERRIDE_LR: hparams['lr'] = {hparams['lr']}")

models = []
for i in range(args.nmodel):
    if args.model == 'UrbanWindViT':
        from models.UrbanWindViT_v12 import UrbanWindViT
        model = UrbanWindViT(
            grid_size=64,
            grid_x_range=(-2.0, 4.0),
            grid_y_range=(-1.5, 1.5),
            pointnet_scales=((0.15, 32), (0.5, 64)),
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
        raise ValueError(f"main_v12 only supports UrbanWindViT, got {args.model}")

    log_path = osp.join(args.save_path, args.task, args.model)
    print('start training')
    model = train.main(device, train_dataset, val_dataset, model, hparams, log_path,
                       criterion='MSE_weighted', val_iter=10, reg=args.weight, name_mod=args.model, val_sample=True)
    print('end training')
    models.append(model)
torch.save(models, osp.join(args.save_path, args.task, args.model, args.model))

if bool(args.score):
    print('start score')
    s = args.task + '_test' if args.task != 'scarce' else 'full_test'
    coefs = metrics.Results_test(device, [models], [hparams], coef_norm, args.my_path, path_out='scores', n_test=3,
                                 criterion='MSE', s=s)
    np.save(osp.join('scores', args.task, 'true_coefs'), coefs[0])
    np.save(osp.join('scores', args.task, 'pred_coefs_mean'), coefs[1])
    np.save(osp.join('scores', args.task, 'pred_coefs_std'), coefs[2])
    for n, file in enumerate(coefs[3]):
        np.save(osp.join('scores', args.task, 'true_surf_coefs_' + str(n)), file)
    for n, file in enumerate(coefs[4]):
        np.save(osp.join('scores', args.task, 'surf_coefs_' + str(n)), file)
    np.save(osp.join('scores', args.task, 'true_bls'), coefs[5])
    np.save(osp.join('scores', args.task, 'bls'), coefs[6])
    print('end score')
