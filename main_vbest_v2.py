"""V_best_v2 main: V_best + FiLM modulation on Uinf in each ViT layer.

Change vs V_best: model is UrbanWindViT_vbest_v2, which has a 5-layer ViT
where each TransformerBlock applies a learned FiLM affine transformation
(gamma, beta from Linear(2, 2*dim)) conditioned on the case-level Uinf
after the FFN residual. Adds ~10K parameters total.

Training pipeline, loss weights, dataset, all identical to V_best.
"""
import argparse, yaml, json, os
import torch
import train_vbest as train
# Cached loader (preprocess.py output via AIRFRANS_CACHE_DIR env var).
from dataset.dataset_cached import Dataset
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='UrbanWindViT')
parser.add_argument('-n', '--nmodel', default=1, type=int)
parser.add_argument('-w', '--weight', default=1, type=float)
parser.add_argument('-t', '--task', default='full', type=str)
parser.add_argument('-s', '--score', default=0, type=int)
parser.add_argument('--my_path', default='data/Dataset', type=str)
parser.add_argument('--save_path', default='metrics', type=str)
args = parser.parse_args()

with open(args.my_path + '/manifest.json', 'r') as f:
    manifest = json.load(f)
manifest_train = manifest[args.task + '_train']
train_names = manifest_train
print("start load data")
train_dataset, coef_norm = Dataset(train_names, norm=True, sample=None,
                                   my_path=args.my_path, task=args.task)
print("load data finish")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

with open('params.yaml', 'r') as f:
    hparams = yaml.safe_load(f)[args.model]

override_lr = os.environ.get('OVERRIDE_LR')
if override_lr is not None:
    hparams['lr'] = float(override_lr)
    print(f"OVERRIDE_LR: hparams['lr'] = {hparams['lr']}")

models = []
for i in range(args.nmodel):
    from models.UrbanWindViT_vbest_v2 import UrbanWindViT
    # Cache is the source of truth for grid params; see main_vbest.py note.
    grid_size = int(coef_norm['grid_size'])
    grid_x_range = tuple(coef_norm['grid_x_range'])
    grid_y_range = tuple(coef_norm['grid_y_range'])
    model = UrbanWindViT(
        grid_size=grid_size,
        grid_x_range=grid_x_range,
        grid_y_range=grid_y_range,
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

    log_path = osp.join(args.save_path, args.task, args.model)
    print('start training')
    model = train.main(
        device, train_dataset, model, hparams, log_path,
        criterion='MSE_weighted', reg=args.weight,
        name_mod=args.model,
        auto_eval=True, my_path=args.my_path, task=args.task,
        coef_norm=coef_norm, DatasetClass=Dataset, save_path=args.save_path,
    )
    print('end training')
    models.append(model)

torch.save(models, osp.join(args.save_path, args.task, args.model, args.model))
