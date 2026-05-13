"""V13 score evaluation. Same monkey-patch pattern as eval_v12, but uses
dataset_v13 (9-channel: V8 7ch + log10(Re) + AoA)."""

import argparse, yaml, json, os, os.path as osp
import torch
import numpy as np

import utils.metrics as metrics
from dataset.dataset_v13 import Dataset as Dataset_v13
metrics.Dataset = Dataset_v13

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='UrbanWindViT', type=str)
parser.add_argument('-t', '--task', default='full', type=str)
parser.add_argument('--my_path', required=True, type=str)
parser.add_argument('--save_path', default='metrics_v13', type=str)
parser.add_argument('--out_path', default='scores_v13', type=str)
parser.add_argument('--n_test', default=3, type=int)
args = parser.parse_args()

with open(args.my_path + '/manifest.json') as f:
    manifest = json.load(f)
manifest_train = manifest[args.task + '_train']
n = int(.1 * len(manifest_train))
train_names = manifest_train[:-n]

print(f"Loading train ({len(train_names)} cases) with V13 layout for normalization...")
_, coef_norm = Dataset_v13(train_names, norm=True, sample=None, my_path=args.my_path)
print(f"coef_norm mean_in shape = {coef_norm[0].shape} (should be 9)")

with open('params.yaml') as f:
    hparams = yaml.safe_load(f)[args.model]
override_lr = os.environ.get('OVERRIDE_LR')
if override_lr is not None:
    hparams['lr'] = float(override_lr)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
base_dir = osp.join(args.save_path, args.task, args.model)

# Try in order: main pickle, train pickle, then state_dict + re-instantiate.
candidates = [
    (osp.join(base_dir, args.model), 'main_pickle'),
    (osp.join(base_dir, 'model'), 'train_pickle'),
    (osp.join(base_dir, 'best_model.pt'), 'best_state_dict'),
]
loaded = None
for p, kind in candidates:
    if not osp.exists(p):
        print(f"  not found: {p}")
        continue
    print(f"Found {kind} at {p}, loading...")
    if kind == 'best_state_dict':
        # V13 model architecture has identity_dim=11 (V8 7ch + log10(Re) + AoA broadcast).
        # We import the V12 model class because V13 uses the SAME architecture
        # (just different data semantics). If you've defined a separate UrbanWindViT_v13
        # class, swap the import here.
        try:
            from models.UrbanWindViT_v13 import UrbanWindViT
        except ImportError:
            from models.UrbanWindViT_v12 import UrbanWindViT
        m = UrbanWindViT(
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
        ).to(device)
        sd = torch.load(p, map_location=device, weights_only=False)
        m.load_state_dict(sd)
        loaded = [m]
    else:
        obj = torch.load(p, map_location=device, weights_only=False)
        loaded = obj if isinstance(obj, list) else [obj]
    break

if loaded is None:
    raise FileNotFoundError(
        f"No usable model in {base_dir}. Expected one of: {args.model}, model, best_model.pt"
    )
for m in loaded:
    m.to(device).eval()
print(f"Loaded {len(loaded)} model(s)")

out_dir = osp.join(args.out_path, args.task)
os.makedirs(out_dir, exist_ok=True)
s = args.task + '_test' if args.task != 'scarce' else 'full_test'
print(f"Running Results_test on {s}...")
coefs = metrics.Results_test(
    device, [loaded], [hparams], coef_norm, args.my_path,
    path_out=args.out_path, n_test=args.n_test, criterion='MSE', s=s
)

np.save(osp.join(out_dir, 'true_coefs'), coefs[0])
np.save(osp.join(out_dir, 'pred_coefs_mean'), coefs[1])
np.save(osp.join(out_dir, 'pred_coefs_std'), coefs[2])
for i, f in enumerate(coefs[3]):
    np.save(osp.join(out_dir, f'true_surf_coefs_{i}'), f)
for i, f in enumerate(coefs[4]):
    np.save(osp.join(out_dir, f'surf_coefs_{i}'), f)
np.save(osp.join(out_dir, 'true_bls'), coefs[5])
np.save(osp.join(out_dir, 'bls'), coefs[6])
print(f"\ndone; outputs in {out_dir}")
