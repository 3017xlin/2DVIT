"""V12 score evaluation that monkey-patches metrics.Dataset.

`utils/metrics.py` hardcodes `from dataset.dataset import Dataset` (the 7-channel
V8 version). For V12, data.x has 9 channels (extra wall_grad), so the default
`Results_test` call breaks at model forward (shape mismatch).

This script:
  1. Imports utils.metrics
  2. Replaces metrics.Dataset with dataset_v12.Dataset
  3. Loads the trained V12 model
  4. Calls Results_test with the patched Dataset
"""

import argparse, yaml, json, os, os.path as osp
import torch
import numpy as np

import utils.metrics as metrics
from dataset.dataset_v12 import Dataset as Dataset_v12
# Monkey-patch: make metrics.Results_test use the 9-channel V12 dataset loader.
metrics.Dataset = Dataset_v12

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='UrbanWindViT', type=str)
parser.add_argument('-t', '--task', default='full', type=str)
parser.add_argument('--my_path', required=True, type=str)
parser.add_argument('--save_path', default='metrics_v12', type=str,
                    help='Where the trained model lives (must contain <task>/<model>/<model> file).')
parser.add_argument('--out_path', default='scores_v12', type=str,
                    help='Where to write score outputs.')
parser.add_argument('--n_test', default=3, type=int)
args = parser.parse_args()

# 1. coef_norm — must match training (V12 had 9-channel x)
with open(args.my_path + '/manifest.json') as f:
    manifest = json.load(f)
manifest_train = manifest[args.task + '_train']
n = int(.1 * len(manifest_train))
train_names = manifest_train[:-n]

print(f"Loading train ({len(train_names)} cases) with V12 9-channel layout for normalization...")
_, coef_norm = Dataset_v12(train_names, norm=True, sample=None, my_path=args.my_path)
print(f"coef_norm computed; mean_in shape = {coef_norm[0].shape} (should be 9)")

# 2. hparams
with open('params.yaml') as f:
    hparams = yaml.safe_load(f)[args.model]
override_lr = os.environ.get('OVERRIDE_LR')
if override_lr is not None:
    hparams['lr'] = float(override_lr)

# 3. Load trained model — try 3 places in order:
#    a) <save>/<task>/<model>/<model>  (main.py end-of-run pickle of model list)
#    b) <save>/<task>/<model>/model    (train.py end-of-training full pickle)
#    c) <save>/<task>/<model>/best_model.pt  (state_dict from best val; requires re-instantiation)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
base_dir = osp.join(args.save_path, args.task, args.model)

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
        # Re-instantiate V12 model with the SAME hparams as training, then load state_dict
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
        f"No usable model found in {base_dir}. "
        f"Expected one of: {args.model}, model, best_model.pt"
    )
for m in loaded:
    m.to(device).eval()
print(f"Loaded {len(loaded)} model(s)")

# 4. Run scoring
out_dir = osp.join(args.out_path, args.task)
os.makedirs(out_dir, exist_ok=True)
s = args.task + '_test' if args.task != 'scarce' else 'full_test'
print(f"Running Results_test on {s}...")
coefs = metrics.Results_test(
    device, [loaded], [hparams], coef_norm, args.my_path,
    path_out=args.out_path, n_test=args.n_test, criterion='MSE', s=s
)

# 5. Save outputs
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
