"""Quick OOB diagnostic: how many mesh nodes fall outside the model's grid?

Reads up to 20 cached .pt files from CACHE_DIR and reports, per case and in
aggregate, what fraction of mesh nodes lie outside the latent grid window
GRID_X x GRID_Y. If OOB is high, query points get clamped to the grid border
by F.grid_sample(padding_mode='border'), silently degrading predictions.

Run with the same Python env you use for training:
    python diag_pos_range.py
"""
import torch
import glob
import os
import numpy as np

CACHE_DIR = '/projects_vol/gp_hongying.li/yiheng/airfrans_cache'
GRID_X = (-2.0, 4.0)
GRID_Y = (-1.5, 1.5)
N_CASES_TO_INSPECT = 20

files = sorted(glob.glob(os.path.join(CACHE_DIR, 'airFoil2D_*.pt')))[:N_CASES_TO_INSPECT]
print(f"Inspecting {len(files)} cached cases from {CACHE_DIR}")

if len(files) == 0:
    print("No .pt files found. Check CACHE_DIR path.")
    raise SystemExit

oob_ratios = []
oob_surf_ratios = []
oob_vol_ratios = []
all_x_min, all_x_max = [], []
all_y_min, all_y_max = [], []
N_list = []

for f in files:
    case = torch.load(f, map_location='cpu', weights_only=False)
    pos = case['full_pos'].numpy()      # [N, 2]
    surf = case['surf'].numpy().astype(bool)  # [N]
    N = pos.shape[0]
    N_list.append(N)

    in_x = (pos[:, 0] >= GRID_X[0]) & (pos[:, 0] <= GRID_X[1])
    in_y = (pos[:, 1] >= GRID_Y[0]) & (pos[:, 1] <= GRID_Y[1])
    in_grid = in_x & in_y
    out_of = ~in_grid

    oob_ratios.append(out_of.sum() / N)
    if surf.sum() > 0:
        oob_surf_ratios.append((out_of & surf).sum() / max(surf.sum(), 1))
    if (~surf).sum() > 0:
        oob_vol_ratios.append((out_of & ~surf).sum() / max((~surf).sum(), 1))

    all_x_min.append(pos[:, 0].min())
    all_x_max.append(pos[:, 0].max())
    all_y_min.append(pos[:, 1].min())
    all_y_max.append(pos[:, 1].max())

print()
print(f"Mesh node OOB ratio (% nodes outside grid [-2,4] x [-1.5,1.5]):")
print(f"  ALL nodes  - mean: {np.mean(oob_ratios)*100:.1f}%  "
      f"min: {np.min(oob_ratios)*100:.1f}%  max: {np.max(oob_ratios)*100:.1f}%")
if oob_vol_ratios:
    print(f"  VOL nodes  - mean: {np.mean(oob_vol_ratios)*100:.1f}%  "
          f"min: {np.min(oob_vol_ratios)*100:.1f}%  max: {np.max(oob_vol_ratios)*100:.1f}%")
if oob_surf_ratios:
    print(f"  SURF nodes - mean: {np.mean(oob_surf_ratios)*100:.1f}%  "
          f"min: {np.min(oob_surf_ratios)*100:.1f}%  max: {np.max(oob_surf_ratios)*100:.1f}%")

print()
print(f"Actual position range across {len(files)} cases:")
print(f"  x: [{min(all_x_min):.2f}, {max(all_x_max):.2f}]   (grid x: [{GRID_X[0]}, {GRID_X[1]}])")
print(f"  y: [{min(all_y_min):.2f}, {max(all_y_max):.2f}]   (grid y: [{GRID_Y[0]}, {GRID_Y[1]}])")
print()
print(f"Nodes per case: mean {int(np.mean(N_list))}, "
      f"min {min(N_list)}, max {max(N_list)}")
