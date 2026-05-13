# 2DVIT — Airfoil ViT (V_best / V_best_v2)

A Vision Transformer–based surrogate for the 2D AirfRANS airfoil CFD task. The
model encodes unstructured mesh points onto a 64×64 latent grid via PointNet,
processes the grid with a ViT, and decodes target fields (pressure / velocity
components / wall shear) at arbitrary query points.

Only two model variants live here:

- **V_best** (`main_vbest.py` → `models/UrbanWindViT_vbest.py`)
  Baseline ViT pipeline. Uniform per-variable MSE loss weights.

- **V_best_v2** (`main_vbest_v2.py` → `models/UrbanWindViT_vbest_v2.py`)
  V_best + per-layer FiLM modulation on free-stream velocity `Uinf`.

## Setup

```bash
pip install -r requirements.txt
```

You also need [pytorch_geometric](https://github.com/pyg-team/pytorch_geometric).

## Data

Experiment data comes from
[AirfRANS](https://github.com/Extrality/AirfRANS) (download
[here](https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip), 9.3 GB).

Pre-process VTU cases into per-case `.pt` cache files:

```bash
python -m models.preprocess --my_path <AIRFRANS_ROOT> --task full
```

Then point `AIRFRANS_CACHE_DIR` at the cache directory so the dataloader picks
it up instead of re-parsing VTU each epoch.

## Train

```bash
# V_best
python main_vbest.py     --my_path <AIRFRANS_ROOT> --task full --save_path metrics

# V_best_v2 (FiLM-on-Uinf)
python main_vbest_v2.py  --my_path <AIRFRANS_ROOT> --task full --save_path metrics
```

Both scripts auto-run evaluation at the end of training.

## Layout

```
main_vbest.py           # V_best entry point
main_vbest_v2.py        # V_best_v2 entry point
train_vbest.py          # shared training loop + auto-eval
params.yaml             # ViT hyperparameters

dataset/
  dataset.py            # VTU-parsing dataset (slow path / fallback)
  dataset_cached.py     # reads preprocess.py .pt cache

models/
  UrbanWindViT_vbest.py    # full V_best model
  UrbanWindViT_vbest_v2.py # V_best + FiLM
  preprocess.py            # offline VTU -> .pt cache

utils/
  metrics.py            # evaluation harness
  metrics_NACA.py
  naca_generator.py
  reorganize.py
```

## Acknowledgement

Built on top of [AirfRANS](https://github.com/Extrality/AirfRANS).
