"""dataset_cached.py — drop-in replacement for dataset.dataset.Dataset that
reads the per-case `.pt` cache produced by models/preprocess.py.

Cache layout (one `.pt` per case + a global coef_norm file):
    <cache_dir>/<case_name>.pt
        keys: case_name, airfoil_pos, full_pos, full_x (raw), full_y (raw),
              surf, sdf_grid, sdf_grad_grid
    <cache_dir>/coef_norm_<task>.pt
        keys: mean_in, std_in, mean_out, std_out, task, grid_size, ...

This module exposes a `Dataset(...)` function with the SAME signature and
return shape as dataset.dataset.Dataset:
    norm=True, coef_norm=None  -> returns (list_of_pyg_Data, coef_norm_4tuple)
    coef_norm given            -> returns list_of_pyg_Data
    no normalization           -> returns list_of_pyg_Data (raw, x/y not normalized)

Routing:
    1. If `cache_dir` arg provided   -> use cache.
    2. Else if env var AIRFRANS_CACHE_DIR set -> use cache.
    3. Else                                   -> fall back to dataset.dataset.Dataset
                                                 (VTU parsing, slow path).
"""
import os
import os.path as osp

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm


_CACHE_ENV_VAR = 'AIRFRANS_CACHE_DIR'


def _resolve_cache_dir(cache_dir):
    """Return a usable cache_dir or None if neither arg nor env var is set."""
    if cache_dir is None:
        cache_dir = os.environ.get(_CACHE_ENV_VAR, None)
    if cache_dir is None:
        return None
    if not osp.isdir(cache_dir):
        print(f"[dataset_cached] cache_dir {cache_dir!r} doesn't exist — falling back to VTU.")
        return None
    return cache_dir


def _load_coef_norm(cache_dir, task):
    """Load preprocess.py-produced coef_norm_<task>.pt as a 4-tuple of numpy arrays."""
    coef_path = osp.join(cache_dir, f'coef_norm_{task}.pt')
    if not osp.exists(coef_path):
        raise FileNotFoundError(
            f"[dataset_cached] Missing {coef_path}. "
            f"Re-run preprocess.py with --task {task} to generate it."
        )
    cn = torch.load(coef_path, map_location='cpu', weights_only=False)

    def _to_np(x):
        if torch.is_tensor(x):
            return x.numpy().astype(np.float32)
        return np.asarray(x, dtype=np.float32)

    return (
        _to_np(cn['mean_in']),
        _to_np(cn['std_in']),
        _to_np(cn['mean_out']),
        _to_np(cn['std_out']),
    )


def _load_one_case(cache_dir, cname):
    """Load a single case's `.pt` cache file and return a PyG Data object (raw)."""
    cpath = osp.join(cache_dir, cname + '.pt')
    if not osp.exists(cpath):
        return None, cname
    case = torch.load(cpath, map_location='cpu', weights_only=False)
    data = Data(
        pos=case['full_pos'].float(),
        x=case['full_x'].float(),         # [N, 7] RAW (un-normalized)
        y=case['full_y'].float(),         # [N, 4] RAW
        surf=case['surf'].bool(),
        airfoil_pos=case['airfoil_pos'].float(),
    )
    return data, None


def Dataset(set, norm=False, coef_norm=None, crop=None, sample=None,
            n_boot=int(5e5), surf_ratio=.1, my_path='/data/path',
            cache_dir=None, task='full'):
    """Cached drop-in replacement for dataset.dataset.Dataset.

    Args mirror the original Dataset() signature; two new args:
        cache_dir (str|None): path to preprocess.py output. If None, reads
                              env var AIRFRANS_CACHE_DIR. If neither is set,
                              falls back to dataset.dataset.Dataset (VTU path).
        task (str): which coef_norm_<task>.pt to load. Defaults to 'full'.

    `crop`, `sample`, `n_boot`, `surf_ratio` are accepted for signature
    compatibility but only the `sample=None` path is meaningful in cache mode
    (preprocess.py uses sample=None — it caches all mesh nodes, no cell
    resampling). If `sample` is not None we fall back to the VTU path because
    cell-resampling needs the original mesh.
    """
    cache_dir = _resolve_cache_dir(cache_dir)

    # Fall back to original Dataset for cases the cache doesn't cover:
    #   - cache disabled / missing
    #   - sample != None (preprocess only handles sample=None)
    if cache_dir is None or sample is not None:
        from dataset.dataset import Dataset as _OrigDataset
        return _OrigDataset(set, norm=norm, coef_norm=coef_norm, crop=crop, sample=sample,
                            n_boot=n_boot, surf_ratio=surf_ratio, my_path=my_path)

    if norm and coef_norm is not None:
        raise ValueError(
            "If coef_norm is not None and norm is True, the normalization is fully "
            "determined by coef_norm — set norm=False to silence this."
        )

    # --- 1. Load every case in `set` from the cache ---
    dataset = []
    missing = []
    for cname in tqdm(set, desc=f"[cached] loading {len(set)} cases"):
        data, miss = _load_one_case(cache_dir, cname)
        if miss is not None:
            missing.append(miss)
            continue
        dataset.append(data)

    if missing:
        head = ", ".join(missing[:5])
        raise RuntimeError(
            f"[dataset_cached] {len(missing)} cases missing from cache (first 5: {head}). "
            f"Re-run preprocess.py to fill them in."
        )

    # --- 2. Resolve coef_norm ---
    if norm and coef_norm is None:
        coef_norm = _load_coef_norm(cache_dir, task)
        _apply_norm(dataset, coef_norm)
        return dataset, coef_norm

    if coef_norm is not None:
        _apply_norm(dataset, coef_norm)
        return dataset

    # norm=False, coef_norm=None -> return raw tensors as-is
    return dataset


def _apply_norm(dataset, coef_norm):
    """In-place normalization of data.x and data.y for every PyG Data in `dataset`."""
    mean_in, std_in, mean_out, std_out = coef_norm
    mean_in_t = torch.from_numpy(np.asarray(mean_in, dtype=np.float32))
    std_in_t = torch.from_numpy(np.asarray(std_in, dtype=np.float32))
    mean_out_t = torch.from_numpy(np.asarray(mean_out, dtype=np.float32))
    std_out_t = torch.from_numpy(np.asarray(std_out, dtype=np.float32))
    for data in dataset:
        data.x = (data.x - mean_in_t) / (std_in_t + 1e-8)
        data.y = (data.y - mean_out_t) / (std_out_t + 1e-8)
