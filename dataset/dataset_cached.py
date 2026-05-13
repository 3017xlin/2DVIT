"""dataset_cached.py — load preprocess.py v2 caches as PyG Data lists.

Schema (matches preprocess.CACHE_SCHEMA_VERSION = 2):

Per-case .pt:
    case_name, airfoil_pos [M,2], full_pos [N,2], full_y [N,4], surf [N] bool,
    uinf [2], mesh_sdf [N], mesh_sdf_grad [N,2],
    grid_sdf [G], grid_sdf_grad [G,2],
    grid_size, grid_x_range, grid_y_range, version=2

Global file (per task):
    coef_norm_<task>.pt — dict with mean/std for {uinf, sdf, out} + meta.

`Dataset(...)` signature is kept compatible with the original (positional
args), but the cache path is now mandatory: the legacy VTU fall-back has
been removed because v2 caches carry precomputed SDF + gradient that the
old per-VTU Dataset can't reconstruct.

Returns (matching the original calling convention):
    norm=True, coef_norm=None  -> (list[Data], coef_norm_dict)
    coef_norm given            -> list[Data]
    no normalization           -> list[Data] with raw tensors
"""
import os
import os.path as osp

import torch
from torch_geometric.data import Data
from tqdm import tqdm


_CACHE_ENV_VAR = 'AIRFRANS_CACHE_DIR'
CACHE_SCHEMA_VERSION = 2


def _resolve_cache_dir(cache_dir):
    """Pick cache_dir arg > env var > repo-local 'cache/'."""
    if cache_dir is None:
        cache_dir = os.environ.get(_CACHE_ENV_VAR, 'cache')
    if not osp.isdir(cache_dir):
        raise RuntimeError(
            f"cache_dir {cache_dir!r} does not exist. Run preprocess.py first."
        )
    return cache_dir


def _load_coef_norm(cache_dir, task):
    """Load and version-check the per-task coef_norm dict."""
    coef_path = osp.join(cache_dir, f'coef_norm_{task}.pt')
    if not osp.exists(coef_path):
        raise FileNotFoundError(
            f"Missing {coef_path}. Re-run preprocess.py with --task {task}."
        )
    cn = torch.load(coef_path, map_location='cpu', weights_only=False)
    if cn.get('version') != CACHE_SCHEMA_VERSION:
        raise RuntimeError(
            f"coef_norm version {cn.get('version')!r} != {CACHE_SCHEMA_VERSION}. "
            f"Re-run preprocess.py to regenerate."
        )
    return cn


def _load_one_case(cache_dir, cname):
    """Load a single case .pt into a PyG Data object (raw values, no normalization)."""
    cpath = osp.join(cache_dir, cname + '.pt')
    if not osp.exists(cpath):
        return None, cname
    case = torch.load(cpath, map_location='cpu', weights_only=False)
    if case.get('version') != CACHE_SCHEMA_VERSION:
        raise RuntimeError(
            f"{cname}: cache version {case.get('version')!r} "
            f"!= {CACHE_SCHEMA_VERSION}. Re-run preprocess.py."
        )
    data = Data(
        pos=case['full_pos'].float(),                  # [N, 2] raw physical coords
        airfoil_pos=case['airfoil_pos'].float(),       # [M, 2] polygon vertices
        surf=case['surf'].bool(),                      # [N]
        uinf=case['uinf'].float(),                     # [2]   raw
        sdf=case['mesh_sdf'].float(),                  # [N]   raw signed dist
        sdf_grad=case['mesh_sdf_grad'].float(),        # [N, 2] unit gradient (raw)
        grid_sdf=case['grid_sdf'].float(),             # [G]
        grid_sdf_grad=case['grid_sdf_grad'].float(),   # [G, 2]
        y=case['full_y'].float(),                      # [N, 4] raw targets
    )
    return data, None


def _apply_norm(dataset, coef_norm):
    """In-place z-score for uinf, sdf, grid_sdf, y.

    NOT normalized (intentional):
      - pos:            raw physical, used by Fourier via decoder.physical_to_norm
      - sdf_grad, grid_sdf_grad: unit vectors by construction (eikonal); z-score
                                  would destroy that
      - airfoil_pos:    raw, used in case anything wants the polygon directly
      - surf:           boolean
    """
    mean_uinf = coef_norm['mean_uinf']
    std_uinf  = coef_norm['std_uinf']
    mean_sdf  = coef_norm['mean_sdf']
    std_sdf   = coef_norm['std_sdf']
    mean_out  = coef_norm['mean_out']
    std_out   = coef_norm['std_out']

    eps = 1e-8
    for data in dataset:
        data.uinf     = (data.uinf - mean_uinf) / (std_uinf + eps)
        data.sdf      = (data.sdf - mean_sdf) / (std_sdf + eps)
        data.grid_sdf = (data.grid_sdf - mean_sdf) / (std_sdf + eps)
        data.y        = (data.y - mean_out) / (std_out + eps)


def Dataset(set, norm=False, coef_norm=None, crop=None, sample=None,
            n_boot=int(5e5), surf_ratio=.1, my_path='/data/path',
            cache_dir=None, task='full'):
    """Cache-only Dataset (v2 schema).

    Args mirror the original dataset.dataset.Dataset for drop-in compatibility,
    but `crop`, `sample`, `n_boot`, `surf_ratio` are ignored in cache mode
    (preprocess.py always stores the full mesh, no per-load resampling).

    Args:
        set (list[str]): list of case names to load.
        norm (bool): if True, return (data_list, coef_norm) with normalization applied.
        coef_norm (dict|None): if given, apply this normalization (no return of stats).
        my_path (str): AirfRANS data root (unused in cache path, kept for signature compat).
        cache_dir (str|None): defaults to env var AIRFRANS_CACHE_DIR.
        task (str): which coef_norm_<task>.pt to load (only used if norm=True).

    Returns:
        same shape as dataset.dataset.Dataset.
    """
    cache_dir = _resolve_cache_dir(cache_dir)

    if sample is not None:
        raise NotImplementedError(
            "sample!=None is not supported by the v2 cache path "
            "(preprocess.py always stores full mesh nodes)."
        )
    if norm and coef_norm is not None:
        raise ValueError(
            "norm=True together with a provided coef_norm is ambiguous; "
            "pass one or the other."
        )

    # --- 1. load each case from cache ---
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
            f"{len(missing)} cases missing from cache (first 5: {head}). "
            f"Re-run preprocess.py to fill them in."
        )

    # --- 2. resolve normalization ---
    if norm and coef_norm is None:
        coef_norm = _load_coef_norm(cache_dir, task)
        _apply_norm(dataset, coef_norm)
        return dataset, coef_norm

    if coef_norm is not None:
        _apply_norm(dataset, coef_norm)
        return dataset

    # norm=False, coef_norm=None → return raw tensors as-is
    return dataset
