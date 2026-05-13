"""
preprocess.py — one-shot precomputation of static, non-learnable case data.

Reads the AirfRANS pyvista files once, extracts everything that does not
depend on per-epoch random subsampling, and writes a small ``.pt`` per case
to a cache directory. The training loop (``train_v15.py``) then reads these
caches and never touches pyvista.

Per-case cached tensors:
    airfoil_pos     [M, 2]   airfoil polygon vertices, physical coords
    full_pos        [N, 2]   all CFD mesh node positions
    full_x          [N, 7]   raw input features [pos_x, pos_y, uinf_x, uinf_y, sdf, n_x, n_y]
    full_y          [N, 4]   raw targets [u, v, p, nut]
    surf            [N]      bool, True on airfoil surface
    sdf_grid        [G]      signed distance from latent grid points to airfoil polygon
    sdf_grad_grid   [G, 2]   finite-difference SDF gradient on the grid

Global file:
    coef_norm_<task>.pt — (mean_in, std_in, mean_out, std_out) computed
    over the training cases of the requested task. Same convention as
    dataset/dataset.py.

Usage:
    python preprocess.py \\
        --my_path /data/path/to/AirfRANS \\
        --cache_dir /scratch/airfrans_cache \\
        --task full \\
        --grid_size 64 \\
        --grid_x_range -2 4 \\
        --grid_y_range -1.5 1.5
"""

import argparse
import json
import multiprocessing as mp
import os
import os.path as osp
import sys

import numpy as np
import torch
from tqdm import tqdm

# ``utils.reorganize`` lives next to this script in the AirfRANS layout.
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from utils.reorganize import reorganize


# Module-level globals populated by the worker initializer (Linux fork copies
# the parent's memory, so big tensors are shared without per-task pickle cost).
_W_GRID_COORDS = None
_W_GRID_SIZE = None
_W_DX = None
_W_DY = None
_W_MY_PATH = None
_W_CACHE_DIR = None
_W_SKIP_EXISTING = None


def _init_worker(grid_coords, grid_size, dx, dy, my_path, cache_dir, skip_existing):
    global _W_GRID_COORDS, _W_GRID_SIZE, _W_DX, _W_DY
    global _W_MY_PATH, _W_CACHE_DIR, _W_SKIP_EXISTING
    _W_GRID_COORDS = grid_coords
    _W_GRID_SIZE = grid_size
    _W_DX = dx
    _W_DY = dy
    _W_MY_PATH = my_path
    _W_CACHE_DIR = cache_dir
    _W_SKIP_EXISTING = skip_existing
    # Keep numpy / torch from spawning its own threads inside each worker —
    # we already parallelize at the case level, internal threading just
    # contends for the same CPU.
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _worker_process_case(cname):
    out_path = osp.join(_W_CACHE_DIR, cname + ".pt")
    if _W_SKIP_EXISTING and osp.exists(out_path):
        return ("skipped", cname, None)
    try:
        case = process_case(
            cname, _W_MY_PATH, _W_GRID_COORDS, _W_GRID_SIZE, _W_DX, _W_DY,
        )
        torch.save(case, out_path)
        return ("ok", cname, None)
    except Exception as exc:                                  # noqa: BLE001
        return ("failed", cname, str(exc))


# --------------------------------------------------------------------------- #
# Static helpers (SDF + gradient on the latent grid)
# --------------------------------------------------------------------------- #


def build_grid_coords(grid_size, x_range, y_range):
    """Return ``[grid_size**2, 2]`` row-major grid coordinates."""
    x = torch.linspace(x_range[0], x_range[1], grid_size)
    y = torch.linspace(y_range[0], y_range[1], grid_size)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(-1, 2).float()


def signed_distance_2d(grid, polygon):
    """Signed distance from each grid point to a closed polygon.

    Negative inside, positive outside. Sign determined by horizontal ray
    casting (toward +x).

    Args:
        grid:    ``[G, 2]``
        polygon: ``[M, 2]`` ordered vertices.
    """
    dist = torch.cdist(grid, polygon).min(dim=1).values  # [G]

    p1 = polygon
    p2 = torch.roll(polygon, -1, dims=0)

    gx = grid[:, 0:1]
    gy = grid[:, 1:2]

    cond_y = (p1[:, 1][None, :] > gy) != (p2[:, 1][None, :] > gy)         # [G, M]
    denom = p2[:, 1][None, :] - p1[:, 1][None, :]
    x_intersect = p1[:, 0][None, :] + (gy - p1[:, 1][None, :]) * (
        p2[:, 0][None, :] - p1[:, 0][None, :]
    ) / (denom + 1e-12)
    cond_x = x_intersect > gx
    crossings = (cond_y & cond_x).sum(dim=1)
    inside = (crossings % 2) == 1

    return torch.where(inside, -dist, dist)


def sdf_gradient_2d(sdf_2d, dx, dy):
    """Central finite-difference gradient on a [H, W] grid (1-sided at edges).

    Returns ``[H, W, 2]`` with components ordered (dSDF/dx, dSDF/dy).
    """
    H, W = sdf_2d.shape
    grad_x = torch.zeros_like(sdf_2d)
    grad_y = torch.zeros_like(sdf_2d)

    if W >= 3:
        grad_x[:, 1:-1] = (sdf_2d[:, 2:] - sdf_2d[:, :-2]) / (2.0 * dx)
    if H >= 3:
        grad_y[1:-1, :] = (sdf_2d[2:, :] - sdf_2d[:-2, :]) / (2.0 * dy)

    if W >= 2:
        grad_x[:, 0] = (sdf_2d[:, 1] - sdf_2d[:, 0]) / dx
        grad_x[:, -1] = (sdf_2d[:, -1] - sdf_2d[:, -2]) / dx
    if H >= 2:
        grad_y[0, :] = (sdf_2d[1, :] - sdf_2d[0, :]) / dy
        grad_y[-1, :] = (sdf_2d[-1, :] - sdf_2d[-2, :]) / dy

    return torch.stack([grad_x, grad_y], dim=-1)


# --------------------------------------------------------------------------- #
# Per-case extraction (matches dataset/dataset.py with sample=None)
# --------------------------------------------------------------------------- #


def process_case(case_name, my_path, grid_coords, grid_size, dx, dy):
    """Extract one case from pyvista files. Returns a dict of CPU tensors."""
    import pyvista as pv  # imported lazily so the script can be inspected without pv installed

    internal = pv.read(osp.join(my_path, case_name, case_name + "_internal.vtu"))
    aerofoil = pv.read(osp.join(my_path, case_name, case_name + "_aerofoil.vtp"))

    airfoil_pos = torch.tensor(aerofoil.points[:, :2], dtype=torch.float)

    surf_bool = internal.point_data["U"][:, 0] == 0
    geom = -internal.point_data["implicit_distance"][:, None]  # SDF (positive outside)

    # Inlet velocity is encoded in the case name: AIRFOIL_AOA_UINF_..._XXX
    parts = case_name.split("_")
    Uinf = float(parts[2])
    alpha = float(parts[3]) * np.pi / 180.0

    u = (
        np.array([np.cos(alpha), np.sin(alpha)]) * Uinf
    ).reshape(1, 2) * np.ones_like(internal.point_data["U"][:, :1])
    normal = np.zeros_like(u)
    normal[surf_bool] = reorganize(
        aerofoil.points[:, :2],
        internal.points[surf_bool, :2],
        -aerofoil.point_data["Normals"][:, :2],
    )

    pos = internal.points[:, :2]
    init = np.concatenate([pos, u, geom, normal], axis=1).astype(np.float32)  # [N, 7]

    target = np.concatenate(
        [
            internal.point_data["U"][:, :2],
            internal.point_data["p"][:, None],
            internal.point_data["nut"][:, None],
        ],
        axis=-1,
    ).astype(np.float32)  # [N, 4]

    full_pos = torch.from_numpy(pos.astype(np.float32))
    full_x = torch.from_numpy(init)
    full_y = torch.from_numpy(target)
    surf = torch.from_numpy(surf_bool).bool()

    # SDF on the latent grid (computed against the airfoil polygon).
    sdf_grid = signed_distance_2d(grid_coords, airfoil_pos)            # [G]
    sdf_grid_2d = sdf_grid.reshape(grid_size, grid_size)
    sdf_grad_2d = sdf_gradient_2d(sdf_grid_2d, dx, dy)                 # [H, W, 2]
    sdf_grad_grid = sdf_grad_2d.reshape(-1, 2)                         # [G, 2]

    return {
        "case_name": case_name,
        "airfoil_pos": airfoil_pos,
        "full_pos": full_pos,
        "full_x": full_x,
        "full_y": full_y,
        "surf": surf,
        "sdf_grid": sdf_grid,
        "sdf_grad_grid": sdf_grad_grid,
    }


# --------------------------------------------------------------------------- #
# Coef norm (input/output mean/std over training cases)
# --------------------------------------------------------------------------- #


def compute_coef_norm(cache_dir, train_cases):
    """Compute (mean_in, std_in, mean_out, std_out) over training cases.

    Streamed (Welford-style) so we never hold the full concatenation in memory.
    Same convention as dataset/dataset.py — running averages weighted by N.
    """
    print(f"computing coef_norm over {len(train_cases)} training cases...")

    mean_in = None
    mean_out = None
    n_seen = 0
    for cname in tqdm(train_cases, desc="coef_norm pass 1 (means)"):
        cache_path = osp.join(cache_dir, cname + ".pt")
        case = torch.load(cache_path, map_location="cpu")
        x = case["full_x"].numpy().astype(np.float64)
        y = case["full_y"].numpy().astype(np.float64)
        n = x.shape[0]
        if mean_in is None:
            mean_in = x.mean(axis=0)
            mean_out = y.mean(axis=0)
            n_seen = n
        else:
            new_n = n_seen + n
            mean_in += (x.sum(axis=0) - n * mean_in) / new_n
            mean_out += (y.sum(axis=0) - n * mean_out) / new_n
            n_seen = new_n

    var_in = None
    var_out = None
    n_seen = 0
    for cname in tqdm(train_cases, desc="coef_norm pass 2 (vars)"):
        cache_path = osp.join(cache_dir, cname + ".pt")
        case = torch.load(cache_path, map_location="cpu")
        x = case["full_x"].numpy().astype(np.float64)
        y = case["full_y"].numpy().astype(np.float64)
        n = x.shape[0]
        if var_in is None:
            var_in = ((x - mean_in) ** 2).sum(axis=0) / n
            var_out = ((y - mean_out) ** 2).sum(axis=0) / n
            n_seen = n
        else:
            new_n = n_seen + n
            var_in += (((x - mean_in) ** 2).sum(axis=0) - n * var_in) / new_n
            var_out += (((y - mean_out) ** 2).sum(axis=0) - n * var_out) / new_n
            n_seen = new_n

    std_in = np.sqrt(var_in).astype(np.float32)
    std_out = np.sqrt(var_out).astype(np.float32)
    return (
        mean_in.astype(np.float32),
        std_in,
        mean_out.astype(np.float32),
        std_out,
    )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--my_path", required=True, help="AirfRANS data root.")
    ap.add_argument(
        "--cache_dir", required=True, help="Output dir for per-case .pt files."
    )
    ap.add_argument(
        "--task",
        default="full",
        choices=["full", "scarce", "reynolds", "aoa"],
        help="Used to compute coef_norm against the right train manifest.",
    )
    ap.add_argument("--grid_size", type=int, default=64)
    ap.add_argument("--grid_x_range", type=float, nargs=2, default=[-2.0, 4.0])
    ap.add_argument("--grid_y_range", type=float, nargs=2, default=[-1.5, 1.5])
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="Don't reprocess cases whose .pt already exists.",
    )
    ap.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Parallel pyvista workers. Set to 1 to disable multiprocessing "
             "(useful for debugging or when CPU-bound by other steps). "
             "Match this to your allocated CPU count.",
    )
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    grid_coords = build_grid_coords(args.grid_size, args.grid_x_range, args.grid_y_range)
    dx = (args.grid_x_range[1] - args.grid_x_range[0]) / (args.grid_size - 1)
    dy = (args.grid_y_range[1] - args.grid_y_range[0]) / (args.grid_size - 1)

    with open(osp.join(args.my_path, "manifest.json")) as f:
        manifest = json.load(f)

    # Process every case the manifest references for this task plus its test set.
    if args.task == "scarce":
        train_cases = manifest["scarce_train"]
        test_cases = manifest["full_test"]
    else:
        train_cases = manifest[args.task + "_train"]
        test_cases = manifest[args.task + "_test"]
    all_cases = list(dict.fromkeys(train_cases + test_cases))

    print(f"task={args.task}: train={len(train_cases)}, test={len(test_cases)}, "
          f"total={len(all_cases)}")
    print(f"writing per-case caches to: {args.cache_dir}")
    print(f"grid: {args.grid_size}x{args.grid_size}, x∈{args.grid_x_range}, "
          f"y∈{args.grid_y_range}, dx={dx:.4f}, dy={dy:.4f}")
    print(f"num_workers: {args.num_workers}")

    failures = []
    if args.num_workers <= 1:
        # Single process — original sequential path.
        for cname in tqdm(all_cases, desc="cases"):
            out_path = osp.join(args.cache_dir, cname + ".pt")
            if args.skip_existing and osp.exists(out_path):
                continue
            try:
                case = process_case(cname, args.my_path, grid_coords, args.grid_size, dx, dy)
                torch.save(case, out_path)
            except Exception as exc:                          # noqa: BLE001
                print(f"!! failed {cname}: {exc}")
                failures.append((cname, str(exc)))
    else:
        # Multiprocessing path. Linux fork() shares grid_coords by COW, so the
        # initargs cost is only paid once per worker, not per task.
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=args.num_workers,
            initializer=_init_worker,
            initargs=(grid_coords, args.grid_size, dx, dy, args.my_path,
                      args.cache_dir, args.skip_existing),
        ) as pool:
            for status, cname, msg in tqdm(
                pool.imap_unordered(_worker_process_case, all_cases, chunksize=4),
                total=len(all_cases),
                desc="cases",
            ):
                if status == "failed":
                    print(f"!! failed {cname}: {msg}")
                    failures.append((cname, msg))

    if failures:
        print(f"{len(failures)} cases failed, see failures_{args.task}.txt")
        with open(osp.join(args.cache_dir, f"failures_{args.task}.txt"), "w") as f:
            for cname, msg in failures:
                f.write(f"{cname}\t{msg}\n")

    # Compute coef_norm against the (successfully processed) training cases.
    train_cases_ok = [
        c for c in train_cases if osp.exists(osp.join(args.cache_dir, c + ".pt"))
    ]
    if not train_cases_ok:
        raise RuntimeError("no training cases were successfully cached.")

    mean_in, std_in, mean_out, std_out = compute_coef_norm(args.cache_dir, train_cases_ok)
    coef_path = osp.join(args.cache_dir, f"coef_norm_{args.task}.pt")
    torch.save(
        {
            "mean_in": torch.from_numpy(mean_in),
            "std_in": torch.from_numpy(std_in),
            "mean_out": torch.from_numpy(mean_out),
            "std_out": torch.from_numpy(std_out),
            "task": args.task,
            "grid_size": args.grid_size,
            "grid_x_range": list(args.grid_x_range),
            "grid_y_range": list(args.grid_y_range),
        },
        coef_path,
    )
    print(f"wrote {coef_path}")
    print("done.")


if __name__ == "__main__":
    main()
