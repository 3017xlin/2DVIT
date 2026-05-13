"""preprocess.py — one-shot precomputation of static, non-learnable case data.

Version 2: analytic polygon-edge SDF for both mesh nodes and the latent grid,
plus the eikonal gradient. Drops OpenFOAM's discretized implicit_distance and
the finite-difference grid gradient.

Per-case cached tensors (one .pt per case):
    case_name      str        AirfRANS case identifier
    airfoil_pos    [M, 2]     airfoil polygon vertices (ordered, closed)
    full_pos       [N, 2]     all CFD mesh node positions
    full_y         [N, 4]     raw targets (Ux, Uy, p, nut)
    surf           [N] bool   airfoil surface mask
    uinf           [2]        case-constant free-stream velocity (Ux_inf, Uy_inf)
    mesh_sdf       [N]        signed distance, mesh nodes (~0 on surface, >0 outside)
    mesh_sdf_grad  [N, 2]     unit gradient at mesh nodes (= outward surface normal)
    grid_sdf       [G]        signed distance on grid_size**2 latent grid
    grid_sdf_grad  [G, 2]     unit gradient on the grid (correctly signed inside)
    grid_size, grid_x_range, grid_y_range  — for sanity check at load time
    version        int        schema version (current: 2)

Global file:
    coef_norm_<task>.pt — dict with mean/std for uinf (per axis), sdf (scalar,
    pooled over mesh+grid), output (per channel). Used at train time for z-score.

Usage:
    python -m models.preprocess \\
        --my_path /data/path/to/AirfRANS \\
        --cache_dir /scratch/airfrans_cache \\
        --task full
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import os.path as osp
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from utils.reorganize import reorganize


CACHE_SCHEMA_VERSION = 2


# Worker globals (populated by fork-time initializer; reused across all tasks
# in that worker so we don't pay per-task pickle cost).
_W_GRID_COORDS = None
_W_GRID_SIZE = None
_W_GRID_X_RANGE = None
_W_GRID_Y_RANGE = None
_W_MY_PATH = None
_W_CACHE_DIR = None
_W_SKIP_EXISTING = None


def _init_worker(grid_coords, grid_size, grid_x_range, grid_y_range,
                 my_path, cache_dir, skip_existing):
    global _W_GRID_COORDS, _W_GRID_SIZE, _W_GRID_X_RANGE, _W_GRID_Y_RANGE
    global _W_MY_PATH, _W_CACHE_DIR, _W_SKIP_EXISTING
    _W_GRID_COORDS = grid_coords
    _W_GRID_SIZE = grid_size
    _W_GRID_X_RANGE = grid_x_range
    _W_GRID_Y_RANGE = grid_y_range
    _W_MY_PATH = my_path
    _W_CACHE_DIR = cache_dir
    _W_SKIP_EXISTING = skip_existing
    # Stop torch/openmp from spawning extra threads inside each worker — we
    # already parallelize at the case level.
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _worker_process_case(cname):
    out_path = osp.join(_W_CACHE_DIR, cname + ".pt")
    if _W_SKIP_EXISTING and osp.exists(out_path):
        return ("skipped", cname, None)
    try:
        case = process_case(
            cname, _W_MY_PATH, _W_GRID_COORDS, _W_GRID_SIZE,
            _W_GRID_X_RANGE, _W_GRID_Y_RANGE,
        )
        torch.save(case, out_path)
        return ("ok", cname, None)
    except Exception as exc:                                  # noqa: BLE001
        return ("failed", cname, str(exc))


# --------------------------------------------------------------------------- #
# Geometry helpers — analytic 2D polygon SDF (no Open3D, no FD)
# --------------------------------------------------------------------------- #


def build_grid_coords(grid_size, x_range, y_range):
    """Row-major [grid_size**2, 2] grid coordinates over the physical window."""
    x = torch.linspace(x_range[0], x_range[1], grid_size)
    y = torch.linspace(y_range[0], y_range[1], grid_size)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(-1, 2).float()


def polygon_sdf_and_grad(query, polygon, chunk_size=20000):
    """Signed distance and unit gradient from each query to a closed 2D polygon.

    Geometry:
      - distance: minimum point-to-EDGE distance (not vertex)
      - sign: winding-number test, robust for any simple polygon (CW or CCW)
      - gradient: sign(SDF) * (query - closest) / max(|query - closest|, eps)
        Note: at points lying exactly on the polygon, |query - closest| is zero
        and the gradient is ill-defined. Callers with an authoritative normal
        (e.g. mesh surface nodes from the VTU file) should overwrite the
        gradient at those points.

    Args:
        query:      [N, 2] float tensor
        polygon:    [M, 2] float tensor, ordered closed polygon vertices
        chunk_size: process queries in chunks to bound peak memory

    Returns:
        sdf:     [N]        signed distance (>0 outside, <0 inside)
        grad:    [N, 2]     unit gradient of SDF
        closest: [N, 2]     nearest point on polygon for each query
    """
    assert query.dim() == 2 and query.shape[-1] == 2
    assert polygon.dim() == 2 and polygon.shape[-1] == 2

    a = polygon                                  # [M, 2]
    b = torch.roll(polygon, -1, dims=0)           # [M, 2]
    seg = b - a                                   # [M, 2]
    seg_len_sq = (seg ** 2).sum(-1) + 1e-12       # [M]

    N = query.shape[0]
    device = query.device
    dtype = query.dtype
    sdf_out = torch.empty(N, dtype=dtype, device=device)
    grad_out = torch.empty(N, 2, dtype=dtype, device=device)
    closest_out = torch.empty(N, 2, dtype=dtype, device=device)

    two_pi = 2.0 * math.pi

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        q = query[start:end]                                              # [B, 2]
        B = q.shape[0]

        # --- point-to-segment distance for every (B, M) pair ---
        qa = q[:, None, :] - a[None, :, :]                                 # [B, M, 2]
        t = (qa * seg[None, :, :]).sum(-1) / seg_len_sq[None, :]           # [B, M]
        t = t.clamp(0.0, 1.0)
        closest_all = a[None, :, :] + t[..., None] * seg[None, :, :]        # [B, M, 2]
        diff_all = q[:, None, :] - closest_all                              # [B, M, 2]
        dist_sq = (diff_all ** 2).sum(-1)                                   # [B, M]
        min_dist_sq, min_idx = dist_sq.min(dim=1)                            # [B]
        min_dist = min_dist_sq.sqrt()
        closest = closest_all[torch.arange(B, device=device), min_idx]      # [B, 2]

        # --- inside/outside via winding number (sum of signed turn angles) ---
        a_q = polygon - q[:, None, :]                                       # [B, M, 2]
        b_q = torch.roll(polygon, -1, dims=0) - q[:, None, :]
        cross = a_q[..., 0] * b_q[..., 1] - a_q[..., 1] * b_q[..., 0]
        dot   = a_q[..., 0] * b_q[..., 0] + a_q[..., 1] * b_q[..., 1]
        winding = torch.atan2(cross, dot).sum(dim=1) / two_pi              # [B]
        is_inside = winding.round().long().ne(0)                            # [B]

        # --- assemble signed distance + unit gradient ---
        signed = torch.where(is_inside, -min_dist, min_dist)
        safe = min_dist.clamp(min=1e-7)
        direction = (q - closest) / safe[..., None]
        grad = torch.where(is_inside[..., None], -direction, direction)

        sdf_out[start:end] = signed
        grad_out[start:end] = grad
        closest_out[start:end] = closest

    return sdf_out, grad_out, closest_out


# --------------------------------------------------------------------------- #
# Per-case extraction
# --------------------------------------------------------------------------- #


def process_case(case_name, my_path, grid_coords, grid_size,
                 grid_x_range, grid_y_range):
    """Extract one case from PyVista files into the v2 cache dict."""
    import pyvista as pv  # local import so the script is importable w/o pv

    internal = pv.read(osp.join(my_path, case_name, case_name + "_internal.vtu"))
    aerofoil = pv.read(osp.join(my_path, case_name, case_name + "_aerofoil.vtp"))

    airfoil_pos = torch.tensor(aerofoil.points[:, :2], dtype=torch.float32)
    full_pos = torch.tensor(internal.points[:, :2], dtype=torch.float32)

    # AirfRANS convention: wall no-slip BC → U_x == 0 exactly on the airfoil.
    surf_np = internal.point_data["U"][:, 0] == 0
    surf = torch.from_numpy(surf_np).bool()

    # Free-stream velocity from the AirfRANS case name encoding
    # "airFoil2D_<SOLVER>_<Uinf>_<alpha>_..._..."
    parts = case_name.split("_")
    Uinf = float(parts[2])
    alpha_rad = float(parts[3]) * np.pi / 180.0
    uinf = torch.tensor(
        [np.cos(alpha_rad) * Uinf, np.sin(alpha_rad) * Uinf],
        dtype=torch.float32,
    )

    full_y = torch.tensor(
        np.concatenate([
            internal.point_data["U"][:, :2],
            internal.point_data["p"][:, None],
            internal.point_data["nut"][:, None],
        ], axis=-1),
        dtype=torch.float32,
    )

    # --- Analytic SDF on the latent grid (sign matters: inside the airfoil → negative) ---
    grid_sdf, grid_sdf_grad, _ = polygon_sdf_and_grad(grid_coords, airfoil_pos)

    # --- Analytic SDF on the mesh nodes (mesh lives outside the airfoil) ---
    mesh_sdf, mesh_sdf_grad, _ = polygon_sdf_and_grad(full_pos, airfoil_pos)

    # Mesh surface nodes sit ON the polygon — the geometric gradient there is
    # ill-conditioned (query ≈ closest_point). Use AirfRANS' stored airfoil
    # normals (already known to be geometrically correct, just need reordering
    # to match the internal mesh node ordering).
    if surf_np.any():
        outward_normals = -aerofoil.point_data["Normals"][:, :2]  # OF stores inward
        reordered = reorganize(
            aerofoil.points[:, :2],
            internal.points[surf_np, :2],
            outward_normals,
        )
        mesh_sdf[surf] = 0.0
        mesh_sdf_grad[surf] = torch.from_numpy(np.ascontiguousarray(reordered)).float()

    return {
        "version": CACHE_SCHEMA_VERSION,
        "case_name": case_name,
        "airfoil_pos": airfoil_pos,
        "full_pos": full_pos,
        "full_y": full_y,
        "surf": surf,
        "uinf": uinf,
        "mesh_sdf": mesh_sdf,
        "mesh_sdf_grad": mesh_sdf_grad,
        "grid_sdf": grid_sdf,
        "grid_sdf_grad": grid_sdf_grad,
        "grid_size": grid_size,
        "grid_x_range": tuple(grid_x_range),
        "grid_y_range": tuple(grid_y_range),
    }


# --------------------------------------------------------------------------- #
# Cross-case normalization stats
# --------------------------------------------------------------------------- #


def _stream_mean(values_iter, dtype=np.float64):
    """Streaming Welford-style mean (running average weighted by element count)."""
    mean = None
    n_seen = 0
    for arr in values_iter:
        arr = arr.astype(dtype)
        n = arr.shape[0] if arr.ndim >= 1 else 1
        if mean is None:
            mean = arr.mean(axis=0) if arr.ndim >= 1 else arr
            n_seen = n
        else:
            arr_sum = arr.sum(axis=0) if arr.ndim >= 1 else arr
            new_n = n_seen + n
            mean = mean + (arr_sum - n * mean) / new_n
            n_seen = new_n
    return mean


def _stream_var(values_iter, mean, dtype=np.float64):
    """Streaming variance using the precomputed mean."""
    var = None
    n_seen = 0
    for arr in values_iter:
        arr = arr.astype(dtype)
        n = arr.shape[0] if arr.ndim >= 1 else 1
        sq = ((arr - mean) ** 2)
        sq_sum = sq.sum(axis=0) if sq.ndim >= 1 else sq
        if var is None:
            var = sq_sum / n
            n_seen = n
        else:
            new_n = n_seen + n
            var = var + (sq_sum - n * var) / new_n
            n_seen = new_n
    return var


def compute_coef_norm(cache_dir, train_cases):
    """Mean/std stats over training cases.

    Returns a dict with:
      mean_uinf [2], std_uinf [2]  — per-axis, equally weighted across cases
      mean_sdf  scalar, std_sdf  scalar — pooled over mesh+grid SDF values
                                          (same physical quantity)
      mean_out [4], std_out [4]  — per-channel, node-weighted over outputs
    """
    print(f"computing coef_norm over {len(train_cases)} training cases...")

    def load(c):
        return torch.load(osp.join(cache_dir, c + ".pt"),
                          map_location="cpu", weights_only=False)

    # --- uinf: small, just stack across cases ---
    uinfs = np.stack(
        [load(c)["uinf"].numpy() for c in tqdm(train_cases, desc="coef_norm uinf")],
        axis=0,
    ).astype(np.float64)
    mean_uinf = uinfs.mean(axis=0).astype(np.float32)
    std_uinf  = uinfs.std(axis=0).astype(np.float32)

    # --- sdf: streamed Welford, pooled (mesh ∪ grid) ---
    def sdf_iter():
        for c in train_cases:
            case = load(c)
            yield np.concatenate(
                [case["mesh_sdf"].numpy(), case["grid_sdf"].numpy()]
            )
    mean_sdf = _stream_mean(tqdm(sdf_iter(), total=len(train_cases),
                                  desc="coef_norm sdf pass1"))
    var_sdf  = _stream_var(tqdm(sdf_iter(), total=len(train_cases),
                                 desc="coef_norm sdf pass2"), mean_sdf)
    std_sdf  = np.sqrt(var_sdf)

    # --- output: streamed Welford, per-channel, node-weighted ---
    def y_iter():
        for c in train_cases:
            yield load(c)["full_y"].numpy()
    mean_out = _stream_mean(tqdm(y_iter(), total=len(train_cases),
                                  desc="coef_norm output pass1"))
    var_out  = _stream_var(tqdm(y_iter(), total=len(train_cases),
                                 desc="coef_norm output pass2"), mean_out)
    std_out  = np.sqrt(var_out)

    return {
        "mean_uinf": torch.from_numpy(mean_uinf).float(),
        "std_uinf":  torch.from_numpy(std_uinf).float(),
        "mean_sdf":  torch.tensor(float(mean_sdf), dtype=torch.float32),
        "std_sdf":   torch.tensor(float(std_sdf), dtype=torch.float32),
        "mean_out":  torch.from_numpy(mean_out.astype(np.float32)),
        "std_out":   torch.from_numpy(std_out.astype(np.float32)),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--my_path", default="data/Dataset",
                    help="AirfRANS data root.")
    ap.add_argument("--cache_dir", default="cache",
                    help="Output dir for per-case .pt files.")
    ap.add_argument("--task", default="full",
                    choices=["full", "scarce", "reynolds", "aoa"])
    ap.add_argument("--grid_size", type=int, default=64)
    ap.add_argument("--grid_x_range", type=float, nargs=2, default=[-2.0, 4.0])
    ap.add_argument("--grid_y_range", type=float, nargs=2, default=[-1.5, 1.5])
    ap.add_argument("--skip_existing", action="store_true",
                    help="Don't reprocess cases whose .pt already exists.")
    ap.add_argument("--num_workers", type=int, default=4,
                    help="Parallel pyvista workers (1 disables multiprocessing).")
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    grid_coords = build_grid_coords(args.grid_size, args.grid_x_range, args.grid_y_range)

    with open(osp.join(args.my_path, "manifest.json")) as f:
        manifest = json.load(f)

    if args.task == "scarce":
        train_cases = manifest["scarce_train"]
        test_cases = manifest["full_test"]
    else:
        train_cases = manifest[args.task + "_train"]
        test_cases = manifest[args.task + "_test"]
    all_cases = list(dict.fromkeys(train_cases + test_cases))

    print(f"task={args.task}: train={len(train_cases)}, "
          f"test={len(test_cases)}, total={len(all_cases)}")
    print(f"writing per-case caches to: {args.cache_dir}")
    print(f"grid: {args.grid_size}x{args.grid_size}, "
          f"x∈{args.grid_x_range}, y∈{args.grid_y_range}")
    print(f"num_workers: {args.num_workers}")

    failures = []
    if args.num_workers <= 1:
        for cname in tqdm(all_cases, desc="cases"):
            out_path = osp.join(args.cache_dir, cname + ".pt")
            if args.skip_existing and osp.exists(out_path):
                continue
            try:
                case = process_case(
                    cname, args.my_path, grid_coords, args.grid_size,
                    args.grid_x_range, args.grid_y_range,
                )
                torch.save(case, out_path)
            except Exception as exc:                          # noqa: BLE001
                print(f"!! failed {cname}: {exc}")
                failures.append((cname, str(exc)))
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=args.num_workers,
            initializer=_init_worker,
            initargs=(grid_coords, args.grid_size, args.grid_x_range,
                      args.grid_y_range, args.my_path, args.cache_dir,
                      args.skip_existing),
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

    train_cases_ok = [
        c for c in train_cases if osp.exists(osp.join(args.cache_dir, c + ".pt"))
    ]
    if not train_cases_ok:
        raise RuntimeError("no training cases were successfully cached.")

    stats = compute_coef_norm(args.cache_dir, train_cases_ok)
    coef_path = osp.join(args.cache_dir, f"coef_norm_{args.task}.pt")
    torch.save(
        {
            "version": CACHE_SCHEMA_VERSION,
            **stats,
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
