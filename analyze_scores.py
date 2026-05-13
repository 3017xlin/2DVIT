"""Analyze score outputs (Cd, Cl) for any version (v8 / v11 / v12 / v13 / ...).

Usage:
    python analyze_scores.py scores_v12/full
    python analyze_scores.py scores_v13/full
"""

import os, os.path as osp, sys
import numpy as np


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main(score_dir):
    print(f"=== Analysis of {score_dir} ===\n")

    true_path = osp.join(score_dir, 'true_coefs.npy')
    pred_path = osp.join(score_dir, 'pred_coefs_mean.npy')
    if not osp.exists(true_path) or not osp.exists(pred_path):
        print(f"ERROR: missing {true_path} or {pred_path}")
        sys.exit(1)

    true = np.load(true_path)        # shape (200, 2)
    pred = np.load(pred_path)        # shape (200, 1, 2)
    pred = pred.squeeze()             # (200, 2)
    assert true.shape == pred.shape, f"shape mismatch: {true.shape} vs {pred.shape}"

    n = true.shape[0]
    print(f"Cases: {n}")
    print(f"True coefs columns: [Cd, Cl]")

    abs_err = np.abs(pred - true)
    rel_err = abs_err / (np.abs(true) + 1e-9)

    for i, name in enumerate(['Cd (drag)', 'Cl (lift)']):
        print(f"\n--- {name} ---")
        print(f"  True range:  [{true[:, i].min():.5f}, {true[:, i].max():.5f}]")
        print(f"  Pred range:  [{pred[:, i].min():.5f}, {pred[:, i].max():.5f}]")
        n_neg_pred = int((pred[:, i] < 0).sum())
        n_neg_true = int((true[:, i] < 0).sum())
        print(f"  Pred negatives: {n_neg_pred} cases (true negatives: {n_neg_true})")
        print(f"  Abs err:     mean={abs_err[:, i].mean():.5f}  median={np.median(abs_err[:, i]):.5f}  max={abs_err[:, i].max():.5f}")
        print(f"  Rel err %:   mean={rel_err[:, i].mean()*100:6.2f}%  median={np.median(rel_err[:, i])*100:6.2f}%  max={rel_err[:, i].max()*100:6.2f}%")
        print(f"  Pearson:     {pearson(true[:, i], pred[:, i]):.4f}")
        print(f"  Spearman:    {spearman(true[:, i], pred[:, i]):.4f}")

    # Summary table line for easy comparison across versions
    cd_sp = spearman(true[:, 0], pred[:, 0])
    cl_sp = spearman(true[:, 1], pred[:, 1])
    cd_med = np.median(rel_err[:, 0]) * 100
    cl_med = np.median(rel_err[:, 1]) * 100
    print(f"\n=== SUMMARY (paste this into your comparison sheet) ===")
    print(f"  {osp.basename(osp.dirname(score_dir.rstrip('/')))} | "
          f"Cd Spearman={cd_sp:.4f}  Cd_med_rel={cd_med:.2f}%  | "
          f"Cl Spearman={cl_sp:.4f}  Cl_med_rel={cl_med:.2f}%")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_scores.py <scores_dir>")
        print("Example: python analyze_scores.py scores_v12/full")
        sys.exit(1)
    main(sys.argv[1])
