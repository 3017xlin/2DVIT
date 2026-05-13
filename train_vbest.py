"""V_best train: SWA-final, no-validation protocol.

Training config:
  - AdamW + weight_decay=0.01
  - FFN dropout 0.1 (set per-model)
  - LR schedule: linear warmup 5% + cosine decay to 1% * base_lr
  - Gradient clip max_norm=1.0
  - bf16 autocast
  - LOSS_WEIGHTS = [1.0, 1.0, 1.0, 1.0]  (AirfRANS standard protocol)

Checkpointing:
  - swa_model.pt          — Stochastic Weight Averaging (Izmailov et al., UAI 2018)
                            over the last 25% of epochs. This is the eval checkpoint.
  - model_state_dict.pt   — final-epoch weights (comparison / fine-tuning)
  - model                 — full pickled architecture (for auto-eval reload)
  - checkpoint_latest.pt  — saved every 20 epochs for crash-recovery resume

No validation set: the AirfRANS manifest's '<task>_train' is used in full at
training time. SWA replaces best-val for model selection.

Auto-eval at end runs the 3-step suite (Cd/Cl + field MSE + Spearman) on the
SWA checkpoint (falls back to final-epoch weights if no SWA epochs were
accumulated, i.e. nb_epochs too small for the 25% window).
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time, json

import torch
import torch.distributed as dist
import torch.nn as nn
import torch_geometric.nn as nng
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from tqdm import tqdm

from pathlib import Path
import os.path as osp


# V_best: UNIFORM per-variable weights (AirfRANS standard protocol).
# V8 used [1.0, 1.0, 1.5, 0.3] which is a unilateral protocol violation.
LOSS_WEIGHTS = (1.0, 1.0, 1.0, 1.0)


def _setup_ddp():
    if 'LOCAL_RANK' not in os.environ:
        return False, 0, 1, None
    local_rank = int(os.environ['LOCAL_RANK'])
    if not dist.is_initialized():
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    return True, dist.get_rank(), dist.get_world_size(), device


def get_nb_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def train(device, model, train_loader, optimizer, scheduler, criterion='MSE', reg=1, max_grad_norm=1.0):
    model.train()
    avg_loss_per_var = torch.zeros(4, device=device)
    avg_loss = 0
    avg_loss_surf_var = torch.zeros(4, device=device)
    avg_loss_vol_var = torch.zeros(4, device=device)
    avg_loss_surf = 0
    avg_loss_vol = 0
    grad_norm_sum = 0.0
    grad_norm_max = 0.0
    iter = 0
    use_amp = torch.cuda.is_available()

    for data in train_loader:
        data_clone = data.clone()
        data_clone = data_clone.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            out = model(data_clone)
        out = out.float()
        targets = data_clone.y

        if criterion == 'MSE' or criterion == 'MSE_weighted':
            loss_criterion = nn.MSELoss(reduction='none')
        elif criterion == 'MAE':
            loss_criterion = nn.L1Loss(reduction='none')
        loss_per_var = loss_criterion(out, targets).mean(dim=0)
        total_loss = loss_per_var.mean()
        loss_surf_var = loss_criterion(out[data_clone.surf, :], targets[data_clone.surf, :]).mean(dim=0)
        loss_vol_var = loss_criterion(out[~data_clone.surf, :], targets[~data_clone.surf, :]).mean(dim=0)
        loss_surf = loss_surf_var.mean()
        loss_vol = loss_vol_var.mean()

        # UNIFORM weights — same as AirfRANS protocol.
        _w = torch.tensor(LOSS_WEIGHTS, device=loss_per_var.device, dtype=loss_per_var.dtype)
        _norm = _w.sum()
        loss_surf_w = (loss_surf_var * _w).sum() / _norm
        loss_vol_w = (loss_vol_var * _w).sum() / _norm
        total_loss_w = (loss_per_var * _w).sum() / _norm

        if criterion == 'MSE_weighted':
            (loss_vol_w + reg * loss_surf_w).backward()
        else:
            total_loss_w.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm).item()
        grad_norm_sum += grad_norm
        if grad_norm > grad_norm_max:
            grad_norm_max = grad_norm

        optimizer.step()
        scheduler.step()
        avg_loss_per_var += loss_per_var
        avg_loss += total_loss
        avg_loss_surf_var += loss_surf_var
        avg_loss_vol_var += loss_vol_var
        avg_loss_surf += loss_surf
        avg_loss_vol += loss_vol
        iter += 1

    return (avg_loss.cpu().data.numpy() / iter,
            avg_loss_per_var.cpu().data.numpy() / iter,
            avg_loss_surf_var.cpu().data.numpy() / iter,
            avg_loss_vol_var.cpu().data.numpy() / iter,
            avg_loss_surf.cpu().data.numpy() / iter,
            avg_loss_vol.cpu().data.numpy() / iter,
            grad_norm_sum / iter,
            grad_norm_max)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# ============================================================================
# AUTO-EVAL PIPELINE (called automatically at end of main())
# ============================================================================

def _save_case_plot_pair(plots_dir, case_name, pos, true_phys, pred_phys, airfoil_poly,
                          grid_x_range=(-2.0, 4.0), grid_y_range=(-1.5, 1.5)):
    """Render two 2x2 PNGs for one case: <name>_true.png and <name>_pred.png.

    Each PNG has 4 panels: Ux, Uy, p, nu_t — drawn with tricontourf over the
    unstructured mesh nodes (matplotlib auto-Delaunay), the airfoil overlaid as
    a white-filled polygon. Color scale is shared between true and pred per
    field (1st and 99th percentile of the pooled true+pred values) so the two
    images are directly visually comparable.
    """
    from matplotlib import tri as _mpltri

    titles = ['$U_x$', '$U_y$', '$p$', r'$\nu_t$']
    vmins = np.empty(4)
    vmaxs = np.empty(4)
    for k in range(4):
        pooled = np.concatenate([true_phys[:, k], pred_phys[:, k]])
        vmins[k] = np.percentile(pooled, 1)
        vmaxs[k] = np.percentile(pooled, 99)

    triang = _mpltri.Triangulation(pos[:, 0], pos[:, 1])
    poly_closed = np.vstack([airfoil_poly, airfoil_poly[0:1]])

    for kind, fields in (('true', true_phys), ('pred', pred_phys)):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        for k, (ax, title) in enumerate(zip(axes.flatten(), titles)):
            c = ax.tricontourf(triang, fields[:, k], levels=50, cmap='RdBu_r',
                               vmin=vmins[k], vmax=vmaxs[k])
            plt.colorbar(c, ax=ax)
            ax.fill(poly_closed[:, 0], poly_closed[:, 1],
                    facecolor='white', edgecolor='black', linewidth=0.6)
            ax.set_xlim(*grid_x_range)
            ax.set_ylim(*grid_y_range)
            ax.set_aspect('equal')
            ax.set_title(f'{title} ({kind})')
        fig.suptitle(f'{case_name} — {kind.upper()}', fontsize=11)
        plt.tight_layout()
        fig.savefig(osp.join(plots_dir, f'{case_name}_{kind}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)


def _format_eval_summary(s):
    """Render the eval summary dict as a human-readable text block."""
    lines = []
    lines.append("=" * 72)
    lines.append(f"V_best evaluation summary")
    lines.append(f"checkpoint   : {s['checkpoint']}")
    lines.append(f"task         : {s['task']}")
    lines.append(f"test cases   : {s['n_test_cases']}")
    lines.append("=" * 72)

    for name, key in (('Cd (drag)', 'cd'), ('Cl (lift)', 'cl')):
        d = s[key]
        lines.append('')
        lines.append(f"== {name} ==")
        lines.append(f"  Spearman rho       : {d['spearman_rho']:.4f}")
        lines.append(f"  MSE                : {d['mse']:.6e}")
        lines.append(f"  Median |rel err|   : {d['median_rel_err']*100:.2f}%")
        lines.append(f"  Mean   |rel err|   : {d['mean_rel_err']*100:.2f}%")
        lines.append(f"  Max    |rel err|   : {d['max_rel_err']*100:.2f}%")
        lines.append(f"  True range         : [{d['true_range'][0]:.4f}, "
                     f"{d['true_range'][1]:.4f}]")
        lines.append(f"  Pred range         : [{d['pred_range'][0]:.4f}, "
                     f"{d['pred_range'][1]:.4f}]")

    fm = s['field_mse']
    lines.append('')
    lines.append("== Field MSE (normalized space, AirfRANS multi-pass protocol) ==")
    lines.append(f"  Vol  MSE  mean +/- std : {fm['vol_mse_mean']:.6f}  +/-  {fm['vol_mse_std']:.6f}")
    lines.append(f"  Surf MSE  mean +/- std : {fm['surf_mse_mean']:.6f}  +/-  {fm['surf_mse_std']:.6f}")
    lines.append(f"  Vol  per-var  [Ux, Uy, p, nut]: {fm['vol_per_var']}")
    lines.append(f"  Surf per-var  [Ux, Uy, p, nut]: {fm['surf_per_var']}")

    lines.append('')
    lines.append(f"Plotted cases ({len(s['plotted_cases'])}):")
    for n in s['plotted_cases']:
        lines.append(f"  - {n}")
    lines.append("=" * 72)
    return '\n'.join(lines)


def _run_post_training_eval(
    save_path,
    model_name,
    task,
    my_path,
    coef_norm,
    hparams,
    DatasetClass,
    log_dir=None,
):
    """Post-training evaluation. Single pass over the test set producing:

      - Cd / Cl per case via PyVista surface integration (AirfRANS standard)
      - Cd / Cl aggregate stats: Spearman rho, MSE, mean/median/max rel err
      - Field MSE per case (Vol/Surf, per-variable) on multi-pass coverage
        predictions, matching the AirfRANS evaluation protocol
      - 10 randomly chosen test cases rendered as 2x2 PNGs (true + pred,
        20 images total)

    Uses swa_model.pt; falls back to model_state_dict.pt if SWA didn't run.
    """
    import random as _random
    import scipy.stats as _stats
    import pyvista as _pv
    import utils.metrics as metrics

    # Reproducible test-set selection AND reproducible multi-pass coverage
    # inference inside Infer_test (which calls random.sample internally).
    _random.seed(42)
    np.random.seed(42)

    metrics.Dataset = DatasetClass

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if log_dir is None:
        log_dir = osp.join(save_path, task, model_name)
    eval_dir = osp.join(log_dir, 'eval')
    plots_dir = osp.join(eval_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    arch_path = osp.join(log_dir, 'model')
    swa_path = osp.join(log_dir, 'swa_model.pt')
    final_path = osp.join(log_dir, 'model_state_dict.pt')

    print("\n" + "=" * 72)
    print("[AUTO-EVAL] V_best evaluation")
    print(f"[AUTO-EVAL] outputs -> {eval_dir}")
    print("=" * 72)

    print(f"[AUTO-EVAL] Loading architecture from {arch_path}")
    model = torch.load(arch_path, map_location=device, weights_only=False)
    if osp.exists(swa_path):
        ckpt_used = 'swa_model.pt'
        print(f"[AUTO-EVAL] Loading SWA state from {swa_path}")
        state = torch.load(swa_path, map_location=device, weights_only=False)
    else:
        ckpt_used = 'model_state_dict.pt'
        print(f"[AUTO-EVAL] swa_model.pt missing — falling back to {final_path}")
        state = torch.load(final_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device).eval()

    s_key = task + '_test' if task != 'scarce' else 'full_test'
    with open(osp.join(my_path, 'manifest.json')) as f:
        manifest = json.load(f)
    test_names = manifest[s_key]
    print(f"[AUTO-EVAL] Loading {len(test_names)} test cases via {DatasetClass.__module__}")

    test_dataset = DatasetClass(test_names, sample=None, coef_norm=coef_norm,
                                my_path=my_path, task=task)

    # 10 random cases to render. We grab indices into the test set so we can
    # spot them inside the per-case loop without an extra pass.
    n_plot = min(10, len(test_names))
    plot_indices = sorted(_random.sample(range(len(test_names)), n_plot))
    plot_indices_set = set(plot_indices)
    plot_case_names = [test_names[i] for i in plot_indices]

    mean_out_t = coef_norm['mean_out']            # torch [4]
    std_out_t = coef_norm['std_out']

    cd_true_list, cd_pred_list = [], []
    cl_true_list, cl_pred_list = [], []
    vol_pc, surf_pc = [], []
    vol_pv, surf_pv = [], []

    print(f"[AUTO-EVAL] Evaluating {len(test_names)} cases "
          f"(rendering 10: {plot_case_names[0]} ... {plot_case_names[-1]})")
    for i, (case_name, data) in enumerate(zip(test_names, test_dataset)):
        # 1) AirfRANS-protocol multi-pass coverage inference (normalized space).
        outs, _ = metrics.Infer_test(device, [model], [hparams], data,
                                     coef_norm=coef_norm)
        out_norm = outs[0]                                # [N, 4] CPU tensor

        # 2) Field MSE (normalized space).
        loss_v = ((out_norm[~data.surf] - data.y[~data.surf]) ** 2).mean(dim=0)
        loss_s = ((out_norm[data.surf]  - data.y[data.surf])  ** 2).mean(dim=0)
        vol_pc.append(float(loss_v.mean().item()))
        surf_pc.append(float(loss_s.mean().item()))
        vol_pv.append(loss_v.cpu().numpy())
        surf_pv.append(loss_s.cpu().numpy())

        # 3) Read the case's VTU + airfoil and integrate Cd/Cl.
        internal = _pv.read(osp.join(my_path, case_name, case_name + '_internal.vtu'))
        aerofoil = _pv.read(osp.join(my_path, case_name, case_name + '_aerofoil.vtp'))
        parts = case_name.split('_')
        Uinf = float(parts[2])
        angle = float(parts[3])

        tc = metrics.Compute_coefficients([internal], [aerofoil],
                                          data.surf, Uinf, angle, keep_vtk=False)
        cd_true_list.append(float(tc[0][0]))
        cl_true_list.append(float(tc[0][1]))

        intern_pred, aero_pred = metrics.Airfoil_test(internal, aerofoil,
                                                      [out_norm], coef_norm, data.surf)
        pc = metrics.Compute_coefficients(intern_pred, aero_pred,
                                          data.surf, Uinf, angle, keep_vtk=False)
        cd_pred_list.append(float(pc[0][0]))
        cl_pred_list.append(float(pc[0][1]))

        # 4) Plots for the 10 selected cases — denormalize once for visualization.
        if i in plot_indices_set:
            pred_phys = (out_norm * (std_out_t + 1e-8) + mean_out_t).cpu().numpy()
            true_phys = (data.y   * (std_out_t + 1e-8) + mean_out_t).cpu().numpy()
            # Zero-out predicted U and nu_t on surface (no-slip BC; matches
            # what AirfRANS' Airfoil_test does in physical space).
            surf_np = data.surf.cpu().numpy()
            pred_phys[surf_np, :2] = 0.0
            pred_phys[surf_np, 3]  = 0.0
            _save_case_plot_pair(
                plots_dir, case_name,
                data.pos.cpu().numpy(),
                true_phys, pred_phys,
                data.airfoil_pos.cpu().numpy(),
            )

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(test_names)}")

    # --- Aggregate Cd/Cl ---
    cd_true = np.asarray(cd_true_list, dtype=np.float64)
    cd_pred = np.asarray(cd_pred_list, dtype=np.float64)
    cl_true = np.asarray(cl_true_list, dtype=np.float64)
    cl_pred = np.asarray(cl_pred_list, dtype=np.float64)

    def _coef_stats(true, pred):
        sp, _ = _stats.spearmanr(true, pred)
        mse = float(np.mean((true - pred) ** 2))
        rel = np.abs(true - pred) / (np.abs(true) + 1e-8)
        return {
            'spearman_rho':   float(sp),
            'mse':            mse,
            'mean_rel_err':   float(np.mean(rel)),
            'median_rel_err': float(np.median(rel)),
            'max_rel_err':    float(np.max(rel)),
            'true_range':     [float(true.min()), float(true.max())],
            'pred_range':     [float(pred.min()), float(pred.max())],
        }

    # --- Aggregate field MSE ---
    vol_pv_arr = np.stack(vol_pv).mean(axis=0)
    surf_pv_arr = np.stack(surf_pv).mean(axis=0)

    summary = {
        'checkpoint':   ckpt_used,
        'task':         task,
        'n_test_cases': len(test_names),
        'cd':           _coef_stats(cd_true, cd_pred),
        'cl':           _coef_stats(cl_true, cl_pred),
        'field_mse': {
            'vol_mse_mean':  float(np.mean(vol_pc)),
            'vol_mse_std':   float(np.std(vol_pc)),
            'surf_mse_mean': float(np.mean(surf_pc)),
            'surf_mse_std':  float(np.std(surf_pc)),
            'vol_per_var':   vol_pv_arr.tolist(),
            'surf_per_var':  surf_pv_arr.tolist(),
        },
        'plotted_cases': plot_case_names,
    }

    with open(osp.join(eval_dir, 'eval_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    txt = _format_eval_summary(summary)
    with open(osp.join(eval_dir, 'eval_summary.txt'), 'w') as f:
        f.write(txt + '\n')

    print('\n' + txt)
    print(f"\n[AUTO-EVAL] Done.")
    print(f"  Summary  : {osp.join(eval_dir, 'eval_summary.json')}")
    print(f"  Report   : {osp.join(eval_dir, 'eval_summary.txt')}")
    print(f"  Plots    : {plots_dir}/  ({n_plot} cases x 2 = {2*n_plot} PNGs)")


# ============================================================================
# TRAINING LOOP
# ============================================================================

def main(device, train_dataset, Net, hparams, path, criterion='MSE', reg=1,
         name_mod='UrbanWindViT',
         auto_eval=True, my_path=None, task='full', coef_norm=None, DatasetClass=None,
         save_path=None):
    """Trains the model, accumulates SWA over the last 25% of epochs, then runs auto-eval.

    Validation has been removed entirely: all training cases go to training; SWA is
    the model-selection mechanism (Izmailov et al., UAI 2018) — a simple equal-weight
    average of weights from epoch `int(0.75 * nb_epochs)` onward. The averaged
    weights are saved as swa_model.pt and used as the auto-eval checkpoint.

    Extra args:
      auto_eval (bool):  run the 3-step eval suite after training.
      my_path (str):     AirfRANS root data path (needed for auto-eval).
      task (str):        'full' / 'scarce' / 'reynolds' / 'aoa'.
      coef_norm (dict):  preprocess.py v2 stats dict (mean_uinf, std_uinf, ...).
      DatasetClass:      Dataset class used at train time (dataset_cached.Dataset).
      save_path (str):   --save_path root for output dirs.
    """
    is_distributed, rank, world_size, dist_device = _setup_ddp()
    if is_distributed:
        device = dist_device
    is_main = (rank == 0)

    if is_main:
        Path(path).mkdir(parents=True, exist_ok=True)

    model = Net.to(device)
    if is_distributed:
        device_ids = [device.index] if device.type == 'cuda' else None
        model = DDP(model, device_ids=device_ids, find_unused_parameters=False)

    def _unwrap(m):
        base = m.module if hasattr(m, 'module') else m
        return getattr(base, '_orig_mod', base)

    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams['lr'], weight_decay=0.01)

    import math as _math
    _effective_world_size = world_size if is_distributed else 1
    _steps_per_epoch = len(train_dataset) // (hparams['batch_size'] * _effective_world_size) + 1
    _total_steps = _steps_per_epoch * hparams['nb_epochs']
    _warmup_steps = int(0.05 * _total_steps)

    def _smart_lr(step):
        if step < _warmup_steps:
            return max(0.01, step / _warmup_steps)
        progress = (step - _warmup_steps) / max(1, _total_steps - _warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + _math.cos(_math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_smart_lr)

    # SWA: simple-average the weights over the final 25% of epochs.
    # Only rank 0 maintains the accumulator; DDP keeps the underlying params in sync
    # across ranks, so any rank's view is fine to feed into update_parameters().
    swa_start_epoch = int(0.75 * hparams['nb_epochs'])
    swa_model = None
    if is_main:
        from torch.optim.swa_utils import AveragedModel
        swa_model = AveragedModel(_unwrap(model))

    start = time.time()

    train_loss_surf_list = []
    train_loss_vol_list = []
    loss_surf_var_list = []
    loss_vol_var_list = []

    pbar_train = tqdm(range(hparams['nb_epochs']), position=0)
    for epoch in pbar_train:
        train_dataset_sampled = []
        for data in train_dataset:
            data_sampled = data.clone()
            idx = random.sample(range(data_sampled.pos.size(0)), hparams['subsampling'])
            idx = torch.tensor(idx)

            # Per-node fields are subsampled; graph-level fields (uinf, grid_sdf,
            # grid_sdf_grad, airfoil_pos) stay full size.
            data_sampled.pos = data_sampled.pos[idx]
            data_sampled.y = data_sampled.y[idx]
            data_sampled.surf = data_sampled.surf[idx]
            data_sampled.sdf = data_sampled.sdf[idx]
            data_sampled.sdf_grad = data_sampled.sdf_grad[idx]

            if name_mod != 'PointNet' and name_mod != 'MLP' and name_mod != 'UrbanWindViT':
                data_sampled.edge_index = nng.radius_graph(x=data_sampled.pos.to(device), r=hparams['r'], loop=True,
                                                           max_num_neighbors=int(hparams['max_neighbors'])).cpu()

            train_dataset_sampled.append(data_sampled)
        if is_distributed:
            train_sampler = DistributedSampler(
                train_dataset_sampled, num_replicas=world_size, rank=rank, shuffle=True
            )
            train_sampler.set_epoch(epoch)
            train_loader = DataLoader(
                train_dataset_sampled, batch_size=hparams['batch_size'], sampler=train_sampler,
                num_workers=2, pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_dataset_sampled, batch_size=hparams['batch_size'], shuffle=True,
                num_workers=2, pin_memory=True,
            )
        del train_dataset_sampled

        (train_loss, _, loss_surf_var, loss_vol_var, loss_surf, loss_vol,
         grad_norm_avg, grad_norm_max) = train(
            device, model, train_loader, optimizer, lr_scheduler, criterion, reg=reg
        )
        current_lr = optimizer.param_groups[0]['lr']
        if is_main:
            print('epoch: ' + str(epoch))
            print('train_loss: ' + str(train_loss))
            print('loss_vol: ' + str(loss_vol))
            print('loss_surf: ' + str(loss_surf))
            print('grad_norm: avg={:.4f} max={:.4f} | lr={:.6f}'.format(
                grad_norm_avg, grad_norm_max, current_lr))

        if criterion == 'MSE_weighted':
            train_loss = reg * loss_surf + loss_vol
        del train_loader

        # SWA accumulation (rank 0 only).
        if is_main and epoch >= swa_start_epoch:
            swa_model.update_parameters(_unwrap(model))

        if is_main and (epoch + 1) % 20 == 0:
            torch.save(
                _unwrap(model).state_dict(),
                osp.join(path, 'checkpoint_latest.pt'),
            )
            print('saved checkpoint_latest.pt at epoch {}'.format(epoch))

        train_loss_surf_list.append(loss_surf)
        train_loss_vol_list.append(loss_vol)
        loss_surf_var_list.append(loss_surf_var)
        loss_vol_var_list.append(loss_vol_var)

        pbar_train.set_postfix(train_loss=train_loss, loss_surf=loss_surf)

    loss_surf_var_list = np.array(loss_surf_var_list)
    loss_vol_var_list = np.array(loss_vol_var_list)

    end = time.time()
    time_elapsed = end - start
    params_model = get_nb_trainable_params(model).astype('float')
    saveable_model = _unwrap(model)
    swa_epochs_averaged = 0
    if is_main:
        print('Number of parameters:', params_model)
        print('Time elapsed: {0:.2f} seconds'.format(time_elapsed))
        torch.save(saveable_model, osp.join(path, 'model'))
        torch.save(saveable_model.state_dict(), osp.join(path, 'model_state_dict.pt'))

        # SWA-averaged weights (the model used by auto-eval and by downstream
        # benchmark reporting). Falls through silently if no SWA epochs were
        # accumulated (e.g. nb_epochs too small for the 25% window).
        if swa_model is not None:
            swa_epochs_averaged = int(swa_model.n_averaged.item())
            if swa_epochs_averaged > 0:
                torch.save(
                    swa_model.module.state_dict(),
                    osp.join(path, 'swa_model.pt'),
                )
                print('saved swa_model.pt averaged over {} epochs (from epoch {})'.format(
                    swa_epochs_averaged, swa_start_epoch))
            else:
                print('WARNING: no SWA epochs accumulated; auto-eval will use final-epoch weights.')

    if is_main:
        sns.set()
        for tag, lst, lst_var, ttl in [
            ('train_loss_surf', train_loss_surf_list, loss_surf_var_list, 'Train losses over the surface'),
            ('train_loss_vol', train_loss_vol_list, loss_vol_var_list, 'Train losses over the volume'),
        ]:
            if lst is None or len(lst) == 0:
                continue
            fig, ax = plt.subplots(figsize=(20, 5))
            ax.plot(lst, label='Mean loss')
            arr = np.array(lst_var)
            if arr.ndim == 2 and arr.shape[1] >= 4:
                ax.plot(arr[:, 0], label=r'$v_x$ loss')
                ax.plot(arr[:, 1], label=r'$v_y$ loss')
                ax.plot(arr[:, 2], label=r'$p$ loss')
                ax.plot(arr[:, 3], label=r'$\nu_t$ loss')
            ax.set_xlabel('epochs')
            ax.set_yscale('log')
            ax.set_title(ttl)
            ax.legend(loc='best')
            fig.savefig(osp.join(path, tag + '.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)

        with open(osp.join(path, name_mod + '_log.json'), 'a') as f:
            json.dump(
                {
                    'regression': 'Total',
                    'loss': 'MSE',
                    'nb_parameters': params_model,
                    'time_elapsed': time_elapsed,
                    'hparams': hparams,
                    'loss_weights': list(LOSS_WEIGHTS),
                    'swa_start_epoch': swa_start_epoch,
                    'swa_epochs_averaged': swa_epochs_averaged,
                    'train_loss_surf': str(train_loss_surf_list[-1]),
                    'train_loss_surf_var': str(loss_surf_var_list[-1]),
                    'train_loss_vol': str(train_loss_vol_list[-1]),
                    'train_loss_vol_var': str(loss_vol_var_list[-1]),
                }, f, indent=12, cls=NumpyEncoder
            )

        # Per-epoch loss curves saved as raw data for downstream plotting
        # / regressions / paper figures. The two PNGs above are the visual
        # version of the same arrays.
        with open(osp.join(path, 'loss_history.json'), 'w') as f:
            json.dump(
                {
                    'epochs': int(hparams['nb_epochs']),
                    'swa_start_epoch': swa_start_epoch,
                    'swa_epochs_averaged': swa_epochs_averaged,
                    'train_loss_surf':         [float(x) for x in train_loss_surf_list],
                    'train_loss_vol':          [float(x) for x in train_loss_vol_list],
                    'train_loss_surf_per_var': loss_surf_var_list.tolist(),
                    'train_loss_vol_per_var':  loss_vol_var_list.tolist(),
                }, f, indent=2,
            )

    if is_distributed:
        dist.barrier()

    # =========================================================================
    # AUTO-EVAL: Cd/Cl + field MSE + per-case rendering on the SWA checkpoint
    # =========================================================================
    if is_main and auto_eval:
        if save_path is None or my_path is None or coef_norm is None or DatasetClass is None:
            print("\n[AUTO-EVAL] SKIPPED — auto_eval=True but missing required arg(s):")
            print(f"  save_path={save_path}, my_path={my_path}, "
                  f"coef_norm={'set' if coef_norm is not None else 'None'}, DatasetClass={DatasetClass}")
        else:
            try:
                _run_post_training_eval(
                    save_path=save_path,
                    model_name=name_mod,
                    task=task,
                    my_path=my_path,
                    coef_norm=coef_norm,
                    hparams=hparams,
                    DatasetClass=DatasetClass,
                )
            except Exception as e:
                import traceback
                print(f"\n[AUTO-EVAL] FAILED with exception: {e}")
                traceback.print_exc()
                print("[AUTO-EVAL] Training still succeeded — you can run eval manually.")

    return saveable_model
