"""V_best train: V8 minus the cheating per-variable loss weights [1,1,1.5,0.3].

All V8 improvements RETAINED:
  - AdamW + weight_decay=0.01
  - FFN dropout 0.1 (set per-model)
  - Smart LR schedule (linear warmup 5% + cosine decay 95%)
  - Gradient clip max_norm=1.0
  - bf16 autocast
  - best_model.pt save on val improvement
  - Periodic checkpoint_latest.pt every 20 epochs

CHANGED vs V8:
  - LOSS_WEIGHTS = [1.0, 1.0, 1.0, 1.0]  (UNIFORM, AirfRANS standard protocol)
  - End of main() now auto-runs the full 3-step evaluation:
       Step 1: AirfRANS Results_test  -> Cd/Cl + surface coefs + BL profiles
       Step 2: Test-set field-level MSE -> per-case + per-variable Vol/Surf MSE
       Step 3: Coefficient post-processing -> Spearman rho, relative errors

All 6 V_best_* variants share this train module.
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


@torch.no_grad()
def test(device, model, test_loader, criterion='MSE'):
    model.eval()
    avg_loss_per_var = np.zeros(4)
    avg_loss = 0
    avg_loss_surf_var = np.zeros(4)
    avg_loss_vol_var = np.zeros(4)
    avg_loss_surf = 0
    avg_loss_vol = 0
    iter = 0
    use_amp = torch.cuda.is_available()

    for data in test_loader:
        data_clone = data.clone()
        data_clone = data_clone.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            out = model(data_clone)
        out = out.float()

        targets = data_clone.y
        if criterion == 'MSE' or criterion == 'MSE_weighted':
            loss_criterion = nn.MSELoss(reduction='none')
        elif criterion == 'MAE':
            loss_criterion = nn.L1Loss(reduction='none')

        loss_per_var = loss_criterion(out, targets).mean(dim=0)
        loss = loss_per_var.mean()
        loss_surf_var = loss_criterion(out[data_clone.surf, :], targets[data_clone.surf, :]).mean(dim=0)
        loss_vol_var = loss_criterion(out[~data_clone.surf, :], targets[~data_clone.surf, :]).mean(dim=0)
        loss_surf = loss_surf_var.mean()
        loss_vol = loss_vol_var.mean()

        avg_loss_per_var += loss_per_var.cpu().numpy()
        avg_loss += loss.cpu().numpy()
        avg_loss_surf_var += loss_surf_var.cpu().numpy()
        avg_loss_vol_var += loss_vol_var.cpu().numpy()
        avg_loss_surf += loss_surf.cpu().numpy()
        avg_loss_vol += loss_vol.cpu().numpy()
        iter += 1

    return avg_loss / iter, avg_loss_per_var / iter, avg_loss_surf_var / iter, avg_loss_vol_var / iter, avg_loss_surf / iter, avg_loss_vol / iter


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# ============================================================================
# AUTO-EVAL PIPELINE (called automatically at end of main())
# ============================================================================

def _run_post_training_eval(
    save_path,
    model_name,
    task,
    my_path,
    coef_norm,
    hparams,
    DatasetClass,
    scores_dir=None,
    log_dir=None,
):
    """Auto-run the 3-step evaluation suite after training.

    Step 1: AirfRANS aerodynamic coefficient evaluation (Results_test)
    Step 2: Test-set field-level MSE
    Step 3: Coefficient post-processing (Spearman + rel err)

    Uses the saved best_model.pt (not the final-epoch model).
    """
    import scipy.stats
    import utils.metrics as metrics

    # Monkey-patch metrics.Dataset to whichever Dataset variant we trained with
    # (default dataset.dataset.Dataset for V_best; same here since V_best stays
    # on standard 7-channel input).
    metrics.Dataset = DatasetClass

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if scores_dir is None:
        scores_dir = f'scores_{save_path}' if not save_path.startswith('metrics_') \
            else 'scores_' + save_path[len('metrics_'):]
    if log_dir is None:
        log_dir = osp.join(save_path, task, model_name)

    arch_path = osp.join(save_path, task, model_name, 'model')
    best_path = osp.join(save_path, task, model_name, 'best_model.pt')

    print("\n" + "=" * 70)
    print("[AUTO-EVAL] V_best 3-step evaluation suite")
    print(f"[AUTO-EVAL] save_path    = {save_path}")
    print(f"[AUTO-EVAL] scores_dir   = {scores_dir}")
    print(f"[AUTO-EVAL] using BEST checkpoint")
    print("=" * 70)

    # Load best-epoch model
    print(f"[AUTO-EVAL] Loading architecture from {arch_path}...")
    model = torch.load(arch_path, map_location=device, weights_only=False)
    print(f"[AUTO-EVAL] Loading BEST state from {best_path}...")
    best_state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_state)
    model.to(device).eval()

    # =========== STEP 1: AirfRANS Results_test (Cd/Cl + surface coefs + BL) ===========
    print("\n" + "─" * 70)
    print("[AUTO-EVAL Step 1/3] AirfRANS aerodynamic coefficient evaluation")
    print("─" * 70)

    s = task + '_test' if task != 'scarce' else 'full_test'
    out_dir = osp.join(scores_dir, task)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Step 1] Outputs -> {out_dir}/")

    coefs = metrics.Results_test(
        device, [[model]], [hparams], coef_norm, my_path,
        path_out=scores_dir, n_test=3, criterion='MSE', s=s
    )
    np.save(osp.join(out_dir, 'true_coefs'), coefs[0])
    np.save(osp.join(out_dir, 'pred_coefs_mean'), coefs[1])
    np.save(osp.join(out_dir, 'pred_coefs_std'), coefs[2])
    for n_, f_ in enumerate(coefs[3]):
        np.save(osp.join(out_dir, f'true_surf_coefs_{n_}'), f_)
    for n_, f_ in enumerate(coefs[4]):
        np.save(osp.join(out_dir, f'surf_coefs_{n_}'), f_)
    np.save(osp.join(out_dir, 'true_bls'), coefs[5])
    np.save(osp.join(out_dir, 'bls'), coefs[6])
    print(f"[Step 1] DONE — .npy files saved in {out_dir}/")

    # =========== STEP 2: Test-set field-level MSE ===========
    print("\n" + "─" * 70)
    print("[AUTO-EVAL Step 2/3] Test-set field-level MSE")
    print("─" * 70)

    with open(my_path + '/manifest.json') as f:
        manifest = json.load(f)
    test_names = manifest[task + '_test' if task != 'scarce' else 'full_test']
    print(f"[Step 2] Loading {len(test_names)} test cases with stored coef_norm...")

    # Pass `task` if DatasetClass accepts it (dataset_cached.Dataset does, original doesn't).
    try:
        test_dataset = DatasetClass(test_names, sample=None, coef_norm=coef_norm,
                                    my_path=my_path, task=task)
    except TypeError:
        test_dataset = DatasetClass(test_names, sample=None, coef_norm=coef_norm, my_path=my_path)

    loss_fn = nn.MSELoss(reduction='none')
    vol_pc, surf_pc, vol_pv, surf_pv = [], [], [], []
    use_amp = torch.cuda.is_available()
    with torch.no_grad():
        for i, data in enumerate(test_dataset):
            data = data.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                out = model(data)
            out = out.float()
            targets = data.y
            loss_v = loss_fn(out[~data.surf], targets[~data.surf]).mean(dim=0)
            loss_s = loss_fn(out[data.surf], targets[data.surf]).mean(dim=0)
            vol_pc.append(loss_v.mean().item())
            surf_pc.append(loss_s.mean().item())
            vol_pv.append(loss_v.cpu().numpy())
            surf_pv.append(loss_s.cpu().numpy())
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(test_dataset)}")

    vol_pv_arr = np.array(vol_pv).mean(axis=0)
    surf_pv_arr = np.array(surf_pv).mean(axis=0)
    field_lines = [
        "=" * 70,
        f"Test set ({len(test_dataset)} cases) field-level MSE — BEST checkpoint  [{save_path}]",
        "=" * 70,
        f"  Vol MSE  (mean):  {np.mean(vol_pc):.6f}",
        f"  Vol MSE  (std):   {np.std(vol_pc):.6f}",
        f"  Surf MSE (mean):  {np.mean(surf_pc):.6f}",
        f"  Surf MSE (std):   {np.std(surf_pc):.6f}",
        "",
        "Per-variable Vol MSE  [Ux, Uy, p, nut]:",
        f"  {vol_pv_arr}",
        "Per-variable Surf MSE [Ux, Uy, p, nut]:",
        f"  {surf_pv_arr}",
    ]
    field_log = "\n".join(field_lines)
    print(field_log)
    field_mse_path = osp.join(log_dir, 'auto_eval_field_mse.txt')
    with open(field_mse_path, 'w') as f:
        f.write(field_log + "\n")
    print(f"[Step 2] DONE — saved to {field_mse_path}")

    # =========== STEP 3: Coefficient post-processing ===========
    print("\n" + "─" * 70)
    print("[AUTO-EVAL Step 3/3] Coefficient post-processing (Spearman + rel err)")
    print("─" * 70)

    true_coefs = np.load(osp.join(out_dir, 'true_coefs.npy'))
    pred_mean = np.load(osp.join(out_dir, 'pred_coefs_mean.npy'))
    if pred_mean.ndim == 3:
        pred_mean = pred_mean.squeeze(1)

    coef_lines = []
    for j, name in enumerate(['Cd (drag)', 'Cl (lift)']):
        sp, _ = scipy.stats.spearmanr(true_coefs[:, j], pred_mean[:, j])
        mse = np.mean((true_coefs[:, j] - pred_mean[:, j]) ** 2)
        rel = np.abs(true_coefs[:, j] - pred_mean[:, j]) / (np.abs(true_coefs[:, j]) + 1e-8)
        coef_lines.extend([
            "",
            f"=== {name} ===",
            f"  Spearman rho       : {sp:.4f}",
            f"  MSE                : {mse:.6e}",
            f"  Median |rel err|   : {np.median(rel)*100:.2f}%",
            f"  Mean   |rel err|   : {np.mean(rel)*100:.2f}%",
            f"  Max    |rel err|   : {np.max(rel)*100:.2f}%",
            f"  True range         : [{true_coefs[:,j].min():.4f}, {true_coefs[:,j].max():.4f}]",
            f"  Pred range         : [{pred_mean[:,j].min():.4f}, {pred_mean[:,j].max():.4f}]",
        ])
    coef_log = "\n".join(coef_lines)
    print(coef_log)
    coef_path = osp.join(log_dir, 'auto_eval_coef_analysis.txt')
    with open(coef_path, 'w') as f:
        f.write(coef_log + "\n")
    print(f"[Step 3] DONE — saved to {coef_path}")

    print("\n" + "=" * 70)
    print("[AUTO-EVAL] All 3 steps complete.")
    print(f"  Cd/Cl raw .npy        -> {out_dir}/")
    print(f"  Field MSE summary     -> {field_mse_path}")
    print(f"  Coef analysis summary -> {coef_path}")
    print("=" * 70)


# ============================================================================
# TRAINING LOOP
# ============================================================================

def main(device, train_dataset, val_dataset, Net, hparams, path, criterion='MSE', reg=1, val_iter=10,
         name_mod='UrbanWindViT', val_sample=True,
         auto_eval=True, my_path=None, task='full', coef_norm=None, DatasetClass=None,
         save_path=None):
    """Trains the model, then auto-runs 3-step eval if auto_eval=True.

    Extra args vs train_v8.main():
      auto_eval (bool):   if True, run the eval suite after training.
      my_path (str):      AirfRANS root data path (needed for auto-eval).
      task (str):         'full' / 'scarce' / 'reynolds' / 'aoa'.
      coef_norm (tuple):  normalization stats from Dataset(..., norm=True).
      DatasetClass:       the Dataset class used (default dataset.dataset.Dataset).
      save_path (str):    the original --save_path so we can locate metrics_*/ for eval.
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
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=2, pin_memory=True)
    start = time.time()
    best_val_metric = float('inf')

    train_loss_surf_list = []
    train_loss_vol_list = []
    loss_surf_var_list = []
    loss_vol_var_list = []
    val_surf_list = []
    val_vol_list = []
    val_surf_var_list = []
    val_vol_var_list = []

    pbar_train = tqdm(range(hparams['nb_epochs']), position=0)
    for epoch in pbar_train:
        train_dataset_sampled = []
        for data in train_dataset:
            data_sampled = data.clone()
            idx = random.sample(range(data_sampled.x.size(0)), hparams['subsampling'])
            idx = torch.tensor(idx)

            data_sampled.pos = data_sampled.pos[idx]
            data_sampled.x = data_sampled.x[idx]
            data_sampled.y = data_sampled.y[idx]
            data_sampled.surf = data_sampled.surf[idx]

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

        if val_iter is not None:
            if epoch % val_iter == val_iter - 1 or epoch == 0:
                if val_sample:
                    val_surf_vars, val_vol_vars, val_surfs, val_vols = [], [], [], []
                    for i in range(5):
                        val_dataset_sampled = []
                        for data in val_dataset:
                            data_sampled = data.clone()
                            idx = random.sample(range(data_sampled.x.size(0)), hparams['subsampling'])
                            idx = torch.tensor(idx)

                            data_sampled.pos = data_sampled.pos[idx]
                            data_sampled.x = data_sampled.x[idx]
                            data_sampled.y = data_sampled.y[idx]
                            data_sampled.surf = data_sampled.surf[idx]

                            if name_mod != 'PointNet' and name_mod != 'MLP' and name_mod != 'UrbanWindViT':
                                data_sampled.edge_index = nng.radius_graph(x=data_sampled.pos.to(device),
                                                                           r=hparams['r'], loop=True,
                                                                           max_num_neighbors=int(
                                                                               hparams['max_neighbors'])).cpu()
                            val_dataset_sampled.append(data_sampled)
                        val_loader_inner = DataLoader(val_dataset_sampled, batch_size=1, shuffle=True,
                                                      num_workers=2, pin_memory=True)
                        del val_dataset_sampled

                        val_loss, _, val_surf_var, val_vol_var, val_surf, val_vol = test(
                            device, model, val_loader_inner, criterion)
                        del val_loader_inner
                        val_surf_vars.append(val_surf_var)
                        val_vol_vars.append(val_vol_var)
                        val_surfs.append(val_surf)
                        val_vols.append(val_vol)
                    val_surf_var = np.array(val_surf_vars).mean(axis=0)
                    val_vol_var = np.array(val_vol_vars).mean(axis=0)
                    val_surf = np.array(val_surfs).mean(axis=0)
                    val_vol = np.array(val_vols).mean(axis=0)
                else:
                    val_loss, _, val_surf_var, val_vol_var, val_surf, val_vol = test(device, model, val_loader,
                                                                                     criterion)
                if criterion == 'MSE_weighted':
                    val_loss = reg * val_surf + val_vol
                if is_main:
                    print("=====validation=====")
                    print('epoch: ' + str(epoch))
                    print('val_vol: ' + str(val_vol))
                    print('val_surf: ' + str(val_surf))
                    val_metric = float(val_vol + reg * val_surf)
                    if val_metric < best_val_metric:
                        best_val_metric = val_metric
                        torch.save(
                            _unwrap(model).state_dict(),
                            osp.join(path, 'best_model.pt'),
                        )
                        print('best_val improved -> {:.4f}, saved best_model.pt'.format(val_metric))
                val_surf_list.append(val_surf)
                val_vol_list.append(val_vol)
                val_surf_var_list.append(val_surf_var)
                val_vol_var_list.append(val_vol_var)
                pbar_train.set_postfix(train_loss=train_loss, loss_surf=loss_surf, val_loss=val_loss, val_surf=val_surf)
            else:
                pbar_train.set_postfix(train_loss=train_loss, loss_surf=loss_surf, val_loss=val_loss, val_surf=val_surf)
        else:
            pbar_train.set_postfix(train_loss=train_loss, loss_surf=loss_surf)

    loss_surf_var_list = np.array(loss_surf_var_list)
    loss_vol_var_list = np.array(loss_vol_var_list)
    val_surf_var_list = np.array(val_surf_var_list)
    val_vol_var_list = np.array(val_vol_var_list)

    end = time.time()
    time_elapsed = end - start
    params_model = get_nb_trainable_params(model).astype('float')
    saveable_model = _unwrap(model)
    if is_main:
        print('Number of parameters:', params_model)
        print('Time elapsed: {0:.2f} seconds'.format(time_elapsed))
        torch.save(saveable_model, osp.join(path, 'model'))
        torch.save(saveable_model.state_dict(), osp.join(path, 'model_state_dict.pt'))

    if is_main:
        sns.set()
        for tag, lst, lst_var, ttl in [
            ('train_loss_surf', train_loss_surf_list, loss_surf_var_list, 'Train losses over the surface'),
            ('train_loss_vol', train_loss_vol_list, loss_vol_var_list, 'Train losses over the volume'),
            ('val_loss_surf', val_surf_list, val_surf_var_list, 'Validation losses over the surface'),
            ('val_loss_vol', val_vol_list, val_vol_var_list, 'Validation losses over the volume'),
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

        if val_iter is not None:
            with open(osp.join(path, name_mod + '_log.json'), 'a') as f:
                json.dump(
                    {
                        'regression': 'Total',
                        'loss': 'MSE',
                        'nb_parameters': params_model,
                        'time_elapsed': time_elapsed,
                        'hparams': hparams,
                        'loss_weights': list(LOSS_WEIGHTS),
                        'train_loss_surf': str(train_loss_surf_list[-1]),
                        'train_loss_surf_var': str(loss_surf_var_list[-1]),
                        'train_loss_vol': str(train_loss_vol_list[-1]),
                        'train_loss_vol_var': str(loss_vol_var_list[-1]),
                        'val_loss_surf': str(val_surf_list[-1]) if len(val_surf_list) else '',
                        'val_loss_surf_var': str(val_surf_var_list[-1]) if len(val_surf_var_list) else '',
                        'val_loss_vol': str(val_vol_list[-1]) if len(val_vol_list) else '',
                        'val_loss_vol_var': str(val_vol_var_list[-1]) if len(val_vol_var_list) else '',
                    }, f, indent=12, cls=NumpyEncoder
                )

    if is_distributed:
        dist.barrier()

    # =========================================================================
    # AUTO-EVAL: run the 3-step evaluation suite using best_model.pt
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
