"""V14 train: V8 train + auxiliary (Cd_proxy, Cl_proxy) supervision.

Forward returns (field_pred, coef_pred). Field MSE is computed exactly
as in V8 (with V8's per-variable weights). Aux loss is computed by
integrating ground-truth surface pressure with surface normals to get
proxy (Cd, Cl) values consistent with the model's prediction.

Total loss = field_loss_w + AUX_W * mse(coef_pred, coef_proxy_true).

Note on the proxy: data is normalized, so the integrated quantity is
not a physical Cd in N/(0.5*rho*U^2*c) units, but it IS a consistent,
informative scalar that supervises the encoder for "front-back pressure
integration". Reasonable trade-off: simpler than de-normalizing and
still puts a global signal on the encoder.
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
import torch.nn.functional as F
import torch_geometric.nn as nng
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from tqdm import tqdm

from pathlib import Path
import os.path as osp


# V8 per-variable loss weights kept in V14 for clean comparability.
LOSS_WEIGHTS = (1.0, 1.0, 1.5, 0.3)

# Auxiliary (Cd, Cl) supervision weight.
AUX_W = 0.05


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


def _coef_proxy(p_norm_surf, n_norm_surf):
    """Compute (Cd_proxy, Cl_proxy) from normalized surface fields.

    Cd_proxy = mean( p * n_x )    (front-back-asymmetric pressure projection)
    Cl_proxy = mean( p * (-n_y) ) (lift-direction pressure projection)

    Sign convention follows AirfRANS: surface normals point inward from
    the model's perspective (-aerofoil.point_data['Normals']), so a positive
    Cd corresponds to drag in the +x direction.

    p_norm_surf: [Ns]
    n_norm_surf: [Ns, 2]
    returns: [2] tensor (Cd_proxy, Cl_proxy)
    """
    if p_norm_surf.numel() == 0:
        return torch.zeros(2, device=p_norm_surf.device, dtype=p_norm_surf.dtype)
    cd = (p_norm_surf * n_norm_surf[:, 0]).mean()
    cl = (p_norm_surf * (-n_norm_surf[:, 1])).mean()
    return torch.stack([cd, cl])


def train(device, model, train_loader, optimizer, scheduler, criterion='MSE', reg=1, max_grad_norm=1.0):
    model.train()
    avg_loss_per_var = torch.zeros(4, device=device)
    avg_loss = 0
    avg_loss_surf_var = torch.zeros(4, device=device)
    avg_loss_vol_var = torch.zeros(4, device=device)
    avg_loss_surf = 0
    avg_loss_vol = 0
    avg_aux_loss = 0.0
    grad_norm_sum = 0.0
    grad_norm_max = 0.0
    iter = 0
    use_amp = torch.cuda.is_available()

    for data in train_loader:
        data_clone = data.clone()
        data_clone = data_clone.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            out, coef_pred = model(data_clone)
        out = out.float()
        coef_pred = coef_pred.float()
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

        # V8/V14 weights
        _w = torch.tensor(LOSS_WEIGHTS, device=loss_per_var.device, dtype=loss_per_var.dtype)
        _norm = _w.sum()
        loss_surf_w = (loss_surf_var * _w).sum() / _norm
        loss_vol_w = (loss_vol_var * _w).sum() / _norm
        total_loss_w = (loss_per_var * _w).sum() / _norm

        # V14 aux: integrate ground-truth p (normalized) with normalized normals.
        surf_mask = data_clone.surf
        with torch.no_grad():
            p_true_surf = targets[surf_mask, 2]
            n_true_surf = data_clone.x[surf_mask, 5:7]
            coef_true = _coef_proxy(p_true_surf, n_true_surf)
        aux_loss = F.mse_loss(coef_pred, coef_true)

        if criterion == 'MSE_weighted':
            (loss_vol_w + reg * loss_surf_w + AUX_W * aux_loss).backward()
        else:
            (total_loss_w + AUX_W * aux_loss).backward()

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
        avg_aux_loss += aux_loss.item()
        iter += 1

    return (avg_loss.cpu().data.numpy() / iter,
            avg_loss_per_var.cpu().data.numpy() / iter,
            avg_loss_surf_var.cpu().data.numpy() / iter,
            avg_loss_vol_var.cpu().data.numpy() / iter,
            avg_loss_surf.cpu().data.numpy() / iter,
            avg_loss_vol.cpu().data.numpy() / iter,
            grad_norm_sum / iter,
            grad_norm_max,
            avg_aux_loss / iter)


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
            out, _ = model(data_clone)  # discard aux at val time; we still report field metrics
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


def main(device, train_dataset, val_dataset, Net, hparams, path, criterion='MSE', reg=1, val_iter=10,
         name_mod='GraphSAGE', val_sample=True):
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
        del (train_dataset_sampled)

        (train_loss, _, loss_surf_var, loss_vol_var, loss_surf, loss_vol,
         grad_norm_avg, grad_norm_max, aux_loss_avg) = train(
            device, model, train_loader, optimizer, lr_scheduler, criterion, reg=reg
        )
        current_lr = optimizer.param_groups[0]['lr']
        if is_main:
            print('epoch: ' + str(epoch))
            print('train_loss： ' + str(train_loss))
            print('loss_vol： ' + str(loss_vol))
            print('loss_surf： ' + str(loss_surf))
            print('aux_loss(Cd/Cl): ' + str(aux_loss_avg))
            print('grad_norm: avg={:.4f} max={:.4f} | lr={:.6f}'.format(
                grad_norm_avg, grad_norm_max, current_lr))

        if criterion == 'MSE_weighted':
            train_loss = reg * loss_surf + loss_vol
        del (train_loader)

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
                        del (val_dataset_sampled)

                        val_loss, _, val_surf_var, val_vol_var, val_surf, val_vol = test(
                            device, model, val_loader_inner, criterion)
                        del (val_loader_inner)
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
                    print('val_vol： ' + str(val_vol))
                    print('val_surf： ' + str(val_surf))
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
            ('train_loss_vol',  train_loss_vol_list,  loss_vol_var_list,  'Train losses over the volume'),
            ('val_loss_surf',   val_surf_list,        val_surf_var_list,  'Validation losses over the surface'),
            ('val_loss_vol',    val_vol_list,         val_vol_var_list,   'Validation losses over the volume'),
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
                        'aux_w': AUX_W,
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
    return saveable_model
