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


def _setup_ddp():
    """Detect a torchrun-style distributed launch and initialize the process group.

    Returns (is_distributed, rank, world_size, device). On a non-distributed
    run (plain python), returns (False, 0, 1, None).
    """
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
    '''
    Return the number of trainable parameters
    '''
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
        # bf16 autocast for the heavy matmuls; H200 Tensor Cores give ~2x throughput
        # vs fp32. bf16 has the same exponent range as fp32 so no GradScaler needed.
        # Autocast is scoped to the model forward only — losses + accumulators stay
        # in fp32 to keep the running stats stable.
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

        # V8: per-variable loss weights for [Ux, Uy, p, nut].
        # nut is the dominant overfit channel (67% of test vol error) so we down-weight it;
        # p is upweighted because Cd_p / Cl both depend on accurate surface pressure.
        # The reported loss_vol/loss_surf above stay UN-weighted (paper-comparable);
        # only the gradient pulls on the weighted objective.
        _w = torch.tensor([1.0, 1.0, 1.5, 0.3], device=loss_per_var.device, dtype=loss_per_var.dtype)
        _norm = _w.sum()
        loss_surf_w = (loss_surf_var * _w).sum() / _norm
        loss_vol_w  = (loss_vol_var  * _w).sum() / _norm
        total_loss_w = (loss_per_var * _w).sum() / _norm

        if criterion == 'MSE_weighted':
            (loss_vol_w + reg * loss_surf_w).backward()
        else:
            total_loss_w.backward()

        # Gradient clipping: prevent any single batch from blowing the model out of its current basin.
        # The returned norm is BEFORE clipping — log it so we can detect early instability.
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


def main(device, train_dataset, val_dataset, Net, hparams, path, criterion='MSE', reg=1, val_iter=10,
         name_mod='GraphSAGE', val_sample=True):
    '''
        Args:
        device (str): device on which you want to do the computation.
        train_dataset (list): list of the data in the training set.
        val_dataset (list): list of the data in the validation set.
        Net (class): network to train.
        hparams (dict): hyper parameters of the network.
        path (str): where to save the trained model and the figures.
        criterion (str, optional): chose between 'MSE', 'MAE', and 'MSE_weigthed'. The latter is the volumetric MSE plus the surface MSE computed independently. Default: 'MSE'.
        reg (float, optional): weigth for the surface loss when criterion is 'MSE_weighted'. Default: 1.
        val_iter (int, optional): number of epochs between each validation step. Default: 10.
        name_mod (str, optional): type of model. Default: 'GraphSAGE'.
    '''
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
        # Peel off DDP first, then torch.compile (OptimizedModule). Saved state_dicts
        # then load cleanly into a plain model with no wrappers in scope.
        base = m.module if hasattr(m, 'module') else m
        return getattr(base, '_orig_mod', base)

    # V8: AdamW with weight_decay for L2-style regularization (helps with the
    # 16x train-val gap we saw in v7). weight_decay=0.01 is the standard ViT default.
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams['lr'], weight_decay=0.01)
    # V8: same smart schedule that won v7 — linear warmup (5% of training) then
    # cosine decay (95%). Combines fast mid-training learning with late fine-tuning.
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
         grad_norm_avg, grad_norm_max) = train(
            device, model, train_loader, optimizer, lr_scheduler, criterion, reg=reg
        )
        current_lr = optimizer.param_groups[0]['lr']
        if is_main:
            print('epoch: ' + str(epoch))
            print('train_loss： ' + str(train_loss))
            print('loss_vol： ' + str(loss_vol))
            print('loss_surf： ' + str(loss_surf))
            print('grad_norm: avg={:.4f} max={:.4f} | lr={:.6f}'.format(
                grad_norm_avg, grad_norm_max, current_lr))

        if criterion == 'MSE_weighted':
            train_loss = reg * loss_surf + loss_vol
        del (train_loader)

        # Periodic checkpoint: every 20 epochs, snapshot the model so a late blow-up
        # doesn't destroy hours of progress. Always overwrite the same file to bound disk use.
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
                    # 5 sub-iterations is enough to track training trend; original 20
                    # was for paper-grade reporting stability and adds ~5h over 350 epochs.
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

                                # if name_mod == 'GNO' or name_mod == 'MGNO':
                                #     x, edge_index = data_sampled.x, data_sampled.edge_index
                                #     x_i, x_j = x[edge_index[0], 0:2], x[edge_index[1], 0:2]
                                #     v_i, v_j = x[edge_index[0], 2:4], x[edge_index[1], 2:4]
                                #     p_i, p_j = x[edge_index[0], 4:5], x[edge_index[1], 4:5]
                                #     v_inf = torch.linalg.norm(v_i, dim = 1, keepdim = True)
                                #     sdf_i, sdf_j = x[edge_index[0], 5:6], x[edge_index[1], 5:6]
                                #     normal_i, normal_j = x[edge_index[0], 6:8], x[edge_index[1], 6:8]

                                #     data_sampled.edge_attr = torch.cat([x_i - x_j, v_i - v_j, p_i - p_j, sdf_i, sdf_j, v_inf, normal_i, normal_j], dim = 1)

                            val_dataset_sampled.append(data_sampled)
                        val_loader = DataLoader(val_dataset_sampled, batch_size=1, shuffle=True,
                                                num_workers=2, pin_memory=True)
                        del (val_dataset_sampled)

                        val_loss, _, val_surf_var, val_vol_var, val_surf, val_vol = test(device, model, val_loader,
                                                                                         criterion)
                        del (val_loader)
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
                    # Best-val checkpoint: save whenever the weighted val metric improves.
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
        # Also save state_dict for clean reload (no compile/DDP wrappers required)
        torch.save(saveable_model.state_dict(), osp.join(path, 'model_state_dict.pt'))

    if is_main:
        sns.set()
        fig_train_surf, ax_train_surf = plt.subplots(figsize=(20, 5))
        ax_train_surf.plot(train_loss_surf_list, label='Mean loss')
        ax_train_surf.plot(loss_surf_var_list[:, 0], label=r'$v_x$ loss')
        ax_train_surf.plot(loss_surf_var_list[:, 1], label=r'$v_y$ loss')
        ax_train_surf.plot(loss_surf_var_list[:, 2], label=r'$p$ loss')
        ax_train_surf.plot(loss_surf_var_list[:, 3], label=r'$\nu_t$ loss')
        ax_train_surf.set_xlabel('epochs')
        ax_train_surf.set_yscale('log')
        ax_train_surf.set_title('Train losses over the surface')
        ax_train_surf.legend(loc='best')
        fig_train_surf.savefig(osp.join(path, 'train_loss_surf.png'), dpi=150, bbox_inches='tight')

        fig_train_vol, ax_train_vol = plt.subplots(figsize=(20, 5))
        ax_train_vol.plot(train_loss_vol_list, label='Mean loss')
        ax_train_vol.plot(loss_vol_var_list[:, 0], label=r'$v_x$ loss')
        ax_train_vol.plot(loss_vol_var_list[:, 1], label=r'$v_y$ loss')
        ax_train_vol.plot(loss_vol_var_list[:, 2], label=r'$p$ loss')
        ax_train_vol.plot(loss_vol_var_list[:, 3], label=r'$\nu_t$ loss')
        ax_train_vol.set_xlabel('epochs')
        ax_train_vol.set_yscale('log')
        ax_train_vol.set_title('Train losses over the volume')
        ax_train_vol.legend(loc='best')
        fig_train_vol.savefig(osp.join(path, 'train_loss_vol.png'), dpi=150, bbox_inches='tight')

        if val_iter is not None:
            fig_val_surf, ax_val_surf = plt.subplots(figsize=(20, 5))
            ax_val_surf.plot(val_surf_list, label='Mean loss')
            ax_val_surf.plot(val_surf_var_list[:, 0], label=r'$v_x$ loss')
            ax_val_surf.plot(val_surf_var_list[:, 1], label=r'$v_y$ loss')
            ax_val_surf.plot(val_surf_var_list[:, 2], label=r'$p$ loss')
            ax_val_surf.plot(val_surf_var_list[:, 3], label=r'$\nu_t$ loss')
            ax_val_surf.set_xlabel('epochs')
            ax_val_surf.set_yscale('log')
            ax_val_surf.set_title('Validation losses over the surface')
            ax_val_surf.legend(loc='best')
            fig_val_surf.savefig(osp.join(path, 'val_loss_surf.png'), dpi=150, bbox_inches='tight')

            fig_val_vol, ax_val_vol = plt.subplots(figsize=(20, 5))
            ax_val_vol.plot(val_vol_list, label='Mean loss')
            ax_val_vol.plot(val_vol_var_list[:, 0], label=r'$v_x$ loss')
            ax_val_vol.plot(val_vol_var_list[:, 1], label=r'$v_y$ loss')
            ax_val_vol.plot(val_vol_var_list[:, 2], label=r'$p$ loss')
            ax_val_vol.plot(val_vol_var_list[:, 3], label=r'$\nu_t$ loss')
            ax_val_vol.set_xlabel('epochs')
            ax_val_vol.set_yscale('log')
            ax_val_vol.set_title('Validation losses over the volume')
            ax_val_vol.legend(loc='best')
            fig_val_vol.savefig(osp.join(path, 'val_loss_vol.png'), dpi=150, bbox_inches='tight')

            with open(osp.join(path, name_mod + '_log.json'), 'a') as f:
                json.dump(
                    {
                        'regression': 'Total',
                        'loss': 'MSE',
                        'nb_parameters': params_model,
                        'time_elapsed': time_elapsed,
                        'hparams': hparams,
                        'train_loss_surf': str(train_loss_surf_list[-1]),
                        'train_loss_surf_var': str(loss_surf_var_list[-1]),
                        'train_loss_vol': str(train_loss_vol_list[-1]),
                        'train_loss_vol_var': str(loss_vol_var_list[-1]),
                        'val_loss_surf': str(val_surf_list[-1]),
                        'val_loss_surf_var': str(val_surf_var_list[-1]),
                        'val_loss_vol': str(val_vol_list[-1]),
                        'val_loss_vol_var': str(val_vol_var_list[-1]),
                    }, f, indent=12, cls=NumpyEncoder
                )

    if is_distributed:
        dist.barrier()
    return saveable_model
