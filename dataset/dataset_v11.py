"""V11 dataset: applies log transform to nut target before normalization.

The only difference from dataset.dataset.Dataset is that target[:, 3] (nut)
is replaced with log(nut + EPS) before computing normalization statistics.

This compresses nut's 6-orders-of-magnitude dynamic range so that linear MSE
becomes balanced across the channel. At inference, predictions for the nut
channel must be inverse-transformed (exp) before computing physical-unit
metrics — see test_field_mse_v11.py / eval_score_v11.py.
"""
import numpy as np
import pyvista as pv
from utils.reorganize import reorganize
import os.path as osp

import torch
from torch_geometric.data import Data

from tqdm import tqdm

from dataset.dataset import cell_sampling_2d, cell_sampling_1d


NUT_LOG_EPS = 1e-8


def Dataset(set, norm=False, coef_norm=None, crop=None, sample=None, n_boot=int(5e5), surf_ratio=.1,
            my_path='/data/path'):
    """V11 variant of dataset.Dataset with log(nut + eps) target transform."""
    if norm and coef_norm is not None:
        raise ValueError('If coef_norm is not None and norm is True, the normalization will be done via coef_norm')

    dataset = []

    for k, s in enumerate(tqdm(set)):
        internal = pv.read(osp.join(my_path, s, s + '_internal.vtu'))
        aerofoil = pv.read(osp.join(my_path, s, s + '_aerofoil.vtp'))
        airfoil_pos = torch.tensor(aerofoil.points[:, :2], dtype=torch.float)
        internal = internal.compute_cell_sizes(length=False, volume=False)

        if crop is not None:
            bounds = (crop[0], crop[1], crop[2], crop[3], 0, 1)
            internal = internal.clip_box(bounds=bounds, invert=False, crinkle=True)

        if sample is not None:
            if sample == 'uniform':
                p = internal.cell_data['Area'] / internal.cell_data['Area'].sum()
                sampled_cell_indices = np.random.choice(internal.n_cells, size=n_boot, p=p)
                surf_p = aerofoil.cell_data['Length'] / aerofoil.cell_data['Length'].sum()
                sampled_line_indices = np.random.choice(aerofoil.n_cells, size=int(n_boot * surf_ratio), p=surf_p)
            elif sample == 'mesh':
                sampled_cell_indices = np.random.choice(internal.n_cells, size=n_boot)
                sampled_line_indices = np.random.choice(aerofoil.n_cells, size=int(n_boot * surf_ratio))

            cell_dict = internal.cells.reshape(-1, 5)[sampled_cell_indices, 1:]
            cell_points = internal.points[cell_dict]
            line_dict = aerofoil.lines.reshape(-1, 3)[sampled_line_indices, 1:]
            line_points = aerofoil.points[line_dict]

            geom = -internal.point_data['implicit_distance'][cell_dict, None]
            Uinf, alpha = float(s.split('_')[2]), float(s.split('_')[3]) * np.pi / 180
            u = (np.array([np.cos(alpha), np.sin(alpha)]) * Uinf).reshape(1, 2) * np.ones_like(
                internal.point_data['U'][cell_dict, :1])
            normal = np.zeros_like(u)

            surf_geom = np.zeros_like(aerofoil.point_data['U'][line_dict, :1])
            surf_u = (np.array([np.cos(alpha), np.sin(alpha)]) * Uinf).reshape(1, 2) * np.ones_like(
                aerofoil.point_data['U'][line_dict, :1])
            surf_normal = -aerofoil.point_data['Normals'][line_dict, :2]

            attr = np.concatenate([u, geom, normal, internal.point_data['U'][cell_dict, :2],
                                   internal.point_data['p'][cell_dict, None],
                                   internal.point_data['nut'][cell_dict, None]], axis=-1)
            surf_attr = np.concatenate([surf_u, surf_geom, surf_normal, aerofoil.point_data['U'][line_dict, :2],
                                        aerofoil.point_data['p'][line_dict, None],
                                        aerofoil.point_data['nut'][line_dict, None]], axis=-1)
            sampled_points = cell_sampling_2d(cell_points, attr)
            surf_sampled_points = cell_sampling_1d(line_points, surf_attr)

            pos = sampled_points[:, :2]
            init = sampled_points[:, :7]
            target = sampled_points[:, 7:].copy()
            surf_pos = surf_sampled_points[:, :2]
            surf_init = surf_sampled_points[:, :7]
            surf_target = surf_sampled_points[:, 7:].copy()

            # V11: log-transform nut channel (column 3) BEFORE normalization stats.
            target[:, 3] = np.log(np.maximum(target[:, 3], 0) + NUT_LOG_EPS)
            surf_target[:, 3] = np.log(np.maximum(surf_target[:, 3], 0) + NUT_LOG_EPS)

            surf = torch.cat([torch.zeros(len(pos)), torch.ones(len(surf_pos))], dim=0)
            pos = torch.cat([torch.tensor(pos, dtype=torch.float), torch.tensor(surf_pos, dtype=torch.float)], dim=0)
            x = torch.cat([torch.tensor(init, dtype=torch.float), torch.tensor(surf_init, dtype=torch.float)], dim=0)
            y = torch.cat([torch.tensor(target, dtype=torch.float), torch.tensor(surf_target, dtype=torch.float)],
                          dim=0)

        else:
            surf_bool = (internal.point_data['U'][:, 0] == 0)
            geom = -internal.point_data['implicit_distance'][:, None]
            Uinf, alpha = float(s.split('_')[2]), float(s.split('_')[3]) * np.pi / 180
            u = (np.array([np.cos(alpha), np.sin(alpha)]) * Uinf).reshape(1, 2) * np.ones_like(
                internal.point_data['U'][:, :1])
            normal = np.zeros_like(u)
            normal[surf_bool] = reorganize(aerofoil.points[:, :2], internal.points[surf_bool, :2],
                                           -aerofoil.point_data['Normals'][:, :2])

            attr = np.concatenate([u, geom, normal,
                                   internal.point_data['U'][:, :2], internal.point_data['p'][:, None],
                                   internal.point_data['nut'][:, None]], axis=-1)

            pos = internal.points[:, :2]
            init = np.concatenate([pos, attr[:, :5]], axis=1)
            target = attr[:, 5:].copy()

            # V11: log-transform nut channel (column 3) BEFORE normalization stats.
            target[:, 3] = np.log(np.maximum(target[:, 3], 0) + NUT_LOG_EPS)

            surf = torch.tensor(surf_bool)
            pos = torch.tensor(pos, dtype=torch.float)
            x = torch.tensor(init, dtype=torch.float)
            y = torch.tensor(target, dtype=torch.float)

        if norm and coef_norm is None:
            if k == 0:
                old_length = init.shape[0]
                mean_in = init.mean(axis=0, dtype=np.double)
                mean_out = target.mean(axis=0, dtype=np.double)
            else:
                new_length = old_length + init.shape[0]
                mean_in += (init.sum(axis=0, dtype=np.double) - init.shape[0] * mean_in) / new_length
                mean_out += (target.sum(axis=0, dtype=np.double) - init.shape[0] * mean_out) / new_length
                old_length = new_length

        data = Data(pos=pos, x=x, y=y, surf=surf.bool(), airfoil_pos=airfoil_pos)
        dataset.append(data)

    if norm and coef_norm is None:
        mean_in = mean_in.astype(np.single)
        mean_out = mean_out.astype(np.single)
        for k, data in enumerate(dataset):
            if k == 0:
                old_length = data.x.numpy().shape[0]
                std_in = ((data.x.numpy() - mean_in) ** 2).sum(axis=0, dtype=np.double) / old_length
                std_out = ((data.y.numpy() - mean_out) ** 2).sum(axis=0, dtype=np.double) / old_length
            else:
                new_length = old_length + data.x.numpy().shape[0]
                std_in += (((data.x.numpy() - mean_in) ** 2).sum(axis=0, dtype=np.double) - data.x.numpy().shape[
                    0] * std_in) / new_length
                std_out += (((data.y.numpy() - mean_out) ** 2).sum(axis=0, dtype=np.double) - data.x.numpy().shape[
                    0] * std_out) / new_length
                old_length = new_length

        std_in = np.sqrt(std_in).astype(np.single)
        std_out = np.sqrt(std_out).astype(np.single)

        for data in dataset:
            data.x = (data.x - mean_in) / (std_in + 1e-8)
            data.y = (data.y - mean_out) / (std_out + 1e-8)

        coef_norm = (mean_in, std_in, mean_out, std_out)
        dataset = (dataset, coef_norm)

    elif coef_norm is not None:
        for data in dataset:
            data.x = (data.x - coef_norm[0]) / (coef_norm[1] + 1e-8)
            data.y = (data.y - coef_norm[2]) / (coef_norm[3] + 1e-8)

    return dataset
