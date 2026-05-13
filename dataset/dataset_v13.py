"""V13 dataset: adds (log10 Re, AoA) as 2 case-level scalar features.

The case name encodes flow conditions (e.g. airFoil2D_SST_<Uinf>_<AoA_deg>_...).
V8 sees Uinf as 2D vector via x[:, 2:4] but never explicitly receives Re or AoA.
AirfRANS test split is OOD by Re and AoA, so explicit injection helps the model
generalize to flow conditions it didn't see during training.

x layout becomes [N, 9]:
    x[:, 0:2] = pos
    x[:, 2:4] = uinf
    x[:, 4:5] = sdf
    x[:, 5:7] = normal
    x[:, 7]   = log10(Re), broadcast to all points (NEW)
    x[:, 8]   = AoA in radians, broadcast (NEW)

Re is approximated as Uinf * c / nu with c = 1, nu = 1.5e-5 (air @ STP).
"""
import numpy as np
import pyvista as pv
from utils.reorganize import reorganize
import os.path as osp

import torch
from torch_geometric.data import Data

from tqdm import tqdm

from dataset.dataset import cell_sampling_2d, cell_sampling_1d


_KIN_VISC = 1.5e-5  # m^2/s, air at standard conditions
_CHORD = 1.0        # AirfRANS normalises to chord = 1


def _flow_features(case_name, n_points):
    """Return [n_points, 2] tile of (log10 Re, AoA_rad)."""
    Uinf = float(case_name.split('_')[2])
    alpha_deg = float(case_name.split('_')[3])
    Re = max(Uinf * _CHORD / _KIN_VISC, 1.0)
    log_re = np.log10(Re).astype(np.float32)
    aoa_rad = np.float32(alpha_deg * np.pi / 180.0)
    return np.tile(np.array([log_re, aoa_rad], dtype=np.float32), (n_points, 1))


def Dataset(set, norm=False, coef_norm=None, crop=None, sample=None, n_boot=int(5e5), surf_ratio=.1,
            my_path='/data/path'):
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
            target = sampled_points[:, 7:]
            surf_pos_arr = surf_sampled_points[:, :2]
            surf_init = surf_sampled_points[:, :7]
            surf_target = surf_sampled_points[:, 7:]

            # V13: append Re/AoA per point
            flow_vol = _flow_features(s, init.shape[0])
            flow_surf = _flow_features(s, surf_init.shape[0])
            init = np.concatenate([init, flow_vol], axis=1)            # [Nv, 9]
            surf_init = np.concatenate([surf_init, flow_surf], axis=1) # [Ns, 9]

            surf = torch.cat([torch.zeros(len(pos)), torch.ones(len(surf_pos_arr))], dim=0)
            pos_t = torch.cat([torch.tensor(pos, dtype=torch.float), torch.tensor(surf_pos_arr, dtype=torch.float)], dim=0)
            x = torch.cat([torch.tensor(init, dtype=torch.float), torch.tensor(surf_init, dtype=torch.float)], dim=0)
            y = torch.cat([torch.tensor(target, dtype=torch.float), torch.tensor(surf_target, dtype=torch.float)],
                          dim=0)
            init = np.concatenate([init, surf_init], axis=0)
            target = np.concatenate([target, surf_target], axis=0)
            pos_for_data = pos_t

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
            init_base = np.concatenate([pos, attr[:, :5]], axis=1)  # [N, 7]
            target = attr[:, 5:]

            # V13: append flow features
            flow = _flow_features(s, init_base.shape[0])
            init = np.concatenate([init_base, flow], axis=1)  # [N, 9]

            surf = torch.tensor(surf_bool)
            pos_for_data = torch.tensor(pos, dtype=torch.float)
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

        data = Data(pos=pos_for_data, x=x, y=y, surf=surf.bool(), airfoil_pos=airfoil_pos)
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
