import os.path as osp
import random
import time

import numpy as np
import torch
import torch_geometric.nn as nng

import pyvista as pv

from utils.reorganize import reorganize

NU = np.array(1.56e-5)


def rsquared(predict, true):
    '''
    Args:
        predict (tensor): Predicted values, shape (N, *)
        true (tensor): True values, shape (N, *)

    Out:
        rsquared (tensor): Coefficient of determination of the prediction, shape (*,)
    '''
    mean = true.mean(dim=0)
    return 1 - ((true - predict) ** 2).sum(dim=0) / ((true - mean) ** 2).sum(dim=0)


def rel_err(a, b):
    return np.abs((a - b) / a)


def WallShearStress(Jacob_U, normals):
    S = .5 * (Jacob_U + Jacob_U.transpose(0, 2, 1))
    S = S - S.trace(axis1=1, axis2=2).reshape(-1, 1, 1) * np.eye(2)[None] / 3
    ShearStress = 2 * NU.reshape(-1, 1, 1) * S
    ShearStress = (ShearStress * normals[:, :2].reshape(-1, 1, 2)).sum(axis=2)

    return ShearStress


@torch.no_grad()
def Infer_test(device, models, hparams, data, coef_norm=None):
    # Inference procedure on new simulation
    outs = [torch.zeros_like(data.y)] * len(models)
    n_out = torch.zeros_like(data.y[:, :1])
    idx_points = set(map(tuple, data.pos[:, :2].numpy()))
    cond = True
    i = 0
    while cond:
        i += 1
        data_sampled = data.clone()
        idx = random.sample(range(data_sampled.pos.size(0)), hparams[0]['subsampling'])
        idx = torch.tensor(idx)
        idx_points = idx_points - set(map(tuple, data_sampled.pos[idx, :2].numpy()))
        # Per-node fields subsampled; graph-level fields (uinf, grid_sdf,
        # grid_sdf_grad, airfoil_pos) stay full size.
        data_sampled.pos = data_sampled.pos[idx]
        data_sampled.y = data_sampled.y[idx]
        data_sampled.surf = data_sampled.surf[idx]
        data_sampled.sdf = data_sampled.sdf[idx]
        data_sampled.sdf_grad = data_sampled.sdf_grad[idx]
        if hasattr(data_sampled, 'batch') and data_sampled.batch is not None:
            data_sampled.batch = data_sampled.batch[idx]

        # try:
        #     data_sampled.edge_index = nng.radius_graph(x = data_sampled.pos.to(device), r = hparams['r'], loop = True, max_num_neighbors = int(hparams['max_neighbors'])).cpu()
        # except KeyError:
        #     None

        out = [torch.zeros_like(data.y)] * len(models)
        tim = np.zeros(len(models))
        for n, model in enumerate(models):
            try:
                data_sampled.edge_index = nng.radius_graph(x=data_sampled.pos.to(device), r=hparams[n]['r'], loop=True,
                                                           max_num_neighbors=int(hparams[n]['max_neighbors'])).cpu()
            except KeyError:
                data_sampled.edge_index = None

            model.eval()
            data_sampled = data_sampled.to(device)
            start = time.time()
            o = model(data_sampled)
            tim[n] += time.time() - start
            out[n][idx] = o.cpu()

            outs[n] = outs[n] + out[n]
        n_out[idx] = n_out[idx] + torch.ones_like(n_out[idx])

        cond = (len(idx_points) > 0)

    for n, out in enumerate(outs):
        outs[n] = out / n_out
        if coef_norm is not None:
            # Surface no-slip BC: physical (U, nut) = 0 → in normalized space
            # the value is -mean / std for each affected channel.
            mean_out = coef_norm['mean_out']
            std_out = coef_norm['std_out']
            neg_norm = -mean_out / (std_out + 1e-8)            # [4]
            outs[n][data.surf, :2] = neg_norm[:2]
            outs[n][data.surf, 3] = neg_norm[3]
        else:
            outs[n][data.surf, :2] = torch.zeros_like(out[data.surf, :2])
            outs[n][data.surf, 3] = torch.zeros_like(out[data.surf, 3])

    return outs, tim / i


def Airfoil_test(internal, airfoil, outs, coef_norm, bool_surf):
    # Produce multiple copies of a simulation for different predictions.
    # stocker les internals, airfoils, calculer le wss, calculer le drag, le lift, plot pressure coef, plot skin friction coef, plot drag/drag, plot lift/lift
    # calcul spearsman coef, boundary layer
    internals = []
    airfoils = []
    for out in outs:
        intern = internal.copy()
        aerofoil = airfoil.copy()

        point_mesh = intern.points[bool_surf, :2]
        point_surf = aerofoil.points[:, :2]
        out = (out * (coef_norm['std_out'] + 1e-8) + coef_norm['mean_out']).numpy()
        out[bool_surf.numpy(), :2] = np.zeros_like(out[bool_surf.numpy(), :2])
        out[bool_surf.numpy(), 3] = np.zeros_like(out[bool_surf.numpy(), 3])
        intern.point_data['U'][:, :2] = out[:, :2]
        intern.point_data['p'] = out[:, 2]
        intern.point_data['nut'] = out[:, 3]

        surf_p = intern.point_data['p'][bool_surf]
        surf_p = reorganize(point_mesh, point_surf, surf_p)
        aerofoil.point_data['p'] = surf_p

        intern = intern.ptc(pass_point_data=True)
        aerofoil = aerofoil.ptc(pass_point_data=True)

        internals.append(intern)
        airfoils.append(aerofoil)

    return internals, airfoils


def Compute_coefficients(internals, airfoils, bool_surf, Uinf, angle, keep_vtk=False):
    # Compute force coefficients, if keet_vtk is True, also return the .vtu/.vtp with wall shear stress added over the airfoil and velocity gradient over the volume.

    coefs = []
    if keep_vtk:
        new_internals = []
        new_airfoils = []

    for internal, airfoil in zip(internals, airfoils):
        intern = internal.copy()
        aerofoil = airfoil.copy()

        point_mesh = intern.points[bool_surf, :2]
        point_surf = aerofoil.points[:, :2]

        intern = intern.compute_derivative(scalars='U', gradient='pred_grad')

        surf_grad = intern.point_data['pred_grad'].reshape(-1, 3, 3)[bool_surf, :2, :2]
        surf_p = intern.point_data['p'][bool_surf]

        surf_grad = reorganize(point_mesh, point_surf, surf_grad)
        surf_p = reorganize(point_mesh, point_surf, surf_p)

        Wss_pred = WallShearStress(surf_grad, -aerofoil.point_data['Normals'])
        aerofoil.point_data['wallShearStress'] = Wss_pred
        aerofoil.point_data['p'] = surf_p

        intern = intern.ptc(pass_point_data=True)
        aerofoil = aerofoil.ptc(pass_point_data=True)

        WP_int = -aerofoil.cell_data['p'][:, None] * aerofoil.cell_data['Normals'][:, :2]

        Wss_int = (aerofoil.cell_data['wallShearStress'] * aerofoil.cell_data['Length'].reshape(-1, 1)).sum(axis=0)
        WP_int = (WP_int * aerofoil.cell_data['Length'].reshape(-1, 1)).sum(axis=0)
        force = Wss_int - WP_int

        alpha = angle * np.pi / 180
        basis = np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
        force_rot = basis @ force
        coef = 2 * force_rot / Uinf ** 2
        coefs.append(coef)
        if keep_vtk:
            new_internals.append(intern)
            new_airfoils.append(aerofoil)

    if keep_vtk:
        return coefs, new_internals, new_airfoils
    else:
        return coefs


