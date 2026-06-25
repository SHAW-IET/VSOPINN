import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from torch.utils.data import DataLoader
import time
from scipy.interpolate import interp1d
import tikzplotlib

sys.path.insert(0, '../source')
import torch.nn.functional as F
from dataset import VaryGeoDataset

from pyMesh import hcubeMesh, visualize2D, plotBC, plotMesh, setAxisLabel,\
    np2cuda
from model import USCNN
from voronoi_utils import VoronoiMultiUSCNN, to4DTensor, VaryGeoDataset_PairedSolution, LearnableKeyPoints,\
    generate_voronoi_input_torch, softVoronoi
from readOF import convertOFMeshToImage, convertOFMeshToImage_StructuredMesh
from sklearn.metrics import mean_squared_error as calMSE
import Ofpp

h = 0.01


case_config = [
    {'file': 'data/lid_400.csv', 'Re': 400, 'nu': 0.0025, 'name': 'Re400'},
    {'file': 'data/lid_200.csv', 'Re': 200, 'nu': 0.005, 'name': 'Re200'},

    {'file': 'data/lid_100.csv', 'Re': 100, 'nu': 0.01, 'name': 'Re100'},
    {'file': 'data/lid_600.csv', 'Re': 600, 'nu': 0.001667, 'name': 'Re600'},
    {'file': 'data/lid_800.csv', 'Re': 800, 'nu': 0.00125, 'name': 'Re800'},
]

num_conditions = len(case_config)
print(f"Detected {num_conditions} condition configurations。")


key_indices = [
    (25, 99),
    (75, 99),
    (30, 70),
    (60, 70)
]


key_indices = [(b, a) for (a, b) in key_indices]

initial_positions = torch.tensor(
    [(y / 100.0, x / 100.0) for y, x in key_indices],
    dtype=torch.float32
)

print("Key point indices:", key_indices)

key_history = []


leftX = np.zeros((101,), dtype=float)
leftY = np.linspace(0.0, 1.0, 101)
lowX = np.linspace(0.0, 1.0, 101)
lowY = np.zeros((101,), dtype=float)
rightX = np.ones((101,), dtype=float)
rightY = np.linspace(0.0, 1.0, 101)
upX = np.linspace(0.0, 1.0, 101)
upY = np.ones((101,), dtype=float)
ny = len(leftX)
nx = len(lowX)

myMesh = hcubeMesh(leftX, leftY, rightX, rightY,
                   lowX, lowY, upX, upY, h, True, True,
                   tolMesh=1e-10, tolJoint=1)
MeshList = []
MeshList.append(myMesh)

batchSize = 1
NvarInput = 2
NvarOutput = 3
nEpochs = 30000

lr = 0.001
Ns = 1
CLIP_NORM = 1.0


loaders_dict = {}
refs_dict = {}

for idx, case in enumerate(case_config):
    print(f"Loading condition {idx + 1}/{num_conditions}: {case['name']} (file: {case['file']})...")


    data_tmp = np.genfromtxt(case['file'], delimiter=',', skip_header=1)

    OFX_flat = data_tmp[:, 0];
    OFY_flat = data_tmp[:, 1]
    OFU_flat = data_tmp[:, 2];
    OFV_flat = data_tmp[:, 3]
    OFMag_flat = data_tmp[:, 4];
    OFP_flat = data_tmp[:, 5]

    OFU = OFU_flat.reshape((ny, nx), order='C')
    OFV = OFV_flat.reshape((ny, nx), order='C')
    OFMag = OFMag_flat.reshape((ny, nx), order='C')
    OFP = OFP_flat.reshape((ny, nx), order='C')


    OFY = np.flip(OFY_flat.reshape((ny, nx), order='C'), axis=0)
    OFU = np.flip(OFU, axis=0)
    OFV = np.flip(OFV, axis=0)
    OFMag = np.flip(OFMag, axis=0)
    OFP = np.flip(OFP, axis=0)


    U_ref = torch.tensor(OFU.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)
    V_ref = torch.tensor(OFV.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)
    Mag_ref = torch.tensor(OFMag.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)
    P_ref = torch.tensor(OFP.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)


    refs_dict[idx] = {
        'U': U_ref, 'V': V_ref, 'P': P_ref, 'Mag': Mag_ref,
        'nu': case['nu'], 'name': case['name']
    }


    SolutionList = [np.stack([OFU, OFV, OFP], axis=-1)]
    dataset_temp = VaryGeoDataset_PairedSolution(MeshList, SolutionList, key_indices)


    loaders_dict[idx] = DataLoader(dataset=dataset_temp, batch_size=batchSize, shuffle=False)

print("All condition data loaded.")


model = VoronoiMultiUSCNN(h, nx, ny, NvarInput, NvarOutput, num_conditions=num_conditions).to('cuda')
key_points_model = LearnableKeyPoints(initial_positions).to('cuda')


optimizer = optim.Adam([
    {'params': model.parameters(), 'lr': 1e-3},
    {'params': key_points_model.parameters(), 'lr': 5e-2, 'weight_decay': 1e-7}
])

criterion = nn.MSELoss()
padSingleSide = 1
udfpad = nn.ConstantPad2d([padSingleSide, padSingleSide, padSingleSide, padSingleSide], 0)


def apply_boundary_conditions(u, v, p, voronoi_input):

    u_vor = voronoi_input[:, 0:1, :, :]
    v_vor = voronoi_input[:, 1:2, :, :]
    p_vor = voronoi_input[:, 2:3, :, :]
    key_mask = voronoi_input[:, 3:4] > 0.5


    u[:, :, -1, 1:-1] = 1.0
    v[:, :, -1, 1:-1] = 0.0


    u[:, :, 0, 1:-1] = 0.0
    v[:, :, 0, 1:-1] = 0.0


    u[:, :, 1:-1, 0] = 0.0
    v[:, :, 1:-1, 0] = 0.0


    u[:, :, 1:-1, -1] = 0.0
    v[:, :, 1:-1, -1] = 0.0


    u[:, :, 0, 0] = 0.5 * (u[:, :, 0, 1] + u[:, :, 1, 0])
    v[:, :, 0, 0] = 0.5 * (v[:, :, 0, 1] + v[:, :, 1, 0])


    u[:, :, 0, -1] = 0.5 * (u[:, :, 0, -2] + u[:, :, 1, -1])
    v[:, :, 0, -1] = 0.5 * (v[:, :, 0, -2] + v[:, :, 1, -1])


    u[:, :, -1, 0] = 0.5 * (u[:, :, -1, 1] + u[:, :, -2, 0])
    v[:, :, -1, 0] = 0.5 * (v[:, :, -1, 1] + v[:, :, -2, 0])


    u[:, :, -1, -1] = 0.5 * (u[:, :, -1, -2] + u[:, :, -2, -1])
    v[:, :, -1, -1] = 0.5 * (v[:, :, -1, -2] + v[:, :, -2, -1])

    return u, v, p


def dfdx(f, dydeta, dydxi, Jinv):
    dfdxi_internal = (-f[:, :, :, 4:] + 8 * f[:, :, :, 3:-1] - 8 * f[:, :, :, 1:-3] + f[:, :, :, 0:-4]) / 12 / h
    dfdxi_left = (-11 * f[:, :, :, 0:-3] + 18 * f[:, :, :, 1:-2] - 9 * f[:, :, :, 2:-1] + 2 * f[:, :, :, 3:]) / 6 / h
    dfdxi_right = (11 * f[:, :, :, 3:] - 18 * f[:, :, :, 2:-1] + 9 * f[:, :, :, 1:-2] - 2 * f[:, :, :, 0:-3]) / 6 / h
    dfdxi = torch.cat((dfdxi_left[:, :, :, 0:2], dfdxi_internal, dfdxi_right[:, :, :, -2:]), 3)
    dfdeta_internal = (-f[:, :, 4:, :] + 8 * f[:, :, 3:-1, :] - 8 * f[:, :, 1:-3, :] + f[:, :, 0:-4, :]) / 12 / h
    dfdeta_low = (-11 * f[:, :, 0:-3, :] + 18 * f[:, :, 1:-2, :] - 9 * f[:, :, 2:-1, :] + 2 * f[:, :, 3:, :]) / 6 / h
    dfdeta_up = (11 * f[:, :, 3:, :] - 18 * f[:, :, 2:-1, :] + 9 * f[:, :, 1:-2, :] - 2 * f[:, :, 0:-3, :]) / 6 / h
    dfdeta = torch.cat((dfdeta_low[:, :, 0:2, :], dfdeta_internal, dfdeta_up[:, :, -2:, :]), 2)
    dfdx = Jinv * (dfdxi * dydeta - dfdeta * dydxi)
    return dfdx


def dfdy(f, dxdxi, dxdeta, Jinv):
    dfdxi_internal = (-f[:, :, :, 4:] + 8 * f[:, :, :, 3:-1] - 8 * f[:, :, :, 1:-3] + f[:, :, :, 0:-4]) / 12 / h
    dfdxi_left = (-11 * f[:, :, :, 0:-3] + 18 * f[:, :, :, 1:-2] - 9 * f[:, :, :, 2:-1] + 2 * f[:, :, :, 3:]) / 6 / h
    dfdxi_right = (11 * f[:, :, :, 3:] - 18 * f[:, :, :, 2:-1] + 9 * f[:, :, :, 1:-2] - 2 * f[:, :, :, 0:-3]) / 6 / h
    dfdxi = torch.cat((dfdxi_left[:, :, :, 0:2], dfdxi_internal, dfdxi_right[:, :, :, -2:]), 3)

    dfdeta_internal = (-f[:, :, 4:, :] + 8 * f[:, :, 3:-1, :] - 8 * f[:, :, 1:-3, :] + f[:, :, 0:-4, :]) / 12 / h
    dfdeta_low = (-11 * f[:, :, 0:-3, :] + 18 * f[:, :, 1:-2, :] - 9 * f[:, :, 2:-1, :] + 2 * f[:, :, 3:, :]) / 6 / h
    dfdeta_up = (11 * f[:, :, 3:, :] - 18 * f[:, :, 2:-1, :] + 9 * f[:, :, 1:-2, :] - 2 * f[:, :, 0:-3, :]) / 6 / h
    dfdeta = torch.cat((dfdeta_low[:, :, 0:2, :], dfdeta_internal, dfdeta_up[:, :, -2:, :]), 2)
    dfdy = Jinv * (dfdeta * dxdxi - dfdxi * dxdeta)
    return dfdy


def train(epoch):

    total_loss_val = 0
    err_dict = {'u': 0, 'v': 0, 'p': 0, 'mag': 0, 'av': 0}

    startTime = time.time()


    key_positions = key_points_model.get_normalized()

    idx_float = key_points_model((ny, nx))

    optimizer.zero_grad()


    combined_loss = 0


    for case_idx in range(num_conditions):
        loader = loaders_dict[case_idx]
        ref = refs_dict[case_idx]
        current_nu = ref['nu']


        batch = next(iter(loader))
        [JJInv, coord_tensor, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta,
         sol_u, sol_v, sol_p] = to4DTensor(batch)


        coord_2d = coord_tensor[0]
        sol_u_2d = sol_u[0, 0]
        sol_v_2d = sol_v[0, 0]
        sol_p_2d = sol_p[0, 0]


        voronoi_input = softVoronoi(
            coord_2d, sol_u_2d, sol_v_2d, sol_p_2d,
            key_positions,
            (ny, nx), 80)


        output = model(coord_tensor, voronoi_input, condition_idx=case_idx)
        output_pad = udfpad(output)


        u = output_pad[:, 0:1, :, :]
        v = output_pad[:, 1:2, :, :]
        p = output_pad[:, 2:3, :, :]

        u, v, p = apply_boundary_conditions(u, v, p, voronoi_input)


        ux = dfdx(u, dydeta, dydxi, Jinv);
        uy = dfdy(u, dxdxi, dxdeta, Jinv)
        vx = dfdx(v, dydeta, dydxi, Jinv);
        vy = dfdy(v, dxdxi, dxdeta, Jinv)
        px = dfdx(p, dydeta, dydxi, Jinv);
        py = dfdy(p, dxdxi, dxdeta, Jinv)
        uxx = dfdx(ux, dydeta, dydxi, Jinv);
        uyy = dfdy(uy, dxdxi, dxdeta, Jinv)
        vxx = dfdx(vx, dydeta, dydxi, Jinv);
        vyy = dfdy(vy, dxdxi, dxdeta, Jinv)

        Rc = ux + vy
        Ru = u * ux + v * uy + px - current_nu * (uxx + uyy)
        Rv = u * vx + v * vy + py - current_nu * (vxx + vyy)

        sl = slice(1, -1)
        loss_pde = (
                criterion(Rc[:, :, sl, sl], torch.zeros_like(Rc[:, :, sl, sl])) +
                criterion(Ru[:, :, sl, sl], torch.zeros_like(Ru[:, :, sl, sl])) +
                criterion(Rv[:, :, sl, sl], torch.zeros_like(Rv[:, :, sl, sl]))
        )


        grid_pts = torch.stack([
            2 * idx_float[:, 1] - 1,
            2 * idx_float[:, 0] - 1
        ], dim=-1).view(1, -1, 1, 2)


        pred_uvp = F.grid_sample(output_pad, grid_pts, mode='bilinear', align_corners=True).view(3, -1)
        u_pred, v_pred, p_pred = pred_uvp


        ref_uvp = F.grid_sample(torch.cat([ref['U'], ref['V'], ref['P']], dim=1),
                                grid_pts, align_corners=True).view(3, -1)
        u_ref, v_ref, p_ref = ref_uvp

        loss_data = (criterion(u_pred, u_ref) +
                     criterion(v_pred, v_ref) +
                     criterion(p_pred, p_ref))


        loss_case = loss_pde + loss_data


        combined_loss += loss_case


        with torch.no_grad():
            u_internal = u[0, 0, 1:-1, 1:-1]
            v_internal = v[0, 0, 1:-1, 1:-1]
            p_internal = p[0, 0, 1:-1, 1:-1]
            Mag_internal = torch.sqrt(u_internal ** 2 + v_internal ** 2)

            u_ref_internal = ref['U'][0, 0, 1:-1, 1:-1]
            v_ref_internal = ref['V'][0, 0, 1:-1, 1:-1]
            p_ref_internal = ref['P'][0, 0, 1:-1, 1:-1]
            Mag_ref_internal = ref['Mag'][0, 0, 1:-1, 1:-1]

            err_dict['u'] += (torch.sqrt(torch.mean((u_internal - u_ref_internal) ** 2)) / torch.sqrt(
                torch.mean(u_ref_internal ** 2))).item()
            err_dict['v'] += (torch.sqrt(torch.mean((v_internal - v_ref_internal) ** 2)) / torch.sqrt(
                torch.mean(v_ref_internal ** 2))).item()
            err_dict['p'] += (torch.sqrt(torch.mean((p_internal - p_ref_internal) ** 2)) / torch.sqrt(
                torch.mean(p_ref_internal ** 2))).item()
            err_dict['mag'] += (torch.sqrt(torch.mean((Mag_internal - Mag_ref_internal) ** 2)) / torch.sqrt(
                torch.mean(Mag_ref_internal ** 2))).item()


    combined_loss.backward()


    for name, param in list(model.named_parameters()) + list(key_points_model.named_parameters()):
        if param.grad is not None and not torch.all(torch.isfinite(param.grad)):
            print(f"[WARN] Detected non-finite grad in {name}; zeroing it.")
            param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)

    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
    torch.nn.utils.clip_grad_norm_(key_points_model.parameters(), CLIP_NORM)

    for param in key_points_model.parameters():
        if param.grad is not None:
            param.grad = torch.nan_to_num(param.grad)
    for param in model.parameters():
        if param.grad is not None:
            param.grad = torch.nan_to_num(param.grad)

    optimizer.step()

    with torch.no_grad():
        key_points_model.raw.clamp_(-4.6, 4.6)


    if epoch % 100 == 0:
        print("Optimized key points (normalized):")
        scale = idx_float.new_tensor([ny - 1, nx - 1])
        idx_int = (idx_float * scale).detach().round().long()
        key_indices_dyn = [(iy.item(), ix.item()) for iy, ix in idx_int]

        for i, pos in enumerate(key_points_model.get_normalized().detach().cpu().numpy()):
            actual_pos = pos * [nx - 1, ny - 1]
            print(f"Point {i + 1}: ({actual_pos[0]:.2f}, {actual_pos[1]:.2f})")


        avg_u = err_dict['u'] / num_conditions
        avg_v = err_dict['v'] / num_conditions
        avg_p = err_dict['p'] / num_conditions
        avg_mag = err_dict['mag'] / num_conditions
        err_av = (avg_u + avg_v + avg_p + avg_mag) / 4.0

        print(f"[Epoch {epoch:4d}] Total Loss = {combined_loss.item():.2e}")
        print(f"Avg L2 Errors: U={avg_u:.4f}, V={avg_v:.4f}, P={avg_p:.4f}, Mag={avg_mag:.4f}, AV={err_av:.4f}")


        key_history.append(key_points_model.get_normalized().detach().cpu().numpy())


    if epoch % 3000 == 0:

        vis_idx = 0
        vis_ref = refs_dict[vis_idx]


        with torch.no_grad():
            batch = next(iter(loaders_dict[vis_idx]))
            [_, coord_tensor, _, _, _, _, _, _, _, _, sol_u, sol_v, sol_p] = to4DTensor(batch)
            coord_2d = coord_tensor[0]

            voronoi_input = softVoronoi(coord_2d, sol_u[0, 0], sol_v[0, 0], sol_p[0, 0], key_positions, (ny, nx), 80)

            output = model(coord_tensor, voronoi_input, condition_idx=vis_idx)
            output_pad = udfpad(output)
            u_vis = output_pad[:, 0:1, :, :];
            v_vis = output_pad[:, 1:2, :, :];
            p_vis = output_pad[:, 2:3, :, :]
            u_vis, v_vis, p_vis = apply_boundary_conditions(u_vis, v_vis, p_vis, voronoi_input)


        X = coord_tensor[0, 0, :, :].cpu().numpy()
        Y = coord_tensor[0, 1, :, :].cpu().numpy()
        U_p = u_vis[0, 0, :, :].cpu().detach().numpy()
        V_p = v_vis[0, 0, :, :].cpu().detach().numpy()
        Mag_p = np.sqrt(U_p ** 2 + V_p ** 2)
        P_p = p_vis[0, 0, :, :].cpu().detach().numpy()

        U_r = vis_ref['U'][0, 0, :, :].cpu().numpy()
        V_r = vis_ref['V'][0, 0, :, :].cpu().numpy()
        Mag_r = vis_ref['Mag'][0, 0, :, :].cpu().numpy()
        P_r = vis_ref['P'][0, 0, :, :].cpu().numpy()


        u_min = min(U_p.min(), U_r.min());
        u_max = max(U_p.max(), U_r.max())
        v_min = min(V_p.min(), V_r.min());
        v_max = max(V_p.max(), V_r.max())
        mag_min = min(Mag_p.min(), Mag_r.min());
        mag_max = max(Mag_p.max(), Mag_r.max())
        p_min = min(P_p.min(), P_r.min());
        p_max = max(P_p.max(), P_r.max())


        ranges = {
            'U': (u_min, u_max),
            'V': (v_min, v_max),
            'P': (p_min, p_max),
            'Velocity Magnitude': (mag_min, mag_max)
        }


        cmap = 'viridis'

        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        for i, (pred, ref, name) in enumerate(zip(
                [U_p, V_p, P_p, Mag_p], [U_r, V_r, P_r, Mag_r], ['U', 'V', 'P', 'Velocity Magnitude']
        )):
            vmin, vmax = ranges[name]


            im0 = axes[0, i].contourf(X, Y, pred, levels=20, vmin=vmin, vmax=vmax, cmap=cmap)
            axes[0, i].set_title(f'Predicted {name} ({vis_ref["name"]})')
            plt.colorbar(im0, ax=axes[0, i])


            im1 = axes[1, i].contourf(X, Y, ref, levels=20, vmin=vmin, vmax=vmax, cmap=cmap)
            axes[1, i].set_title(f'Reference {name}')
            plt.colorbar(im1, ax=axes[1, i])


            error = np.abs(pred - ref)
            im2 = axes[2, i].contourf(X, Y, error, levels=20, cmap='hot')
            axes[2, i].set_title(f'Abs Error {name}')
            plt.colorbar(im2, ax=axes[2, i])


            scale = idx_float.new_tensor([ny - 1, nx - 1])
            idx_int = (idx_float * scale).detach().round().long()
            key_indices_dyn = [(iy.item(), ix.item()) for iy, ix in idx_int]

            for point_idx, (iy, ix) in enumerate(key_indices_dyn):

                x_val = X[iy, ix]
                y_val = Y[iy, ix]
                for row in range(3):
                    axes[row, i].plot(x_val, y_val, 'ro', markersize=6, markeredgewidth=1, markerfacecolor='none')
                    axes[row, i].text(x_val, y_val, str(point_idx + 1), color='white', fontsize=8, ha='center',
                                      va='center')


        plt.suptitle(f'Epoch {epoch}: Key Points (red circles) - {vis_ref["name"]}', fontsize=16)
        plt.tight_layout()
        os.makedirs('./output/improved/6 with_data_points - optimized_data_points - multi_conditions/', exist_ok=True)
        plt.savefig(f'./output/improved/6 with_data_points - optimized_data_points - multi_conditions/epoch_{epoch:04d}.png', dpi=150)
        plt.close()


    avg_u = err_dict['u'] / num_conditions
    avg_v = err_dict['v'] / num_conditions
    avg_p = err_dict['p'] / num_conditions
    avg_mag = err_dict['mag'] / num_conditions
    err_av = (avg_u + avg_v + avg_p + avg_mag) / 4.0

    return combined_loss.item(), avg_u, avg_v, avg_p, avg_mag, err_av


if __name__ == '__main__':

    losses = []
    errors_u = []
    errors_v = []
    errors_p = []
    errors_mag = []
    errors_av = []

    t0 = time.time()
    for ep in range(1, nEpochs + 1):

        loss, err_u, err_v, err_p, err_mag, err_av = train(ep)


        losses.append(loss)
        errors_u.append(err_u)
        errors_v.append(err_v)
        errors_p.append(err_p)
        errors_mag.append(err_mag)
        errors_av.append(err_av)


        if ep % 100 == 0 or ep == nEpochs:
            out_dir = "./output/improved/6 with_data_points - optimized_data_points - multi_conditions/"
            os.makedirs(out_dir, exist_ok=True)

            print(f"Saved model: model_epoch_{ep}.pth")
            torch.save(model.state_dict(), os.path.join(out_dir, f"model_epoch_{ep}.pth"))


            np.savetxt(os.path.join(out_dir, f'training_data_epoch_{ep}.csv'),
                       np.column_stack([losses, errors_u, errors_v, errors_p, errors_mag, errors_av]),
                       delimiter=',',
                       header='loss,err_u,err_v,err_p, errors_mag, errors_av')


    dt = time.time() - t0
    print(f"Total training time: {dt:.2f} seconds")

    out_dir = "./output/improved/6 with_data_points - optimized_data_points - multi_conditions/"
    os.makedirs(out_dir, exist_ok=True)


    np.savetxt(os.path.join(out_dir, 'final_training_data.csv'),
               np.column_stack([losses, errors_u, errors_v, errors_p, errors_mag, errors_av]),
               delimiter=',',
               header='loss,err_u,err_v,err_p, errors_mag, errors_av')


    key_hist_arr = np.stack(key_history, axis=0)


    flat_hist = key_hist_arr.reshape(len(key_history), -1)
    np.savetxt(os.path.join(out_dir, "keypoints_history.csv"),
               flat_hist, delimiter=",",
               header=",".join([f"{axis}{i + 1}"
                                for i in range(key_hist_arr.shape[1])
                                for axis in ("y", "x")]),
               comments='')


    np.save(os.path.join(out_dir, "keypoints_history.npy"), key_hist_arr)

    print(f"Key-point history saved to {out_dir}")


    np.savetxt(os.path.join(out_dir, 'training_time.txt'), [dt])


    plt.figure(figsize=(12, 8))


    plt.subplot(2, 1, 1)
    plt.semilogy(losses, 'b-', label='PDE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Convergence')
    plt.legend()
    plt.grid(True)


    plt.subplot(2, 1, 2)
    plt.semilogy(errors_u, 'r-', label='U Error')
    plt.semilogy(errors_v, 'g-', label='V Error')
    plt.semilogy(errors_p, 'b-', label='P Error')
    plt.xlabel('Epoch')
    plt.ylabel('Relative L2 Error')
    plt.title('Field Errors Convergence')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_convergence.png'), dpi=300)
    plt.show()


    epochs = np.arange(1, len(errors_u) + 1)
    u_arr = np.array(errors_u)
    v_arr = np.array(errors_v)
    p_arr = np.array(errors_p)
    mag_arr = np.array(errors_mag)
    av_arr = np.array(errors_av)


    def print_min(name, arr, epochs):
        idx = arr.argmin()
        print(f"{name} minimum L2 error：{arr[idx]:.4f}，at epoch {epochs[idx]}")


    print("\n—— minimum errors for all epochs ——")
    print_min("U", u_arr, epochs)
    print_min("V", v_arr, epochs)
    print_min("P", p_arr, epochs)
    print_min("Mag", mag_arr, epochs)
    print_min("AV", av_arr, epochs)


    mask = (epochs % 100 == 0)
    mask_epochs = epochs[mask]
    print("\n—— minimum errors for epochs divisible by 100 ——")
    if len(mask_epochs) > 0:
        print_min("U", u_arr[mask], mask_epochs)
        print_min("V", v_arr[mask], mask_epochs)
        print_min("P", p_arr[mask], mask_epochs)
        print_min("Mag", mag_arr[mask], mask_epochs)
        print_min("AV", av_arr[mask], mask_epochs)


    with open(os.path.join(out_dir, 'min_errors_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('—— minimum errors for all epochs ——\n')
        for name, arr in [('U', u_arr), ('V', v_arr), ('P', p_arr), ('Mag', mag_arr), ('AV', av_arr)]:
            idx = arr.argmin()
            f.write(f"{name} minimum L2 error：{arr[idx]:.4f}，at epoch {epochs[idx]}\n")

        f.write('\n—— minimum errors for epochs divisible by 100 ——\n')
        if len(mask_epochs) > 0:
            for name, arr in [('U', u_arr), ('V', v_arr), ('P', p_arr), ('Mag', mag_arr), ('AV', av_arr)]:
                sub = arr[mask]
                idx = sub.argmin()
                f.write(f"{name} minimum L2 error：{sub[idx]:.4f}，at epoch {mask_epochs[idx]}\n")

    print(f"Saved minimum error results to {os.path.join(out_dir, 'min_errors_summary.txt')}")
    print("Training completed!")
