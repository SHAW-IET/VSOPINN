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

from pyMesh import hcubeMesh, visualize2D, plotBC, plotMesh, setAxisLabel, \
    np2cuda
from model import USCNN

from voronoi_utils import VoronoiMultiUSCNN, to4DTensor, VaryGeoDataset_PairedSolution, LearnableKeyPoints, \
    generate_voronoi_input_torch, softVoronoi, compute_cvt_update
from readOF import convertOFMeshToImage, convertOFMeshToImage_StructuredMesh
from sklearn.metrics import mean_squared_error as calMSE
import Ofpp

RUN_MODE = 'OPTIMIZE'
INIT_STRATEGY = 'MANUAL'
INIT_SEED = 1234


FIXED_SENSOR_CENTROIDS = [
    [0.2, 0.2],
    [0.8, 0.2],
    [0.5, 0.5],
    [0.5, 0.8]
]
MANUAL_INIT_COORDS = [
    (0.2, 0.2),
    (0.8, 0.8),
    (0.2, 0.8),
    (0.8, 0.2)
]

h = 0.01

case_config = [
    {'file': 'data/lid_100.csv', 'Re': 100, 'nu': 0.01, 'name': 'Re100'},
    {'file': 'data/lid_300.csv', 'Re': 300, 'nu': 1.0 / 300.0, 'name': 'Re300'},
    {'file': 'data/lid_400.csv', 'Re': 400, 'nu': 0.0025, 'name': 'Re400'},
    {'file': 'data/lid_800.csv', 'Re': 800, 'nu': 0.00125, 'name': 'Re800'},
]

num_conditions = len(case_config)
print(f"检测到 {num_conditions} 个工况配置。当前模式: {RUN_MODE}")

if RUN_MODE == 'OPTIMIZE':
    print(f">>> 优化模式: 正在初始化传感器位置 (策略: {INIT_STRATEGY})...")

    if INIT_STRATEGY == 'RANDOM':

        initial_positions = torch.rand(4, 2) * 0.9 + 0.05

    elif INIT_STRATEGY == 'SEED':

        print(f"    使用随机种子: {INIT_SEED}")

        g = torch.Generator()
        g.manual_seed(INIT_SEED)
        initial_positions = torch.rand(4, 2, generator=g) * 0.9 + 0.05

    elif INIT_STRATEGY == 'MANUAL':

        initial_positions = torch.tensor(MANUAL_INIT_COORDS, dtype=torch.float32)

    print(f"初始位置:\n{initial_positions}")


elif RUN_MODE == 'VALIDATE':
    print(">>> 验证模式: 使用固定的聚类中心点...")
    initial_positions = torch.tensor(FIXED_SENSOR_CENTROIDS, dtype=torch.float32)
    print(f"固定初始位置:\n{initial_positions}")

else:
    raise ValueError("RUN_MODE 必须是 'OPTIMIZE' 或 'VALIDATE'")

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
dummy_key_indices = [(50, 50)] * 4

for idx, case in enumerate(case_config):
    print(f"正在加载工况 {idx + 1}/{num_conditions}: {case['name']} (文件: {case['file']})...")


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
    dataset_temp = VaryGeoDataset_PairedSolution(MeshList, SolutionList, dummy_key_indices)


    loaders_dict[idx] = DataLoader(dataset=dataset_temp, batch_size=batchSize, shuffle=False)

print("所有工况数据加载完成。")

model = VoronoiMultiUSCNN(h, nx, ny, NvarInput, NvarOutput, num_conditions=num_conditions).to('cuda')
key_points_model = LearnableKeyPoints(initial_positions).to('cuda')

if RUN_MODE == 'OPTIMIZE':

    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 1e-3},
        {'params': key_points_model.parameters(), 'lr': 5e-2, 'weight_decay': 1e-7}
    ])
else:
    for param in key_points_model.parameters():
        param.requires_grad = False

    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 1e-3}
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

            param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)

    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
    if RUN_MODE == 'OPTIMIZE':
        torch.nn.utils.clip_grad_norm_(key_points_model.parameters(), CLIP_NORM)


    optimizer.step()

    CVT_INTERVAL = 500
    device = 'cuda'

    if epoch > 0 and epoch % CVT_INTERVAL == 0:
        print(f"\n>>> [CVT] Epoch {epoch}: Performing Sensor Optimization...")

        with torch.no_grad():

            curr_ref = torch.cat([sol_u, sol_v, sol_p], dim=1).to(device)
            curr_pred = output_pad.detach()


            diff = curr_pred - curr_ref
            error_sq = torch.sum(diff ** 2, dim=1)


            density_map = torch.sqrt(error_sq).mean(dim=0)


            current_pos_01 = key_points_model.get_normalized()



            current_pos_xy_01 = torch.flip(current_pos_01, dims=[1])
            current_pos_xy_11 = 2.0 * current_pos_xy_01 - 1.0


            new_pos_xy_11 = compute_cvt_update(current_pos_xy_11, density_map, device=device)



            new_pos_xy_01 = (new_pos_xy_11 + 1.0) / 2.0

            new_pos_yx_01 = torch.flip(new_pos_xy_01, dims=[1])

            new_pos_yx_01 = torch.clamp(new_pos_yx_01, 0.01, 0.99)
            new_raw = torch.logit(new_pos_yx_01)

            key_points_model.raw.data.copy_(new_raw)


            shift = torch.norm(new_pos_yx_01 - current_pos_01, dim=1).mean().item()
            print(f"    -> Density Peak: {density_map.max().item():.5f}")
            print(f"    -> Sensors Updated. Mean Shift: {shift:.5f}")


    if RUN_MODE == 'OPTIMIZE':
        with torch.no_grad():
            key_points_model.raw.clamp_(-4.6, 4.6)



    avg_u = err_dict['u'] / num_conditions
    avg_v = err_dict['v'] / num_conditions
    avg_p = err_dict['p'] / num_conditions
    avg_mag = err_dict['mag'] / num_conditions
    err_av = (avg_u + avg_v + avg_p + avg_mag) / 4.0

    if epoch % 100 == 0:
        print(f"[Epoch {epoch:4d}] Loss = {combined_loss.item():.2e} | Avg Mag Err = {avg_mag:.5f}")

        if RUN_MODE == 'OPTIMIZE':
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
        U_r = vis_ref['U'][0, 0, :, :].cpu().numpy()
        V_r = vis_ref['V'][0, 0, :, :].cpu().numpy()
        Mag_r = vis_ref['Mag'][0, 0, :, :].cpu().numpy()
        out_dir = "./output/完善/9 有数据点 - 优化数据点 - 多工况 - CVT/"
        os.makedirs(out_dir, exist_ok=True)

    return combined_loss.item(), avg_u, avg_v, avg_p, avg_mag, err_av

if __name__ == '__main__':

    losses = []
    errors_u = []
    errors_v = []
    errors_p = []
    errors_mag = []
    errors_av = []
    best_mag_error = float('inf')
    best_epoch = -1

    out_dir = "./output/完善/9 有数据点 - 优化数据点 - 多工况 - CVT/"
    os.makedirs(out_dir, exist_ok=True)


    if RUN_MODE == 'OPTIMIZE':

        save_prefix = "Opt"
    else:
        save_prefix = "Val"

    t0 = time.time()
    for ep in range(1, nEpochs + 1):
        loss, err_u, err_v, err_p, err_mag, err_av = train(ep)


        losses.append(loss)
        errors_u.append(err_u)
        errors_v.append(err_v)
        errors_p.append(err_p)
        errors_mag.append(err_mag)
        errors_av.append(err_av)

        if err_mag < best_mag_error:
            best_mag_error = err_mag
            best_epoch = ep


            torch.save(model.state_dict(), os.path.join(out_dir, f"{save_prefix}_best_model.pth"))



            current_best_pos = key_points_model.get_normalized().detach().cpu().numpy()

            np.savetxt(os.path.join(out_dir, f"{save_prefix}_best_sensors.csv"),
                       current_best_pos.reshape(1, -1),
                       delimiter=',',
                       header="y1,x1,y2,x2,y3,x3,y4,x4",
                       comments='')

            if ep % 100 == 0:
                print(f"[*] New Best Found at Epoch {ep}: Mag Err = {best_mag_error:.5f}. Saved.")


        if ep % 100 == 0 or ep == nEpochs:

            np.savetxt(os.path.join(out_dir, f'{save_prefix}_training_log.csv'),
                       np.column_stack([losses, errors_u, errors_v, errors_p, errors_mag, errors_av]),
                       delimiter=',',
                       header='loss,err_u,err_v,err_p, errors_mag, errors_av')

    dt = time.time() - t0
    print(f"总训练时间: {dt:.2f} 秒")
    print(f"最佳 Mag Error: {best_mag_error:.5f} (Epoch {best_epoch})")


    if RUN_MODE == 'OPTIMIZE':
        key_hist_arr = np.stack(key_history, axis=0)
        flat_hist = key_hist_arr.reshape(len(key_history), -1)
        np.savetxt(os.path.join(out_dir, "Opt_keypoints_history.csv"),
                   flat_hist, delimiter=",",
                   header=",".join([f"y{i + 1},x{i + 1}" for i in range(4)]),
                   comments='')
        print(f"优化轨迹已保存。")

    print("训练结束。")