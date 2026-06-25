import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.interpolate import griddata


BASE_DIR = Path(__file__).resolve().parent
MESH_NPZ = BASE_DIR / 'data' / 'mesh_output_v2' / 'structured_mesh_v2.npz'
TEMP_NPZ = BASE_DIR / 'data' / 'interp_output' / 'temperature_field_on_mesh.npz'
SAVE_DIR = BASE_DIR / 'output' / 'output_heat_optVoronoi'
SAVE_DIR.mkdir(parents=True, exist_ok=True)


sys.path.insert(0, str(BASE_DIR))
from dataset import VaryGeoDataset
from pyMesh import to4DTensor
from model import USCNN, VoronoiEnhancedUSCNNT
from voronoi_utils import select_key_indices, LearnableKeyPoints


h = 0.01
batchSize = 1
NvarInput = 2
NvarOutput = 1
nEpochs = 30000
lr = 0.001
padSingleSide = 1
criterion = nn.MSELoss()
udfpad = nn.ConstantPad2d([padSingleSide, padSingleSide, padSingleSide, padSingleSide], 0)


K_TABLE_T = torch.tensor([20.0, 100.0, 200.0, 300.0, 400.0,
                          500.0, 600.0, 700.0, 800.0, 900.0], dtype=torch.float32)
K_TABLE_K = torch.tensor([11.4, 12.5, 13.9, 15.3, 16.7,
                          18.1, 19.5, 21.1, 22.8, 25.1], dtype=torch.float32)


def load_mesh_from_npz(npz_path: Path):

    data = np.load(npz_path)
    mesh = SimpleNamespace(
        x=data['x'],
        y=data['y'],
        xi=data['xi'],
        eta=data['eta'],
        J_ho=data['J_ho'],
        Jinv_ho=data['Jinv_ho'],
        dxdxi_ho=data['dxdxi_ho'],
        dydxi_ho=data['dydxi_ho'],
        dxdeta_ho=data['dxdeta_ho'],
        dydeta_ho=data['dydeta_ho'],
    )
    boundary = {
        'bottom_T': data['bottom_T'].astype(np.float32),
        'right_T': data['right_T'].astype(np.float32),
        'top_T': data['top_T'].astype(np.float32),
        'left_T': data['left_T'].astype(np.float32),
    }
    return mesh, boundary


def dfdx(f, dydeta, dydxi, Jinv):
    dfdxi_internal = (-f[:, :, :, 4:] + 8 * f[:, :, :, 3:-1] - 8 * f[:, :, :, 1:-3] + f[:, :, :, 0:-4]) / 12 / h
    dfdxi_left = (-11 * f[:, :, :, 0:-3] + 18 * f[:, :, :, 1:-2] - 9 * f[:, :, :, 2:-1] + 2 * f[:, :, :, 3:]) / 6 / h
    dfdxi_right = (11 * f[:, :, :, 3:] - 18 * f[:, :, :, 2:-1] + 9 * f[:, :, :, 1:-2] - 2 * f[:, :, :, 0:-3]) / 6 / h
    dfdxi = torch.cat((dfdxi_left[:, :, :, 0:2], dfdxi_internal, dfdxi_right[:, :, :, -2:]), 3)

    dfdeta_internal = (-f[:, :, 4:, :] + 8 * f[:, :, 3:-1, :] - 8 * f[:, :, 1:-3, :] + f[:, :, 0:-4, :]) / 12 / h
    dfdeta_low = (-11 * f[:, :, 0:-3, :] + 18 * f[:, :, 1:-2, :] - 9 * f[:, :, 2:-1, :] + 2 * f[:, :, 3:, :]) / 6 / h
    dfdeta_up = (11 * f[:, :, 3:, :] - 18 * f[:, :, 2:-1, :] + 9 * f[:, :, 1:-2, :] - 2 * f[:, :, 0:-3, :]) / 6 / h
    dfdeta = torch.cat((dfdeta_low[:, :, 0:2, :], dfdeta_internal, dfdeta_up[:, :, -2:, :]), 2)
    return Jinv * (dfdxi * dydeta - dfdeta * dydxi)


def dfdy(f, dxdxi, dxdeta, Jinv):
    dfdxi_internal = (-f[:, :, :, 4:] + 8 * f[:, :, :, 3:-1] - 8 * f[:, :, :, 1:-3] + f[:, :, :, 0:-4]) / 12 / h
    dfdxi_left = (-11 * f[:, :, :, 0:-3] + 18 * f[:, :, :, 1:-2] - 9 * f[:, :, :, 2:-1] + 2 * f[:, :, :, 3:]) / 6 / h
    dfdxi_right = (11 * f[:, :, :, 3:] - 18 * f[:, :, :, 2:-1] + 9 * f[:, :, :, 1:-2] - 2 * f[:, :, :, 0:-3]) / 6 / h
    dfdxi = torch.cat((dfdxi_left[:, :, :, 0:2], dfdxi_internal, dfdxi_right[:, :, :, -2:]), 3)

    dfdeta_internal = (-f[:, :, 4:, :] + 8 * f[:, :, 3:-1, :] - 8 * f[:, :, 1:-3, :] + f[:, :, 0:-4, :]) / 12 / h
    dfdeta_low = (-11 * f[:, :, 0:-3, :] + 18 * f[:, :, 1:-2, :] - 9 * f[:, :, 2:-1, :] + 2 * f[:, :, 3:, :]) / 6 / h
    dfdeta_up = (11 * f[:, :, 3:, :] - 18 * f[:, :, 2:-1, :] + 9 * f[:, :, 1:-2, :] - 2 * f[:, :, 0:-3, :]) / 6 / h
    dfdeta = torch.cat((dfdeta_low[:, :, 0:2, :], dfdeta_internal, dfdeta_up[:, :, -2:, :]), 2)
    return Jinv * (dfdeta * dxdxi - dfdxi * dxdeta)


def interpolate_k_torch(T):

    T_nodes = K_TABLE_T.to(T.device, T.dtype)
    K_nodes = K_TABLE_K.to(T.device, T.dtype)

    T_clamped = torch.clamp(T, T_nodes[0], T_nodes[-1])
    idx_right = torch.bucketize(T_clamped.reshape(-1), T_nodes)
    idx_right = torch.clamp(idx_right, 1, len(T_nodes) - 1)
    idx_left = idx_right - 1

    T0 = T_nodes[idx_left].reshape_as(T_clamped)
    T1 = T_nodes[idx_right].reshape_as(T_clamped)
    K0 = K_nodes[idx_left].reshape_as(T_clamped)
    K1 = K_nodes[idx_right].reshape_as(T_clamped)

    w = (T_clamped - T0) / (T1 - T0 + 1e-12)
    return K0 + w * (K1 - K0)


def apply_temperature_bc(outputT, boundary_tensors, padSingleSide=1):

    B, C, H, W = outputT.shape
    assert C == 1, 'Temperature field channel count must be 1'

    ii = slice(padSingleSide, W - padSingleSide)
    jj = slice(padSingleSide, H - padSingleSide)

    bottom_T = boundary_tensors['bottom_T'].view(1, 1, 1, -1).expand(B, -1, -1, -1)
    top_T = boundary_tensors['top_T'].view(1, 1, 1, -1).expand(B, -1, -1, -1)
    left_T = boundary_tensors['left_T'].view(1, 1, -1, 1).expand(B, -1, -1, -1)
    right_T = boundary_tensors['right_T'].view(1, 1, -1, 1).expand(B, -1, -1, -1)

    outputT[:, :, 0:padSingleSide, ii] = bottom_T[:, :, :, 1:-1]
    outputT[:, :, -padSingleSide:, ii] = top_T[:, :, :, 1:-1]
    outputT[:, :, jj, 0:padSingleSide] = left_T[:, :, 1:-1, :]
    outputT[:, :, jj, -padSingleSide:] = right_T[:, :, 1:-1, :]


    outputT[:, :, 0:padSingleSide, 0:padSingleSide] = boundary_tensors['bottom_T'][0]
    outputT[:, :, 0:padSingleSide, -padSingleSide:] = boundary_tensors['bottom_T'][-1]
    outputT[:, :, -padSingleSide:, 0:padSingleSide] = boundary_tensors['top_T'][0]
    outputT[:, :, -padSingleSide:, -padSingleSide:] = boundary_tensors['top_T'][-1]
    return outputT


def relative_l2(pred, ref):
    return torch.sqrt(torch.mean((pred - ref) ** 2)) / (torch.sqrt(torch.mean(ref ** 2)) + 1e-12)


def soft_voronoi_temperature(coord, sol_t, key_pos, grid_size, alpha):
\
\

    ny, nx = grid_size
    device = coord.device

    gy, gx = torch.meshgrid(
        torch.linspace(0, ny - 1, ny, device=device),
        torch.linspace(0, nx - 1, nx, device=device),
        indexing='ij')
    grid = torch.stack([gy, gx], -1).unsqueeze(2)

    key = key_pos * key_pos.new_tensor([ny - 1, nx - 1])
    key = key.view(1, 1, -1, 2)

    d2 = ((grid - key) ** 2).sum(-1)
    diag2 = (ny - 1) ** 2 + (nx - 1) ** 2
    d2_norm = d2 / (diag2 + 1e-12)
    w = torch.softmax(-alpha * d2_norm, dim=-1)

    grid_pts = torch.stack([
        key_pos[:, 1] * 2 - 1,
        key_pos[:, 0] * 2 - 1,
    ], dim=-1).view(1, -1, 1, 2)

    key_t = F.grid_sample(
        sol_t.unsqueeze(0).unsqueeze(0),
        grid_pts,
        align_corners=True
    ).view(-1)

    t_vor = (w * key_t.view(1, 1, -1)).sum(-1)

    d_min = torch.sqrt(d2).min(-1).values
    beta = 5.0
    mask = torch.exp(-beta * d_min / (d_min.max() + 1e-12))

    vor = torch.stack([t_vor, mask], 0)
    return vor.unsqueeze(0)

def train(epoch):
    model.train()
    grid_size = (ny, nx)
    key_positions = key_points_model.get_normalized()
    idx_float = key_points_model(grid_size)

    for iteration, batch in enumerate(training_data_loader):
        [JJInv, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta] = to4DTensor(batch)

        alpha0, alpha_max = 2.0, 80.0
        alpha_t = min(alpha_max, alpha0 * (1 + epoch / 300) ** 2)

        voronoi_input = soft_voronoi_temperature(
            coord[0], T_ref[0, 0], key_positions, grid_size, alpha=alpha_t
        )

        optimizer.zero_grad()
        output = model(coord, voronoi_input)
        output_pad = udfpad(output)
        outputT = output_pad[:, 0:1, :, :]
        outputT = apply_temperature_bc(outputT, boundary_tensors, padSingleSide=padSingleSide)


        r_coord = coord[:, 1:2, :, :]
        dTdz = dfdx(outputT, dydeta, dydxi, Jinv)
        dTdr = dfdy(outputT, dxdxi, dxdeta, Jinv)

        k_val = interpolate_k_torch(outputT)
        flux_r = r_coord * k_val * dTdr
        flux_z = k_val * dTdz
        residual = dfdy(flux_r, dxdxi, dxdeta, Jinv) + r_coord * dfdx(flux_z, dydeta, dydxi, Jinv)

        sl = slice(1, -1)
        loss_pde = criterion(residual[:, :, sl, sl], torch.zeros_like(residual[:, :, sl, sl]))

        grid_pts = torch.stack([
            2 * idx_float[:, 1] - 1,
            2 * idx_float[:, 0] - 1,
        ], dim=-1).view(1, -1, 1, 2)

        t_pred_pts = F.grid_sample(outputT, grid_pts, mode='bilinear', align_corners=True).view(-1)
        t_ref_pts = F.grid_sample(T_ref, grid_pts, align_corners=True).view(-1)
        loss_data = criterion(t_pred_pts, t_ref_pts)

        loss = loss_pde + loss_data

        with torch.no_grad():
            l2_error_t = relative_l2(outputT, T_ref)
            mse_t = torch.mean((outputT - T_ref) ** 2)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            key_points_model.raw.clamp_(-4.6, 4.6)

    if epoch % 100 == 0:
        print(f"[Epoch {epoch:5d}] Total Loss = {loss.item():.6e}")
        print(f"Loss split: PDE={loss_pde.item():.6e}, Data={loss_data.item():.6e}")
        print(f"T Errors: L2={l2_error_t.item():.6f}, MSE={mse_t.item():.6f}")
        key_history.append(key_points_model.get_normalized().detach().cpu().numpy())

    return loss.item(), loss_pde.item(), loss_data.item(), l2_error_t.item(), mse_t.item()


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('This script depends on the CUDA path in pyMesh.to4DTensor; run it in a CUDA-enabled environment.')

    mesh_npz = np.load(MESH_NPZ)
    temp_npz = np.load(TEMP_NPZ)
    myMesh, boundary_np = load_mesh_from_npz(MESH_NPZ)

    ny, nx = myMesh.x.shape


    key_indices = select_key_indices("grid", ny=ny, nx=nx, N=4, pad=1)


    print("Key point indices:", key_indices)

    initial_positions = torch.tensor(
        [(iy / (ny - 1), ix / (nx - 1)) for iy, ix in key_indices],
        dtype=torch.float32
    )
    key_history = []

    model = VoronoiEnhancedUSCNNT(h, nx, ny, nVarIn=NvarInput, nVarOut=NvarOutput, initWay='kaiming').to('cuda')
    key_points_model = LearnableKeyPoints(initial_positions).to('cuda')
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': lr},
        {'params': key_points_model.parameters(), 'lr': 1e-2}
    ], weight_decay=1e-7)

    MeshList = [myMesh]
    train_set = VaryGeoDataset(MeshList)
    training_data_loader = DataLoader(dataset=train_set, batch_size=batchSize)

    T_ref_np = temp_npz['T_ref'].astype(np.float32)
    T_ref = torch.tensor(T_ref_np, dtype=torch.float32, device='cuda').reshape(1, 1, ny, nx)

    boundary_tensors = {
        key: torch.tensor(val, dtype=torch.float32, device='cuda')
        for key, val in boundary_np.items()
    }


    losses = []
    errors_t = []
    mses_t = []
    losses_pde = []
    losses_data = []
    global_min_t_error = 1e9
    best_epoch_t = -1
    best_model_path = SAVE_DIR / 'best_model_T.pth'
    best_keypoints_path = SAVE_DIR / 'best_keypoints_T.pth'

    t0 = time.time()
    for ep in range(1, nEpochs + 1):
        loss, loss_pde, loss_data, err_t, mse_t = train(ep)
        losses.append(loss)
        losses_pde.append(loss_pde)
        losses_data.append(loss_data)
        errors_t.append(err_t)
        mses_t.append(mse_t)

        if err_t < global_min_t_error:
            global_min_t_error = err_t
            best_epoch_t = ep
            print(f"    >>> [New Record] Best T Error: {err_t:.6f} at Epoch {ep}")
            torch.save(model.state_dict(), best_model_path)
            torch.save(key_points_model.state_dict(), best_keypoints_path)

        if ep % 100 == 0:
            np.savetxt(
                SAVE_DIR / 'training_log.csv',
                np.column_stack([losses, losses_pde, losses_data, errors_t, mses_t]),
                delimiter=',',
                header='loss,loss_pde,loss_data,err_t,mse_t',
                comments=''
            )

    dt = time.time() - t0
    print(f'Total training time: {dt:.2f} seconds')

    np.savetxt(
        SAVE_DIR / 'final_data.csv',
        np.column_stack([losses, losses_pde, losses_data, errors_t, mses_t]),
        delimiter=',',
        header='loss,loss_pde,loss_data,err_t,mse_t',
        comments=''
    )
    np.savetxt(SAVE_DIR / 'time.txt', [dt])
    if len(key_history) > 0:
        np.save(SAVE_DIR / 'keypoints_history.npy', np.stack(key_history, axis=0))


    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.semilogy(losses, 'b-', label='Total Loss')
    plt.semilogy(losses_pde, 'g--', label='PDE Loss')
    plt.semilogy(losses_data, 'm--', label='Data Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Convergence')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.semilogy(errors_t, 'r-', label='T Relative L2 Error')
    plt.semilogy(mses_t, 'g-', label='T MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Temperature Errors Convergence')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'training_convergence.png', dpi=300)
    plt.close()


    print(f"\nTraining Finished. Loading best model from Epoch {best_epoch_t} ...")
    if best_epoch_t != -1:
        model.load_state_dict(torch.load(best_model_path))
        key_points_model.load_state_dict(torch.load(best_keypoints_path))
    model.eval()

    first_batch = next(iter(training_data_loader))
    [_, coord_tensor, _, _, _, _, _, _, _, _] = to4DTensor(first_batch)

    key_positions = key_points_model.get_normalized()
    alpha_final = 80.0
    voronoi_input_final = soft_voronoi_temperature(
        coord_tensor[0], T_ref[0, 0], key_positions, (ny, nx), alpha=alpha_final
    )

    with torch.no_grad():
        output = model(coord_tensor, voronoi_input_final)
        output_pad = udfpad(output)
        outputT = output_pad[:, 0:1, :, :]
        outputT = apply_temperature_bc(outputT, boundary_tensors, padSingleSide=padSingleSide)

    t_pred = outputT[0, 0].detach().cpu().numpy()
    t_true = T_ref_np
    t_err = np.abs(t_pred - t_true)

    x_plot = coord_tensor[0, 0].cpu().numpy()
    y_plot = coord_tensor[0, 1].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    vmin = min(t_pred.min(), t_true.min())
    vmax = max(t_pred.max(), t_true.max())

    cf = axes[0].contourf(x_plot, y_plot, t_pred, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title('Predicted Temperature')
    axes[0].axis('scaled')
    plt.colorbar(cf, ax=axes[0])

    cf = axes[1].contourf(x_plot, y_plot, t_true, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title('Reference Temperature')
    axes[1].axis('scaled')
    plt.colorbar(cf, ax=axes[1])

    cf = axes[2].contourf(x_plot, y_plot, t_err, levels=50, cmap='inferno')
    axes[2].set_title('Absolute Error Temperature')
    axes[2].axis('scaled')
    plt.colorbar(cf, ax=axes[2])

    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'best_result_temperature.png', dpi=300)
    plt.close()


    np.savetxt(
        SAVE_DIR / 'best_temperature_field_reconstructed.csv',
        np.column_stack([
            x_plot.reshape(-1),
            y_plot.reshape(-1),
            t_true.reshape(-1),
            t_pred.reshape(-1),
            t_err.reshape(-1),
        ]),
        delimiter=',',
        header='x,y,T_true,T_pred,abs_error_T',
        comments=''
    )


    best_pos_norm = key_positions.detach().cpu().numpy()
    grid_pts = torch.stack([
        key_positions[:, 1] * 2 - 1,
        key_positions[:, 0] * 2 - 1,
    ], dim=-1).view(1, -1, 1, 2)
    with torch.no_grad():
        key_xy = F.grid_sample(coord_tensor, grid_pts, align_corners=True).view(2, -1).cpu().numpy()

    with open(SAVE_DIR / 'best_sensors_location.txt', 'w', encoding='utf-8') as f:
        f.write(f'Best Epoch: {best_epoch_t}\n')
        f.write(f'Best T Error: {global_min_t_error:.6f}\n')
        f.write('-' * 30 + '\n')
        f.write('Sensor Index (y, x) | Normalized (y, x) | Physical (x, y)\n')
        for i, pos in enumerate(best_pos_norm):
            iy = pos[0] * (ny - 1)
            ix = pos[1] * (nx - 1)
            px = key_xy[0, i]
            py = key_xy[1, i]
            f.write(f'Point {i + 1}: ({iy:.2f}, {ix:.2f}) | ({pos[0]:.4f}, {pos[1]:.4f}) | ({px:.6f}, {py:.6f})\n')

    ep_arr = np.arange(1, nEpochs + 1)
    summary_path = SAVE_DIR / 'final_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        idx_min = int(np.argmin(errors_t))
        f.write('=== Minimum Errors Summary ===\n')
        f.write(f'T Min Error: {errors_t[idx_min]:.6f} at Epoch {ep_arr[idx_min]}\n')
        f.write(f'T Min MSE: {mses_t[int(np.argmin(mses_t))]:.6f} at Epoch {ep_arr[int(np.argmin(mses_t))]}\n')
        f.write('\n=== Best Model Saved Based on T Error ===\n')
        f.write(f'Saved Epoch: {best_epoch_t}\n')
        f.write(f'Saved T Error: {global_min_t_error:.6f}\n')
        f.write(f'Final PDE Loss: {losses_pde[-1]:.6e}\n')
        f.write(f'Final Data Loss: {losses_data[-1]:.6e}\n')
