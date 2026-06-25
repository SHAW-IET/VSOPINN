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
from voronoi_utils import VoronoiEnhancedUSCNN, to4DTensor, VaryGeoDataset_PairedSolution, LearnableKeyPoints,\
    generate_voronoi_input_torch, softVoronoi, VoronoiAttention2USCNN
from readOF import convertOFMeshToImage, convertOFMeshToImage_StructuredMesh
from sklearn.metrics import mean_squared_error as calMSE
import Ofpp

OUT_DIR = "./output/improved/recomputed_boundary_2pts_center_2pts/5 optimized_sensors_attention"
os.makedirs(OUT_DIR, exist_ok=True)

h = 0.01

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

file_path = 'data/lid_100.csv'
data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
print(data.shape)

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
batchSize = 1
NvarInput = 2
NvarOutput = 3
nEpochs = 30000

lr = 0.001
Ns = 1
nu = 0.01
CLIP_NORM = 1.0

model = VoronoiAttention2USCNN(h, nx, ny, NvarInput, NvarOutput).to('cuda')
key_points_model = LearnableKeyPoints(initial_positions).to('cuda')

optimizer = optim.Adam([
    {'params': model.parameters(), 'lr': 1e-3},
    {'params': key_points_model.parameters(), 'lr': 5e-2, 'weight_decay': 1e-7}
])

criterion = nn.MSELoss()

padSingleSide = 1
udfpad = nn.ConstantPad2d([padSingleSide, padSingleSide, padSingleSide, padSingleSide],
                          0)
MeshList = []
MeshList.append(myMesh)

OFX_flat = data[:, 0]
OFY_flat = data[:, 1]
OFU_flat = data[:, 2]
OFV_flat = data[:, 3]
OFMag_flat = data[:, 4]
OFP_flat = data[:, 5]

OFX = OFX_flat.reshape((ny, nx), order='C')
OFY = OFY_flat.reshape((ny, nx), order='C')
OFU = OFU_flat.reshape((ny, nx), order='C')
OFV = OFV_flat.reshape((ny, nx), order='C')
OFMag = OFMag_flat.reshape((ny, nx), order='C')
OFP = OFP_flat.reshape((ny, nx), order='C')


OFY = np.flip(OFY, axis=0)
OFU = np.flip(OFU, axis=0)
OFV = np.flip(OFV, axis=0)
OFMag = np.flip(OFMag, axis=0)
OFP = np.flip(OFP, axis=0)

U_ref = torch.tensor(OFU.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)
V_ref = torch.tensor(OFV.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)
Mag_ref = torch.tensor(OFMag.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)
P_ref = torch.tensor(OFP.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)

SolutionList = [np.stack([OFU, OFV, OFP], axis=-1)]

train_set = VaryGeoDataset_PairedSolution(MeshList, SolutionList, key_indices)

training_data_loader = DataLoader(dataset=train_set,
                                  batch_size=batchSize)

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
    global U_ref, V_ref, P_ref
    loss_data = 0
    startTime = time.time()
    xRes = 0
    yRes = 0
    mRes = 0
    eU = 0
    eV = 0
    eP = 0

    grid_size = (ny, nx)


    key_positions = key_points_model.get_normalized()


    idx_float = key_points_model(grid_size)

    for iteration, batch in enumerate(training_data_loader):
        [JJInv, coord_tensor, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta,
         sol_u, sol_v, sol_p] = to4DTensor(batch)


        coord_2d = coord_tensor[0]
        sol_u_2d = sol_u[0, 0]
        sol_v_2d = sol_v[0, 0]
        sol_p_2d = sol_p[0, 0]

        voronoi_input = softVoronoi(
            coord_2d, sol_u_2d, sol_v_2d, sol_p_2d,
            key_positions,
            grid_size, 80)

        optimizer.zero_grad()
        output = model(coord_tensor, voronoi_input)
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
        Ru = u * ux + v * uy + px - nu * (uxx + uyy)
        Rv = u * vx + v * vy + py - nu * (vxx + vyy)

        sl = slice(1, -1)


        grid_pts = torch.stack([
            2 * idx_float[:, 1] - 1,
            2 * idx_float[:, 0] - 1
        ], dim=-1).view(1, -1, 1, 2)


        pred_uvp = F.grid_sample(output_pad, grid_pts,
                                 mode='bilinear', align_corners=True).view(3, -1)

        u_pred, v_pred, p_pred = pred_uvp


        ref_uvp = F.grid_sample(torch.cat([U_ref, V_ref, P_ref], dim=1),
                                grid_pts, align_corners=True).view(3, -1)
        u_ref, v_ref, p_ref = ref_uvp


        loss_data = (criterion(u_pred, u_ref) +
                     criterion(v_pred, v_ref) +
                     criterion(p_pred, p_ref))

        loss_pde = (
                criterion(Rc[:, :, sl, sl], torch.zeros_like(Rc[:, :, sl, sl])) +
                criterion(Ru[:, :, sl, sl], torch.zeros_like(Ru[:, :, sl, sl])) +
                criterion(Rv[:, :, sl, sl], torch.zeros_like(Rv[:, :, sl, sl]))
        )

        loss = loss_pde + loss_data


        with torch.no_grad():

            u_internal = u[0, 0, 1:-1, 1:-1]
            v_internal = v[0, 0, 1:-1, 1:-1]
            p_internal = p[0, 0, 1:-1, 1:-1]
            Mag_internal = torch.sqrt(u_internal ** 2 + v_internal ** 2)


            u_ref_internal = U_ref[0, 0, 1:-1, 1:-1]
            v_ref_internal = V_ref[0, 0, 1:-1, 1:-1]
            p_ref_internal = P_ref[0, 0, 1:-1, 1:-1]
            Mag_ref_internal = Mag_ref[0, 0, 1:-1, 1:-1]


            l2_error_u = torch.sqrt(torch.mean((u_internal - u_ref_internal) ** 2)) / torch.sqrt(
                torch.mean(u_ref_internal ** 2))
            l2_error_v = torch.sqrt(torch.mean((v_internal - v_ref_internal) ** 2)) / torch.sqrt(
                torch.mean(v_ref_internal ** 2))
            l2_error_p = torch.sqrt(torch.mean((p_internal - p_ref_internal) ** 2)) / torch.sqrt(
                torch.mean(p_ref_internal ** 2))
            l2_error_mag = torch.sqrt(torch.mean((Mag_internal - Mag_ref_internal) ** 2)) / torch.sqrt(
                torch.mean(Mag_ref_internal ** 2))
            l2_error_av = (l2_error_u + l2_error_v + l2_error_p + l2_error_mag) / 4.0

        loss.backward()

        for name, param in list(model.named_parameters()) + list(key_points_model.named_parameters()):
            if param.grad is not None and not torch.all(torch.isfinite(param.grad)):

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
            key_points_model.raw.clamp_(-4.6,
                                        4.6)

    if epoch % 100 == 0:
        print(f"[Epoch {epoch:4d}] Loss = {loss.item():.2e} | Mag Err = {l2_error_mag.item():.5f}")

        key_history.append(key_points_model.get_normalized().detach().cpu().numpy())

    return loss.item(), l2_error_u.item(), l2_error_v.item(), l2_error_p.item(), l2_error_mag.item(), l2_error_av.item()


if __name__ == '__main__':

    losses = []
    errors_u = []
    errors_v = []
    errors_p = []
    errors_mag = []
    errors_av = []

    best_mag_error = float('inf')
    best_epoch = -1

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


            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pth"))


            current_best_pos = key_points_model.get_normalized().detach().cpu().numpy()
            np.savetxt(os.path.join(OUT_DIR, "best_sensors.csv"),
                       current_best_pos.reshape(1, -1),
                       delimiter=',', header="y1,x1,y2,x2,y3,x3,y4,x4,y5,x5,y6,x6,y7,x7,y8,x8", comments='')

            print(f"[*] New Best Found at Epoch {ep}: Mag Err = {best_mag_error:.5f}")


        if ep % 100 == 0 or ep == nEpochs:
            try:
                np.savetxt(os.path.join(OUT_DIR, 'training_log.csv'),
                           np.column_stack([losses, errors_u, errors_v, errors_p, errors_mag, errors_av]),
                           delimiter=',',
                           header='loss,err_u,err_v,err_p, errors_mag, errors_av')
            except PermissionError:
                pass

    dt = time.time() - t0
    print(f"\nTraining finished! Total time: {dt:.2f} seconds")
    print(f"Best Mag Error: {best_mag_error:.5f} (Epoch {best_epoch})")


    print("\nGenerating best result report...")


    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_model.pth")))
    model.eval()


    try:
        best_pos_data = np.loadtxt(os.path.join(OUT_DIR, "best_sensors.csv"), delimiter=',', skiprows=1)
        best_pos_data = best_pos_data.reshape(-1, 2)

        with torch.no_grad():
            eps = 1e-6
            val = torch.tensor(best_pos_data, dtype=torch.float32).clamp(eps, 1 - eps).to('cuda')
            key_points_model.raw.data = torch.logit(val)
        print("Restored best sensor positions.")
    except Exception as e:
        print(f"[Warn] Failed to restore best sensor positions: {e}。using final positions。")


    with torch.no_grad():

        batch = next(iter(training_data_loader))
        [JJInv, coord_tensor, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta,
         sol_u, sol_v, sol_p] = to4DTensor(batch)

        coord_2d = coord_tensor[0]

        key_pos = key_points_model.get_normalized()
        grid_size = (ny, nx)


        voronoi_input = softVoronoi(
            coord_2d, sol_u[0, 0], sol_v[0, 0], sol_p[0, 0],
            key_pos, grid_size, 80)

        output = model(coord_tensor, voronoi_input)
        output_pad = udfpad(output)

        u = output_pad[:, 0:1, :, :];
        v = output_pad[:, 1:2, :, :];
        p = output_pad[:, 2:3, :, :]
        u, v, p = apply_boundary_conditions(u, v, p, voronoi_input)


        X_np = coord_tensor[0, 0, :, :].cpu().numpy()
        Y_np = coord_tensor[0, 1, :, :].cpu().numpy()

        U_pred = u[0, 0, :, :].cpu().numpy()
        V_pred = v[0, 0, :, :].cpu().numpy()
        P_pred = p[0, 0, :, :].cpu().numpy()
        Mag_pred = np.sqrt(U_pred ** 2 + V_pred ** 2)

        U_true = sol_u[0, 0, :, :].cpu().numpy()
        V_true = sol_v[0, 0, :, :].cpu().numpy()
        P_true = sol_p[0, 0, :, :].cpu().numpy()
        Mag_true = np.sqrt(U_true ** 2 + V_true ** 2)

        Err_mag = np.abs(Mag_pred - Mag_true)
        Err_p = np.abs(P_pred - P_true)


    data_matrix = np.column_stack((
        X_np.flatten(), Y_np.flatten(),
        U_true.flatten(), V_true.flatten(), P_true.flatten(),
        U_pred.flatten(), V_pred.flatten(), P_pred.flatten(),
        Mag_true.flatten(), Mag_pred.flatten(), Err_mag.flatten()
    ))
    np.savetxt(os.path.join(OUT_DIR, "Best_Field_Data.csv"),
               data_matrix, delimiter=',',
               header="x,y,u_true,v_true,p_true,u_pred,v_pred,p_pred,mag_true,mag_pred,abs_error_mag",
               comments='')


    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    im0 = axes[0, 0].contourf(X_np, Y_np, Mag_pred, levels=50, cmap='viridis')
    axes[0, 0].set_title(f"Predicted Mag (Ep {best_epoch})");
    plt.colorbar(im0, ax=axes[0, 0])
    im1 = axes[0, 1].contourf(X_np, Y_np, Mag_true, levels=50, cmap='viridis')
    axes[0, 1].set_title(f"Reference Mag");
    plt.colorbar(im1, ax=axes[0, 1])
    im2 = axes[0, 2].contourf(X_np, Y_np, Err_mag, levels=50, cmap='hot')
    axes[0, 2].set_title(f"Mag Abs Error");
    plt.colorbar(im2, ax=axes[0, 2])

    im3 = axes[1, 0].contourf(X_np, Y_np, P_pred, levels=50, cmap='viridis')
    axes[1, 0].set_title(f"Predicted Pressure");
    plt.colorbar(im3, ax=axes[1, 0])
    im4 = axes[1, 1].contourf(X_np, Y_np, P_true, levels=50, cmap='viridis')
    axes[1, 1].set_title(f"Reference Pressure");
    plt.colorbar(im4, ax=axes[1, 1])
    im5 = axes[1, 2].contourf(X_np, Y_np, Err_p, levels=50, cmap='hot')
    axes[1, 2].set_title(f"Pressure Abs Error");
    plt.colorbar(im5, ax=axes[1, 2])


    current_key_indices = (key_pos * torch.tensor([ny - 1, nx - 1], device='cuda')).round().long().cpu().numpy()
    for i in range(2):
        for j in range(3):
            for pt in current_key_indices:
                axes[i, j].plot(X_np[pt[0], pt[1]], Y_np[pt[0], pt[1]], 'ro', markersize=6, markerfacecolor='none')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Best_Comparison.png"), dpi=100)
    plt.close(fig)


    with open(os.path.join(OUT_DIR, "final_summary.txt"), "w") as f:
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Minimum Mag Error: {best_mag_error:.6f}\n")
        f.write(f"Total Training Time: {dt:.2f}s\n")


    key_hist_arr = np.stack(key_history, axis=0)
    flat_hist = key_hist_arr.reshape(len(key_history), -1)
    np.savetxt(os.path.join(OUT_DIR, "keypoints_history.csv"),
               flat_hist, delimiter=",",
               header=",".join([f"{axis}{i + 1}" for i in range(key_hist_arr.shape[1]) for axis in ("y", "x")]),
               comments='')
    np.save(os.path.join(OUT_DIR, "keypoints_history.npy"), key_hist_arr)


    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.semilogy(losses, 'b-', label='PDE Loss')
    plt.xlabel('Epoch');
    plt.ylabel('Loss');
    plt.title('Training Loss Convergence')
    plt.legend();
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.semilogy(errors_u, 'r-', label='U Error')
    plt.semilogy(errors_v, 'g-', label='V Error')
    plt.semilogy(errors_p, 'b-', label='P Error')
    plt.xlabel('Epoch');
    plt.ylabel('Relative L2 Error');
    plt.title('Field Errors Convergence')
    plt.legend();
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'training_convergence.png'), dpi=300)
    plt.show()

    print(f"All results saved to: {OUT_DIR}")
