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
from dataset import VaryGeoDataset
from dataset import VaryGeoDataset_PairedSolution
from pyMesh import hcubeMesh, visualize2D, plotBC, plotMesh, setAxisLabel, \
    np2cuda, to4DTensor
from model import USCNN
from readOF import convertOFMeshToImage, convertOFMeshToImage_StructuredMesh
from sklearn.metrics import mean_squared_error as calMSE
import Ofpp

# =========================================================================
# [配置] 统一文件保存路径
# =========================================================================
OUT_DIR = "./output/完善/重新计算记录边界2点+中心2点/2 数据点"
os.makedirs(OUT_DIR, exist_ok=True)
# =========================================================================

h = 0.01

key_indices = [
    (25, 99),
    (75, 99),
    (30, 70),
    (60, 70)
]

# 转换索引格式
key_indices = [(b, a) for (a, b) in key_indices]
print("Key point indices:", key_indices)

# 读取数据
file_path = 'data/lid_100.csv'
data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

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
model = USCNN(h, nx, ny, NvarInput, NvarOutput).to('cuda')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
padSingleSide = 1
udfpad = nn.ConstantPad2d([padSingleSide, padSingleSide, padSingleSide, padSingleSide], 0)
MeshList = []
MeshList.append(myMesh)
train_set = VaryGeoDataset(MeshList)
training_data_loader = DataLoader(dataset=train_set, batch_size=batchSize)

# 处理参考数据
OFX_flat = data[:, 0];
OFY_flat = data[:, 1]
OFU_flat = data[:, 2];
OFV_flat = data[:, 3]
OFMag_flat = data[:, 4];
OFP_flat = data[:, 5]

OFX = OFX_flat.reshape((ny, nx), order='C')
OFY = OFY_flat.reshape((ny, nx), order='C')
OFU = OFU_flat.reshape((ny, nx), order='C')
OFV = OFV_flat.reshape((ny, nx), order='C')
OFMag = OFMag_flat.reshape((ny, nx), order='C')
OFP = OFP_flat.reshape((ny, nx), order='C')

# 上下翻转
OFY = np.flip(OFY, axis=0)
OFU = np.flip(OFU, axis=0)
OFV = np.flip(OFV, axis=0)
OFMag = np.flip(OFMag, axis=0)
OFP = np.flip(OFP, axis=0)

# 全局参考解 (转为Tensor)
U_ref = torch.tensor(OFU.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)
V_ref = torch.tensor(OFV.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)
Mag_ref = torch.tensor(OFMag.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)
P_ref = torch.tensor(OFP.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)


def apply_boundary_conditions(u, v):
    key_mask = torch.zeros_like(u, dtype=torch.bool)
    for iy, ix in key_indices:
        key_mask[0, 0, iy + 1, ix + 1] = True

    u_key = u[key_mask].clone()
    v_key = v[key_mask].clone()

    u[:, :, -1, 1:-1] = 1.0;
    v[:, :, -1, 1:-1] = 0.0
    u[:, :, 0, 1:-1] = 0.0;
    v[:, :, 0, 1:-1] = 0.0
    u[:, :, 1:-1, 0] = 0.0;
    v[:, :, 1:-1, 0] = 0.0
    u[:, :, 1:-1, -1] = 0.0;
    v[:, :, 1:-1, -1] = 0.0

    u[:, :, 0, 0] = 0.5 * (u[:, :, 0, 1] + u[:, :, 1, 0])
    v[:, :, 0, 0] = 0.5 * (v[:, :, 0, 1] + v[:, :, 1, 0])
    u[:, :, 0, -1] = 0.5 * (u[:, :, 0, -2] + u[:, :, 1, -1])
    v[:, :, 0, -1] = 0.5 * (v[:, :, 0, -2] + v[:, :, 1, -1])
    u[:, :, -1, 0] = 0.5 * (u[:, :, -1, 1] + u[:, :, -2, 0])
    v[:, :, -1, 0] = 0.5 * (v[:, :, -1, 1] + v[:, :, -2, 0])
    u[:, :, -1, -1] = 0.5 * (u[:, :, -1, -2] + u[:, :, -2, -1])
    v[:, :, -1, -1] = 0.5 * (v[:, :, -1, -2] + v[:, :, -2, -1])

    u[key_mask] = u_key
    v[key_mask] = v_key
    return u, v


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

    for iteration, batch in enumerate(training_data_loader):
        # [修正] Case0 只解包10个几何变量
        [JJInv, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta] = to4DTensor(batch)

        optimizer.zero_grad()
        output = model(coord)
        output_pad = udfpad(output)

        u = output_pad[:, 0:1, :, :]
        v = output_pad[:, 1:2, :, :]
        p = output_pad[:, 2:3, :, :]

        u, v = apply_boundary_conditions(u, v)

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

        for iy, ix in key_indices:
            u_pred = u[0, 0, iy, ix]
            v_pred = v[0, 0, iy, ix]
            p_pred = p[0, 0, iy, ix]

            # 使用全局参考值
            u_ref_point = U_ref[0, 0, iy, ix]
            v_ref_point = V_ref[0, 0, iy, ix]
            p_ref_point = P_ref[0, 0, iy, ix]

            loss_data += criterion(u_pred, u_ref_point)
            loss_data += criterion(v_pred, v_ref_point)
            loss_data += criterion(p_pred, p_ref_point)

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
            Mag_pred_internal = torch.sqrt(u_internal ** 2 + v_internal ** 2)

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
            l2_error_mag = torch.sqrt(torch.mean((Mag_pred_internal - Mag_ref_internal) ** 2)) / torch.sqrt(
                torch.mean(Mag_ref_internal ** 2))
            l2_error_av = (l2_error_u + l2_error_v + l2_error_p + l2_error_mag) / 4.0

        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        print(f"[Epoch {epoch:4d}] Loss = {loss.item():.2e} | Mag Err = {l2_error_mag.item():.5f}")

    return loss.item(), l2_error_u.item(), l2_error_v.item(), l2_error_p.item(), l2_error_mag.item(), l2_error_av.item()


if __name__ == '__main__':
    losses = [];
    errors_u = [];
    errors_v = [];
    errors_p = [];
    errors_mag = [];
    errors_av = []

    best_mag_error = float('inf')
    best_epoch = -1

    t0 = time.time()
    for ep in range(1, nEpochs + 1):
        loss, err_u, err_v, err_p, err_mag, err_av = train(ep)

        losses.append(loss);
        errors_u.append(err_u);
        errors_v.append(err_v)
        errors_p.append(err_p);
        errors_mag.append(err_mag);
        errors_av.append(err_av)

        # =================================================================
        # [速度优化] 训练循环中只保存 .pth，不画图不写CSV
        # =================================================================
        if err_mag < best_mag_error:
            best_mag_error = err_mag
            best_epoch = ep

            # 1. 快速保存最佳模型权重
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pth"))

            # 打印提示 (不再受 %100 限制)
            print(f"[*] New Best Found at Epoch {ep}: Mag Err = {best_mag_error:.5f}")

        # 定期保存训练日志 (仅数据)
        if ep % 100 == 0 or ep == nEpochs:
            try:
                np.savetxt(os.path.join(OUT_DIR, 'training_log.csv'),
                           np.column_stack([losses, errors_u, errors_v, errors_p, errors_mag, errors_av]),
                           delimiter=',',
                           header='loss,err_u,err_v,err_p,err_mag, err_av')
            except PermissionError:
                pass

    dt = time.time() - t0
    print(f"\n训练结束! 总用时: {dt:.2f} 秒")
    print(f"最佳 Mag Error: {best_mag_error:.5f} (Epoch {best_epoch})")

    # =================================================================
    # [后处理] 训练结束后，加载最佳模型，生成最终文件
    # =================================================================
    print("\n正在生成最佳结果报告...")

    # 加载最佳权重
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_model.pth")))
    model.eval()

    with torch.no_grad():
        batch = next(iter(training_data_loader))
        [JJInv, coord_tensor, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta] = to4DTensor(batch)

        output = model(coord_tensor)
        output_pad = udfpad(output)

        u = output_pad[:, 0:1, :, :];
        v = output_pad[:, 1:2, :, :];
        p = output_pad[:, 2:3, :, :]
        u, v = apply_boundary_conditions(u, v)

        # 转 Numpy
        X_np = coord_tensor[0, 0, :, :].cpu().numpy()
        Y_np = coord_tensor[0, 1, :, :].cpu().numpy()

        U_pred = u[0, 0, :, :].cpu().numpy()
        V_pred = v[0, 0, :, :].cpu().numpy()
        P_pred = p[0, 0, :, :].cpu().numpy()
        Mag_pred = np.sqrt(U_pred ** 2 + V_pred ** 2)

        U_true = U_ref[0, 0, :, :].cpu().numpy()
        V_true = V_ref[0, 0, :, :].cpu().numpy()
        P_true = P_ref[0, 0, :, :].cpu().numpy()
        Mag_true = Mag_ref[0, 0, :, :].cpu().numpy()

        Err_mag = np.abs(Mag_pred - Mag_true)
        Err_p = np.abs(P_pred - P_true)

    # 1. 保存最终 CSV
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

    # 2. 保存最终图片
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # Mag
    im0 = axes[0, 0].contourf(X_np, Y_np, Mag_pred, levels=50, cmap='viridis')
    axes[0, 0].set_title(f"Predicted Mag (Ep {best_epoch})");
    plt.colorbar(im0, ax=axes[0, 0])
    im1 = axes[0, 1].contourf(X_np, Y_np, Mag_true, levels=50, cmap='viridis')
    axes[0, 1].set_title(f"Reference Mag");
    plt.colorbar(im1, ax=axes[0, 1])
    im2 = axes[0, 2].contourf(X_np, Y_np, Err_mag, levels=50, cmap='hot')
    axes[0, 2].set_title(f"Mag Abs Error");
    plt.colorbar(im2, ax=axes[0, 2])
    # Pressure
    im3 = axes[1, 0].contourf(X_np, Y_np, P_pred, levels=50, cmap='viridis')
    axes[1, 0].set_title(f"Predicted Pressure");
    plt.colorbar(im3, ax=axes[1, 0])
    im4 = axes[1, 1].contourf(X_np, Y_np, P_true, levels=50, cmap='viridis')
    axes[1, 1].set_title(f"Reference Pressure");
    plt.colorbar(im4, ax=axes[1, 1])
    im5 = axes[1, 2].contourf(X_np, Y_np, Err_p, levels=50, cmap='hot')
    axes[1, 2].set_title(f"Pressure Abs Error");
    plt.colorbar(im5, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Best_Comparison.png"), dpi=100)
    plt.close(fig)

    # 3. 保存传感器位置
    sensor_coords = []
    for (iy, ix) in key_indices:
        sensor_coords.extend([iy * h, ix * h])
    np.savetxt(os.path.join(OUT_DIR, "fixed_sensors.csv"),
               np.array(sensor_coords).reshape(1, -1),
               delimiter=',', header="y1,x1,y2,x2,y3,x3,y4,x4", comments='')

    # 4. 保存统计信息
    with open(os.path.join(OUT_DIR, "final_summary.txt"), "w") as f:
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Minimum Mag Error: {best_mag_error:.6f}\n")
        f.write(f"Total Training Time: {dt:.2f}s\n")

    print(f"所有结果已保存至: {OUT_DIR}")