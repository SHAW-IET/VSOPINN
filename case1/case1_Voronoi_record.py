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
# from dataset import VaryGeoDataset_PairedSolution
from pyMesh import hcubeMesh, visualize2D, plotBC, plotMesh, setAxisLabel, \
    np2cuda
from model import USCNN
from voronoi_utils import VoronoiEnhancedUSCNN, to4DTensor, VaryGeoDataset_PairedSolutionOld, generate_voronoi_input
from view import plot_voronoi_uvp_mask

from readOF import convertOFMeshToImage, convertOFMeshToImage_StructuredMesh
from sklearn.metrics import mean_squared_error as calMSE
import Ofpp

# =========================================================================
# [配置] 统一文件保存路径
# =========================================================================
OUT_DIR = "./output/完善/重新计算记录边界2点+中心2点/3 硬Voronoi"
os.makedirs(OUT_DIR, exist_ok=True)
# =========================================================================

h = 0.01

key_indices = [
    (25, 99),
    (75, 99),
    (30, 70),
    (60, 70)
]

# 对着 abs error大的地方选点
# 要变换 x y 坐标   因为 y是第三个维度 而 x是第四个维度
key_indices = [(b, a) for (a, b) in key_indices]

print("Key point indices:", key_indices)

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
NvarInput = 2  # 输入 x y 坐标
NvarOutput = 3  # 输出从 T变为 U V P
nEpochs = 30000
# lr=0.001
lr = 0.001
Ns = 1
nu = 0.01

# model=USCNN(h,nx,ny,NvarInput,NvarOutput).to('cuda')   # h 网格间距
model = VoronoiEnhancedUSCNN(h, nx, ny, NvarInput, NvarOutput).to('cuda')

criterion = nn.MSELoss()  # 选取损失形式
optimizer = optim.Adam(model.parameters(), lr=lr)
padSingleSide = 1  # ConstantPad2d([left, right, top, bottom], value) 会把输入张量的空间维度 H×W 扩展成 (H+2)×(W+2)，四周填充值为常数 0。
udfpad = nn.ConstantPad2d([padSingleSide, padSingleSide, padSingleSide, padSingleSide],
                          0)  # 在网络输出上加上一圈“零值”边框（也就是常数填充），以便后面用有限差分卷积核在边界处也能一致地计算导数
MeshList = []
MeshList.append(myMesh)

OFX_flat = data[:, 0]  # 第一列：X 坐标  (101, )
OFY_flat = data[:, 1]  # 第二列：Y 坐标
OFU_flat = data[:, 2]  # 第三列：u 分量
OFV_flat = data[:, 3]  # 第四列：v 分量
OFMag_flat = data[:, 4]  # 第五列：Velocity Magnitude
OFP_flat = data[:, 5]  # 第六列：Pressure（P）

OFX = OFX_flat.reshape((ny, nx), order='C')  # (101, 101)
OFY = OFY_flat.reshape((ny, nx), order='C')
OFU = OFU_flat.reshape((ny, nx), order='C')
OFV = OFV_flat.reshape((ny, nx), order='C')
OFMag = OFMag_flat.reshape((ny, nx), order='C')
OFP = OFP_flat.reshape((ny, nx), order='C')

# 问题出在 OFY 应该上下颠倒  除了 OFX 都应该上下颠倒 ！！！！！！
OFY = np.flip(OFY, axis=0)  # 上下翻转
OFU = np.flip(OFU, axis=0)
OFV = np.flip(OFV, axis=0)
OFMag = np.flip(OFMag, axis=0)  # 添加速度幅值的翻转
OFP = np.flip(OFP, axis=0)

U_ref = torch.tensor(OFU.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)
V_ref = torch.tensor(OFV.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)
Mag_ref = torch.tensor(OFMag.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)
P_ref = torch.tensor(OFP.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)

SolutionList = [np.stack([OFU, OFV, OFP], axis=-1)]  # 组合成 (101, 101, 3)

train_set = VaryGeoDataset_PairedSolutionOld(MeshList, SolutionList, key_indices)

training_data_loader = DataLoader(dataset=train_set,
                                  batch_size=batchSize)

### 可视化 voronoi图 (保持原代码逻辑，但在保存时加上 try 避免报错)
coord_np = np.stack([myMesh.x, myMesh.y], axis=0)  # [2,101,101]
try:
    vor = generate_voronoi_input(coord_np, OFU, OFV, OFP, key_indices, (ny, nx))
    plot_voronoi_uvp_mask(
        vor, myMesh,
        key_indices=key_indices,
        save_path=os.path.join(OUT_DIR, 'voronoi_uvp_mask.png'),
        save_split_dir=os.path.join(OUT_DIR, 'channels'),
        show=False
    )
    print(f"Voronoi 可视化已保存到 {OUT_DIR}")
except Exception as e:
    print(f"Voronoi 可视化生成失败: {e}")


def apply_boundary_conditions(u, v, p, key_mask=None):
    """应用腔体流边界条件，保留关键点值"""
    # 创建默认掩码（如果没有提供）
    if key_mask is None:
        key_mask = torch.zeros_like(u, dtype=torch.bool)

    # 保存关键点值
    u_key = u[key_mask].clone()
    v_key = v[key_mask].clone()
    p_key = p[key_mask].clone()

    # 应用标准边界条件
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
    p[key_mask] = p_key

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

    for iteration, batch in enumerate(training_data_loader):
        # [注意] 此处必须解包 14 个变量，匹配 VaryGeoDataset_PairedSolutionOld 的输出
        [JJInv, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta,
         sol_u, sol_v, sol_p, voronoi_input] = to4DTensor(batch)

        optimizer.zero_grad()
        # [注意] 这里传入了 voronoi_input
        output = model(coord, voronoi_input)
        output_pad = udfpad(output)

        u = output_pad[:, 0:1, :, :]
        v = output_pad[:, 1:2, :, :]
        p = output_pad[:, 2:3, :, :]

        # 使用 mask 通道进行边界处理
        key_mask = voronoi_input[:, 3:4] > 0.5
        u, v, p = apply_boundary_conditions(u, v, p, key_mask)

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

            # 使用全局 U_ref (和 sol_u 其实是一样的，这里为了保持一致性)
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
            Mag_ref_internal = Mag_ref[0, 0, 1:-1, 1:-1]

            u_ref_internal = U_ref[0, 0, 1:-1, 1:-1]
            v_ref_internal = V_ref[0, 0, 1:-1, 1:-1]
            p_ref_internal = P_ref[0, 0, 1:-1, 1:-1]

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
        # [优化策略] 训练时只保存 .pth (快速)
        # =================================================================
        if err_mag < best_mag_error:
            best_mag_error = err_mag
            best_epoch = ep

            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pth"))
            print(f"[*] New Best Found at Epoch {ep}: Mag Err = {best_mag_error:.5f}")

        # 定期保存训练数据 (仅数值)
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
    # [后处理] 重新加载最佳模型生成最终文件
    # =================================================================
    print("\n正在生成最佳结果报告...")
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_model.pth")))
    model.eval()

    with torch.no_grad():
        batch = next(iter(training_data_loader))
        # [注意] 同样解包 14 个变量
        [JJInv, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta,
         sol_u, sol_v, sol_p, voronoi_input] = to4DTensor(batch)

        output = model(coord, voronoi_input)
        output_pad = udfpad(output)

        u = output_pad[:, 0:1, :, :];
        v = output_pad[:, 1:2, :, :];
        p = output_pad[:, 2:3, :, :]
        key_mask = voronoi_input[:, 3:4] > 0.5
        u, v, p = apply_boundary_conditions(u, v, p, key_mask)

        # 转 Numpy
        X_np = coord[0, 0, :, :].cpu().numpy()
        Y_np = coord[0, 1, :, :].cpu().numpy()

        U_pred = u[0, 0, :, :].cpu().numpy()
        V_pred = v[0, 0, :, :].cpu().numpy()
        P_pred = p[0, 0, :, :].cpu().numpy()
        Mag_pred = np.sqrt(U_pred ** 2 + V_pred ** 2)

        # 使用 sol_u 作为参考 (DataLoader 里读出来的)
        U_true = sol_u[0, 0, :, :].cpu().numpy()
        V_true = sol_v[0, 0, :, :].cpu().numpy()
        P_true = sol_p[0, 0, :, :].cpu().numpy()
        Mag_true = np.sqrt(U_true ** 2 + V_true ** 2)

        Err_mag = np.abs(Mag_pred - Mag_true)
        Err_p = np.abs(P_pred - P_true)

    # 1. 保存 CSV
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

    # 2. 保存图片
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

    # 5. 绘制收敛曲线
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

    print(f"所有结果已保存至: {OUT_DIR}")