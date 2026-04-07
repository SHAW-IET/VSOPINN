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
# from dataset import VaryGeoDataset_PairedSolution
from pyMesh import hcubeMesh, visualize2D, plotBC, plotMesh, setAxisLabel, \
    np2cuda
from model import USCNN
# 注意：这里引用的是 VoronoiMultiUSCNN
from voronoi_utils import VoronoiMultiUSCNN, to4DTensor, VaryGeoDataset_PairedSolution, LearnableKeyPoints, \
    generate_voronoi_input_torch, softVoronoi, compute_cvt_update
from readOF import convertOFMeshToImage, convertOFMeshToImage_StructuredMesh
from sklearn.metrics import mean_squared_error as calMSE
import Ofpp

# =========================================================================
# [新增] 全局控制开关与参数设置
# =========================================================================
# 模式选择:
# 'OPTIMIZE' : 训练并优化传感器位置 (随机初始化，记录最佳位置) -> 用于Re 100/300/400/800
# 'VALIDATE' : 固定传感器位置，仅重新训练网络 (需下方填入固定坐标) -> 用于Re 600/1000
RUN_MODE = 'OPTIMIZE'
INIT_STRATEGY = 'MANUAL'
INIT_SEED = 1234
# [验证模式专用] 聚类后得到的4个最优传感器坐标 (归一化 0-1)
# 仅当 RUN_MODE = 'VALIDATE' 时生效。这里填入你聚类算出的4个中心点
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

# =========================================================================

h = 0.01

# =========================================================================
# [修改] 多工况配置
# 根据你的需求，训练集应为 100, 300, 400, 800
# 请确保 data 目录下有对应的 csv 文件，没有的文件请自行生成或注释掉
# =========================================================================
case_config = [
    {'file': 'data/lid_100.csv', 'Re': 100, 'nu': 0.01, 'name': 'Re100'},
    {'file': 'data/lid_300.csv', 'Re': 300, 'nu': 1.0 / 300.0, 'name': 'Re300'},  # 需确保文件存在
    {'file': 'data/lid_400.csv', 'Re': 400, 'nu': 0.0025, 'name': 'Re400'},
    {'file': 'data/lid_800.csv', 'Re': 800, 'nu': 0.00125, 'name': 'Re800'},

    # 验证时可以解开这些注释，并注释掉上面的训练集
    # {'file': 'data/lid_600.csv', 'Re': 600, 'nu': 1.0/600.0, 'name': 'Re600'},
    # {'file': 'data/lid_1000.csv', 'Re': 1000, 'nu': 0.001, 'name': 'Re1000'},
]

num_conditions = len(case_config)
print(f"检测到 {num_conditions} 个工况配置。当前模式: {RUN_MODE}")
# =========================================================================


# =========================================================================
# [修改] 传感器初始化逻辑
# =========================================================================
if RUN_MODE == 'OPTIMIZE':
    print(f">>> 优化模式: 正在初始化传感器位置 (策略: {INIT_STRATEGY})...")

    if INIT_STRATEGY == 'RANDOM':
        # 完全随机
        initial_positions = torch.rand(4, 2) * 0.9 + 0.05

    elif INIT_STRATEGY == 'SEED':
        # 种子随机
        print(f"    使用随机种子: {INIT_SEED}")
        # 这里使用局部生成器，不影响全局 torch.rand
        g = torch.Generator()
        g.manual_seed(INIT_SEED)
        initial_positions = torch.rand(4, 2, generator=g) * 0.9 + 0.05

    elif INIT_STRATEGY == 'MANUAL':
        # 手动指定
        initial_positions = torch.tensor(MANUAL_INIT_COORDS, dtype=torch.float32)

    print(f"初始位置:\n{initial_positions}")


elif RUN_MODE == 'VALIDATE':
    print(">>> 验证模式: 使用固定的聚类中心点...")
    initial_positions = torch.tensor(FIXED_SENSOR_CENTROIDS, dtype=torch.float32)
    print(f"固定初始位置:\n{initial_positions}")

else:
    raise ValueError("RUN_MODE 必须是 'OPTIMIZE' 或 'VALIDATE'")

key_history = []

# 网格生成 (所有工况共享网格)
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
NvarInput = 2  # 输入 x y 坐标
NvarOutput = 3  # 输出从 T变为 U V P
nEpochs = 30000  # 可以根据需要调整，例如 20000 或 50000
# lr=0.001
lr = 0.001
Ns = 1
CLIP_NORM = 1.0  # 梯度裁剪阈值

# =========================================================================
# [加载] 多工况数据加载
# =========================================================================
loaders_dict = {}  # 存放不同工况的 DataLoader
refs_dict = {}  # 存放不同工况的参考值 (U_ref, V_ref...) 用于计算 Loss

# key_indices 仅用于 Dataset 初始化时的占位，实际并不影响 LearnableKeyPoints
# 为了兼容旧代码接口，保留一个空的或随意的列表
dummy_key_indices = [(50, 50)] * 4

for idx, case in enumerate(case_config):
    print(f"正在加载工况 {idx + 1}/{num_conditions}: {case['name']} (文件: {case['file']})...")

    # 读取数据
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

    # 翻转逻辑 (保持你原代码一致)
    OFY = np.flip(OFY_flat.reshape((ny, nx), order='C'), axis=0)
    OFU = np.flip(OFU, axis=0)
    OFV = np.flip(OFV, axis=0)
    OFMag = np.flip(OFMag, axis=0)
    OFP = np.flip(OFP, axis=0)

    # 制作 Reference Tensor (用于计算 Loss)
    U_ref = torch.tensor(OFU.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)
    V_ref = torch.tensor(OFV.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)
    Mag_ref = torch.tensor(OFMag.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)
    P_ref = torch.tensor(OFP.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)

    # 存入字典，包含当前工况的物理参数 nu
    refs_dict[idx] = {
        'U': U_ref, 'V': V_ref, 'P': P_ref, 'Mag': Mag_ref,
        'nu': case['nu'], 'name': case['name']
    }

    # 制作 Dataset
    SolutionList = [np.stack([OFU, OFV, OFP], axis=-1)]
    dataset_temp = VaryGeoDataset_PairedSolution(MeshList, SolutionList, dummy_key_indices)

    # 创建 DataLoader
    loaders_dict[idx] = DataLoader(dataset=dataset_temp, batch_size=batchSize, shuffle=False)

print("所有工况数据加载完成。")

# =========================================================================
# 模型初始化
# =========================================================================
# 注意使用 VoronoiMultiUSCNN
model = VoronoiMultiUSCNN(h, nx, ny, NvarInput, NvarOutput, num_conditions=num_conditions).to('cuda')
key_points_model = LearnableKeyPoints(initial_positions).to('cuda')

# =========================================================================
# [修改] 优化器配置 (根据 RUN_MODE 区分)
# =========================================================================
if RUN_MODE == 'OPTIMIZE':
    # 优化模式：同时更新 网络参数 和 传感器位置
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 1e-3},
        {'params': key_points_model.parameters(), 'lr': 5e-2, 'weight_decay': 1e-7}
    ])
else:
    # 验证模式：只更新 网络参数，传感器位置不放入优化器（相当于冻结）
    # 显式将 key_points_model 的梯度关闭，双重保险
    for param in key_points_model.parameters():
        param.requires_grad = False

    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 1e-3}
    ])

criterion = nn.MSELoss()  # 选取损失形式
padSingleSide = 1
udfpad = nn.ConstantPad2d([padSingleSide, padSingleSide, padSingleSide, padSingleSide], 0)


#
def apply_boundary_conditions(u, v, p, voronoi_input):
    # 从 Voronoi 输入中提取真实值 [batch, 4, ny, nx]
    u_vor = voronoi_input[:, 0:1, :, :]  # U 分量真实值
    v_vor = voronoi_input[:, 1:2, :, :]  # V 分量真实值
    p_vor = voronoi_input[:, 2:3, :, :]  # P 分量真实值
    key_mask = voronoi_input[:, 3:4] > 0.5  # 关键点掩码

    # 应用标准边界条件
    # 顶边 (y=1) - 数组最后一行
    u[:, :, -1, 1:-1] = 1.0
    v[:, :, -1, 1:-1] = 0.0

    # 底边 (y=0) - 数组第一行
    u[:, :, 0, 1:-1] = 0.0
    v[:, :, 0, 1:-1] = 0.0

    # 左边 (x=0)
    u[:, :, 1:-1, 0] = 0.0
    v[:, :, 1:-1, 0] = 0.0

    # 右边 (x=1)
    u[:, :, 1:-1, -1] = 0.0
    v[:, :, 1:-1, -1] = 0.0

    # 角点处理
    # 左下角 (0,0)
    u[:, :, 0, 0] = 0.5 * (u[:, :, 0, 1] + u[:, :, 1, 0])
    v[:, :, 0, 0] = 0.5 * (v[:, :, 0, 1] + v[:, :, 1, 0])

    # 右下角 (0,1)
    u[:, :, 0, -1] = 0.5 * (u[:, :, 0, -2] + u[:, :, 1, -1])
    v[:, :, 0, -1] = 0.5 * (v[:, :, 0, -2] + v[:, :, 1, -1])

    # 左上角 (1,0)
    u[:, :, -1, 0] = 0.5 * (u[:, :, -1, 1] + u[:, :, -2, 0])
    v[:, :, -1, 0] = 0.5 * (v[:, :, -1, 1] + v[:, :, -2, 0])

    # 右上角 (1,1)
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
    # 记录所有工况的累积损失和误差
    err_dict = {'u': 0, 'v': 0, 'p': 0, 'mag': 0, 'av': 0}

    startTime = time.time()

    # --- 1. 获取共享的传感器位置 ---
    # 1) 连续 [0,1] 坐标（供 Voronoi 用）
    key_positions = key_points_model.get_normalized()  # [N,2] (y,x)
    # 2) 离散索引（带 STE）
    idx_float = key_points_model((ny, nx))  # [N,2] float

    optimizer.zero_grad()

    # 用于反向传播的总 Loss
    combined_loss = 0

    # =====================================================================
    # [循环] 遍历每个工况 (Multi-Condition Training)
    # =====================================================================
    for case_idx in range(num_conditions):
        loader = loaders_dict[case_idx]
        ref = refs_dict[case_idx]
        current_nu = ref['nu']  # 获取当前工况的粘度

        # 获取该工况的一个 batch (通常 batch=1)
        batch = next(iter(loader))
        [JJInv, coord_tensor, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta,
         sol_u, sol_v, sol_p] = to4DTensor(batch)

        # 准备 Voronoi 输入
        coord_2d = coord_tensor[0]
        sol_u_2d = sol_u[0, 0]
        sol_v_2d = sol_v[0, 0]
        sol_p_2d = sol_p[0, 0]

        # 使用共享的关键点位置
        voronoi_input = softVoronoi(
            coord_2d, sol_u_2d, sol_v_2d, sol_p_2d,
            key_positions,
            (ny, nx), 80)

        # === 模型前向传播 (传入 condition_idx) ===
        # 这里的 condition_idx 告诉模型使用哪一个 Decoder
        output = model(coord_tensor, voronoi_input, condition_idx=case_idx)
        output_pad = udfpad(output)

        # ── 分离通道 & 边界处理 ──
        u = output_pad[:, 0:1, :, :]
        v = output_pad[:, 1:2, :, :]
        p = output_pad[:, 2:3, :, :]

        u, v, p = apply_boundary_conditions(u, v, p, voronoi_input)

        # ── PDE Loss 计算 (使用 current_nu) ──
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

        Rc = ux + vy  # 连续性
        Ru = u * ux + v * uy + px - current_nu * (uxx + uyy)  # x方向动量方程 (使用当前 nu)
        Rv = u * vx + v * vy + py - current_nu * (vxx + vyy)  # y方向动量方程 (使用当前 nu)

        sl = slice(1, -1)
        loss_pde = (
                criterion(Rc[:, :, sl, sl], torch.zeros_like(Rc[:, :, sl, sl])) +
                criterion(Ru[:, :, sl, sl], torch.zeros_like(Ru[:, :, sl, sl])) +
                criterion(Rv[:, :, sl, sl], torch.zeros_like(Rv[:, :, sl, sl]))
        )

        # ── Data Loss 计算 ──
        # 构造归一化采样网格
        grid_pts = torch.stack([
            2 * idx_float[:, 1] - 1,  # x ∈ [-1,1]
            2 * idx_float[:, 0] - 1  # y ∈ [-1,1]
        ], dim=-1).view(1, -1, 1, 2)  # (1,N,1,2)

        # 双线性采样预测值
        pred_uvp = F.grid_sample(output_pad, grid_pts, mode='bilinear', align_corners=True).view(3, -1)
        u_pred, v_pred, p_pred = pred_uvp

        # 采样当前工况的参考值
        ref_uvp = F.grid_sample(torch.cat([ref['U'], ref['V'], ref['P']], dim=1),
                                grid_pts, align_corners=True).view(3, -1)
        u_ref, v_ref, p_ref = ref_uvp

        loss_data = (criterion(u_pred, u_ref) +
                     criterion(v_pred, v_ref) +
                     criterion(p_pred, p_ref))

        # 当前工况的总 Loss
        loss_case = loss_pde + loss_data

        # 累加到全局 Loss
        combined_loss += loss_case

        # ── 记录误差 (用于显示) ──
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

    # =====================================================================

    # 反向传播 (对所有工况的累加 Loss 求导)
    combined_loss.backward()

    # 梯度处理
    for name, param in list(model.named_parameters()) + list(key_points_model.named_parameters()):
        if param.grad is not None and not torch.all(torch.isfinite(param.grad)):
            # print(f"[WARN] Detected non-finite grad in {name}; zeroing it.")
            param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)

    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
    if RUN_MODE == 'OPTIMIZE':
        torch.nn.utils.clip_grad_norm_(key_points_model.parameters(), CLIP_NORM)

    # VALIDATE 模式下 key_points_model 没有梯度，step 也无妨（因为 grad 为 None 或 0）
    optimizer.step()

    # =========================================================
    #  [NEW] CVT Sensor Optimization Module (Fixed for Case 6.2 Structure)
    # =========================================================
    CVT_INTERVAL = 500  # 每 500 个 epoch 执行一次
    device = 'cuda'  # 明确指定设备

    # 【修正说明】
    # 原代码结构中，train函数每运行一次就是一个Epoch，且只执行一次optimizer.step()。
    # 因此不需要判断 batch 索引或 case 索引，直接判断 epoch 间隔即可。

    if epoch > 0 and epoch % CVT_INTERVAL == 0:
        print(f"\n>>> [CVT] Epoch {epoch}: Performing Sensor Optimization...")

        with torch.no_grad():
            # 1. 构造全场真值 (Solution) 和 预测值 (Prediction)
            # 注意：这里的 sol_u, output_pad 等变量来自 for case_idx 循环的最后一次迭代。
            # 这意味着我们是基于最后一个工况 (通常是 Re800) 的误差分布来进行优化的。
            # 对于多工况任务，针对最复杂的流场进行传感器优化是最佳策略之一。

            # 拼接真值 (Batch=1, 3, H, W)
            curr_ref = torch.cat([sol_u, sol_v, sol_p], dim=1).to(device)
            curr_pred = output_pad.detach()

            # 2. 计算密度函数 (Density Map)
            diff = curr_pred - curr_ref
            error_sq = torch.sum(diff ** 2, dim=1)

            # 计算误差图 (H, W)
            density_map = torch.sqrt(error_sq).mean(dim=0)

            # 3. 获取当前传感器位置 (0~1 范围)
            current_pos_01 = key_points_model.get_normalized()

            # 4. 坐标系转换 [0, 1] -> [-1, 1]
            # current_pos_01 是 (y, x) -> 翻转为 (x, y)
            current_pos_xy_01 = torch.flip(current_pos_01, dims=[1])
            current_pos_xy_11 = 2.0 * current_pos_xy_01 - 1.0

            # 5. 执行 CVT 计算
            new_pos_xy_11 = compute_cvt_update(current_pos_xy_11, density_map, device=device)

            # 6. 还原坐标并更新
            # [-1, 1] -> [0, 1]
            new_pos_xy_01 = (new_pos_xy_11 + 1.0) / 2.0
            # (x, y) -> (y, x)
            new_pos_yx_01 = torch.flip(new_pos_xy_01, dims=[1])

            # 7. 更新 LearnableKeyPoints
            # 防止数值不稳定，clamp 到 (0.01, 0.99)
            new_pos_yx_01 = torch.clamp(new_pos_yx_01, 0.01, 0.99)
            new_raw = torch.logit(new_pos_yx_01)

            key_points_model.raw.data.copy_(new_raw)

            # 打印统计
            shift = torch.norm(new_pos_yx_01 - current_pos_01, dim=1).mean().item()
            print(f"    -> Density Peak: {density_map.max().item():.5f}")
            print(f"    -> Sensors Updated. Mean Shift: {shift:.5f}")

    # =============================================
    if RUN_MODE == 'OPTIMIZE':
        with torch.no_grad():
            key_points_model.raw.clamp_(-4.6, 4.6)  # ≈ sigmoid → [0.01,0.99]

    # === 打印信息 & 记录关键点 ===
    # 计算平均误差
    avg_u = err_dict['u'] / num_conditions
    avg_v = err_dict['v'] / num_conditions
    avg_p = err_dict['p'] / num_conditions
    avg_mag = err_dict['mag'] / num_conditions
    err_av = (avg_u + avg_v + avg_p + avg_mag) / 4.0

    if epoch % 100 == 0:
        print(f"[Epoch {epoch:4d}] Loss = {combined_loss.item():.2e} | Avg Mag Err = {avg_mag:.5f}")
        # 仅在 OPTIMIZE 模式下记录历史，VALIDATE 模式不需要记录轨迹
        if RUN_MODE == 'OPTIMIZE':
            key_history.append(key_points_model.get_normalized().detach().cpu().numpy())

    # === 绘图 (仅绘制第一个工况以保持稳定) ===
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

        # 绘图数据准备
        X = coord_tensor[0, 0, :, :].cpu().numpy()
        Y = coord_tensor[0, 1, :, :].cpu().numpy()
        U_p = u_vis[0, 0, :, :].cpu().detach().numpy()
        V_p = v_vis[0, 0, :, :].cpu().detach().numpy()
        Mag_p = np.sqrt(U_p ** 2 + V_p ** 2)
        U_r = vis_ref['U'][0, 0, :, :].cpu().numpy()
        V_r = vis_ref['V'][0, 0, :, :].cpu().numpy()
        Mag_r = vis_ref['Mag'][0, 0, :, :].cpu().numpy()

        # ... (简化绘图代码，保持核心逻辑) ...
        # 这里略去详细的 matplotlib 代码以节省篇幅，保持你原有的绘图逻辑即可
        # 建议保留你原来完整的绘图代码块

        # 简单示意保存
        out_dir = "./output/完善/9 有数据点 - 优化数据点 - 多工况 - CVT/"
        os.makedirs(out_dir, exist_ok=True)
    # plt.savefig...

    return combined_loss.item(), avg_u, avg_v, avg_p, avg_mag, err_av


if __name__ == '__main__':
    # 初始化列表记录训练过程
    losses = []
    errors_u = []
    errors_v = []
    errors_p = []
    errors_mag = []
    errors_av = []

    # [新增] 最佳误差记录器
    best_mag_error = float('inf')
    best_epoch = -1

    out_dir = "./output/完善/9 有数据点 - 优化数据点 - 多工况 - CVT/"
    os.makedirs(out_dir, exist_ok=True)

    # [新增] 根据 RUN_MODE 修改输出文件夹，避免覆盖
    if RUN_MODE == 'OPTIMIZE':
        # 可以在文件名加时间戳防止覆盖： Run_2024...
        save_prefix = "Opt"
    else:
        save_prefix = "Val"

    t0 = time.time()
    for ep in range(1, nEpochs + 1):
        loss, err_u, err_v, err_p, err_mag, err_av = train(ep)

        # 记录损失和误差
        losses.append(loss)
        errors_u.append(err_u)
        errors_v.append(err_v)
        errors_p.append(err_p)
        errors_mag.append(err_mag)
        errors_av.append(err_av)

        # =================================================================
        # [新增] 核心功能：保存最佳模型和传感器位置
        # =================================================================
        if err_mag < best_mag_error:
            best_mag_error = err_mag
            best_epoch = ep

            # 1. 保存最佳模型权重
            torch.save(model.state_dict(), os.path.join(out_dir, f"{save_prefix}_best_model.pth"))

            # 2. 保存此时的最佳传感器位置 (无论是 OPTIMIZE 还是 VALIDATE 都可以存一下)
            # 获取当前归一化坐标 (y, x)
            current_best_pos = key_points_model.get_normalized().detach().cpu().numpy()
            # 保存为 CSV (覆盖式写入，始终保持唯一的最佳)
            np.savetxt(os.path.join(out_dir, f"{save_prefix}_best_sensors.csv"),
                       current_best_pos.reshape(1, -1),  # 展平为一行
                       delimiter=',',
                       header="y1,x1,y2,x2,y3,x3,y4,x4",
                       comments='')

            if ep % 100 == 0:
                print(f"[*] New Best Found at Epoch {ep}: Mag Err = {best_mag_error:.5f}. Saved.")

        # 定期保存过程数据
        if ep % 100 == 0 or ep == nEpochs:
            # 保存训练曲线数据
            np.savetxt(os.path.join(out_dir, f'{save_prefix}_training_log.csv'),
                       np.column_stack([losses, errors_u, errors_v, errors_p, errors_mag, errors_av]),
                       delimiter=',',
                       header='loss,err_u,err_v,err_p, errors_mag, errors_av')

    dt = time.time() - t0
    print(f"总训练时间: {dt:.2f} 秒")
    print(f"最佳 Mag Error: {best_mag_error:.5f} (Epoch {best_epoch})")

    # 如果是优化模式，保存完整的轨迹历史
    if RUN_MODE == 'OPTIMIZE':
        key_hist_arr = np.stack(key_history, axis=0)
        flat_hist = key_hist_arr.reshape(len(key_history), -1)
        np.savetxt(os.path.join(out_dir, "Opt_keypoints_history.csv"),
                   flat_hist, delimiter=",",
                   header=",".join([f"y{i + 1},x{i + 1}" for i in range(4)]),
                   comments='')
        print(f"优化轨迹已保存。")

    print("训练结束。")