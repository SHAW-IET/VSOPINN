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
from pyMesh import hcubeMesh, visualize2D, plotBC, plotMesh,setAxisLabel,\
                   np2cuda
from model import USCNN
from voronoi_utils import VoronoiEnhancedUSCNN,to4DTensor,VaryGeoDataset_PairedSolutionOld



from readOF import convertOFMeshToImage,convertOFMeshToImage_StructuredMesh
from sklearn.metrics import mean_squared_error as calMSE
import Ofpp

h=0.01


key_indices = [
	(13, 93),
	(91, 93),
	(52, 32),
	(87, 85)
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






myMesh=hcubeMesh(leftX,leftY,rightX,rightY,
	             lowX,lowY,upX,upY,h,True,True,
	             tolMesh=1e-10,tolJoint=1)
batchSize=1
NvarInput=2   # 输入 x y 坐标
NvarOutput=3  # 输出从 T变为 U V P
nEpochs=20000
# lr=0.001
lr = 0.001
Ns=1
nu=0.01

# model=USCNN(h,nx,ny,NvarInput,NvarOutput).to('cuda')   # h 网格间距
model = VoronoiEnhancedUSCNN(h, nx, ny, NvarInput, NvarOutput).to('cuda')


criterion = nn.MSELoss()     # 选取损失形式
optimizer = optim.Adam(model.parameters(),lr=lr)
padSingleSide=1						# ConstantPad2d([left, right, top, bottom], value) 会把输入张量的空间维度 H×W 扩展成 (H+2)×(W+2)，四周填充值为常数 0。
udfpad=nn.ConstantPad2d([padSingleSide,padSingleSide,padSingleSide,padSingleSide],0)   #  在网络输出上加上一圈“零值”边框（也就是常数填充），以便后面用有限差分卷积核在边界处也能一致地计算导数
MeshList=[]
MeshList.append(myMesh)

# train_set=VaryGeoDataset(MeshList)        # 重新组织数据  未改值



OFX_flat = data[:, 0]  # 第一列：X 坐标  (101, )
OFY_flat = data[:, 1]  # 第二列：Y 坐标
OFU_flat = data[:, 2]  # 第三列：u 分量
OFV_flat = data[:, 3]  # 第四列：v 分量
OFMag_flat = data[:, 4]  # 第五列：Velocity Magnitude
OFP_flat = data[:, 5]  # 第六列：Pressure（P）

OFX = OFX_flat.reshape((ny, nx), order='C') # (101, 101)
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

training_data_loader=DataLoader(dataset=train_set,
	                            batch_size=batchSize)
# 可视化检查

x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0


fig, axes = plt.subplots(1, 4, figsize=(20, 5))

im0 = axes[0].imshow(OFU, origin='lower', extent=[x_min, x_max, y_min, y_max])
axes[0].set_title('OFU Field')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(OFV, origin='lower', extent=[x_min, x_max, y_min, y_max])
axes[1].set_title('OFV Field')
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(OFP, origin='lower', extent=[x_min, x_max, y_min, y_max])
axes[2].set_title('OFP Field')
plt.colorbar(im2, ax=axes[2])

im3 = axes[3].imshow(OFMag, origin='lower', extent=[x_min, x_max, y_min, y_max])
axes[3].set_title('Velocity Magnitude Field')
plt.colorbar(im3, ax=axes[3])

plt.tight_layout()
plt.show()

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

	# 硬编码
	u = torch.where(key_mask, u_vor, u)
	v = torch.where(key_mask, v_vor, v)
	p = torch.where(key_mask, p_vor, p)


	return u, v, p

def dfdx(f,dydeta,dydxi,Jinv):
	dfdxi_internal=(-f[:,:,:,4:]+8*f[:,:,:,3:-1]-8*f[:,:,:,1:-3]+f[:,:,:,0:-4])/12/h
	dfdxi_left=(-11*f[:,:,:,0:-3]+18*f[:,:,:,1:-2]-9*f[:,:,:,2:-1]+2*f[:,:,:,3:])/6/h
	dfdxi_right=(11*f[:,:,:,3:]-18*f[:,:,:,2:-1]+9*f[:,:,:,1:-2]-2*f[:,:,:,0:-3])/6/h
	dfdxi=torch.cat((dfdxi_left[:,:,:,0:2],dfdxi_internal,dfdxi_right[:,:,:,-2:]),3)
	dfdeta_internal=(-f[:,:,4:,:]+8*f[:,:,3:-1,:]-8*f[:,:,1:-3,:]+f[:,:,0:-4,:])/12/h
	dfdeta_low=(-11*f[:,:,0:-3,:]+18*f[:,:,1:-2,:]-9*f[:,:,2:-1,:]+2*f[:,:,3:,:])/6/h
	dfdeta_up=(11*f[:,:,3:,:]-18*f[:,:,2:-1,:]+9*f[:,:,1:-2,:]-2*f[:,:,0:-3,:])/6/h
	dfdeta=torch.cat((dfdeta_low[:,:,0:2,:],dfdeta_internal,dfdeta_up[:,:,-2:,:]),2)
	dfdx=Jinv*(dfdxi*dydeta-dfdeta*dydxi)
	return dfdx
def dfdy(f,dxdxi,dxdeta,Jinv):
	dfdxi_internal=(-f[:,:,:,4:]+8*f[:,:,:,3:-1]-8*f[:,:,:,1:-3]+f[:,:,:,0:-4])/12/h
	dfdxi_left=(-11*f[:,:,:,0:-3]+18*f[:,:,:,1:-2]-9*f[:,:,:,2:-1]+2*f[:,:,:,3:])/6/h
	dfdxi_right=(11*f[:,:,:,3:]-18*f[:,:,:,2:-1]+9*f[:,:,:,1:-2]-2*f[:,:,:,0:-3])/6/h
	dfdxi=torch.cat((dfdxi_left[:,:,:,0:2],dfdxi_internal,dfdxi_right[:,:,:,-2:]),3)

	dfdeta_internal=(-f[:,:,4:,:]+8*f[:,:,3:-1,:]-8*f[:,:,1:-3,:]+f[:,:,0:-4,:])/12/h
	dfdeta_low=(-11*f[:,:,0:-3,:]+18*f[:,:,1:-2,:]-9*f[:,:,2:-1,:]+2*f[:,:,3:,:])/6/h
	dfdeta_up=(11*f[:,:,3:,:]-18*f[:,:,2:-1,:]+9*f[:,:,1:-2,:]-2*f[:,:,0:-3,:])/6/h
	dfdeta=torch.cat((dfdeta_low[:,:,0:2,:],dfdeta_internal,dfdeta_up[:,:,-2:,:]),2)
	dfdy=Jinv*(dfdeta*dxdxi-dfdxi*dxdeta)
	return dfdy
# def train(epoch):
# 	global U_ref, V_ref, P_ref
# 	loss_data = 0
# 	startTime=time.time()
# 	xRes=0
# 	yRes=0
# 	mRes=0
# 	eU=0
# 	eV=0
# 	eP=0
# 	for iteration, batch in enumerate(training_data_loader):
# 		[JJInv,coord,xi,eta,J,Jinv,dxdxi,dydxi,dxdeta,dydeta]=to4DTensor(batch)
# 		# J(1,1,19,84),JJInv(1,2,19,84),Jinv(1,1,19,84)，coord(1,2,19,84),dxdeta（1,1,19,84），dxdxi(1,1,19,84)，dydeta(1,1,19,84)，dydxi(1,1,19,84)，output（1，1，17，82），output_pad（1，1，19，84）
# 		# 二阶中心差分（dxdxi, dydxi, dxdeta, dydeta）只在不靠边的内部点定义   因为要去掉最外面一圈，所以形状从 (19,84) 减少两行两列，变成 (17, 82)。
# 		# 由此得到的雅可比 J, Jinv 也是 (17, 82)。
# 		# DataLoader (batch_size=1) 会把它们堆成形状 (1, 19, 84) —— 在最前面加一个“批次”维度。
# 		# to4DTensor 检测到这是一个三维张量，就再加一个“通道”维度，变成 (1, 1, 19, 84)。
# 		# 如果原来就是二维通道（比如 coord=[x,y] 或 InvariantInput=[J,Jinv]，NumPy 先给你 (2,19,84)），
# 		# DataLoader 会先堆成 (1,2,19,84)，to4DTensor 发现已经有 4 维就不变，仍然是 (1,2,19,84)。
# 		optimizer.zero_grad()
# 		output=model(coord)           # (1,1,17,82)
# 		output_pad=udfpad(output)    # 输出后套一层  用法？？？？？？？？？求偏导？？？？？  (1,1,19,84)
# 		# outputV=output_pad[:,0,:,:].reshape(output_pad.shape[0],1,           # (1,1,19,84) 取出你想要的某个物理量（这里是网络输出 output_pad 的“第 0 通道”），
# 		# 	                                output_pad.shape[2],
# 		# 	                                output_pad.shape[3])
# 		# ── 分离通道 & 硬编码边界 ──
# 		u = output_pad[:, 0:1, :, :]
# 		v = output_pad[:, 1:2, :, :]
# 		p = output_pad[:, 2:3, :, :]
#
# 		u, v = apply_boundary_conditions(u, v)
#
#
# 		ux = dfdx(u, dydeta, dydxi, Jinv);
# 		uy = dfdy(u, dxdxi, dxdeta, Jinv)
# 		vx = dfdx(v, dydeta, dydxi, Jinv);
# 		vy = dfdy(v, dxdxi, dxdeta, Jinv)
# 		px = dfdx(p, dydeta, dydxi, Jinv);
# 		py = dfdy(p, dxdxi, dxdeta, Jinv)
# 		uxx = dfdx(ux, dydeta, dydxi, Jinv);
# 		uyy = dfdy(uy, dxdxi, dxdeta, Jinv)
# 		vxx = dfdx(vx, dydeta, dydxi, Jinv);
# 		vyy = dfdy(vy, dxdxi, dxdeta, Jinv)
#
# 		Rc = ux + vy # 连续性
# 		Ru = u * ux + v * uy + px - nu * (uxx + uyy) # x方向动量方程
# 		Rv = u * vx + v * vy + py - nu * (vxx + vyy) # y方向动量方程
#
#
#
# 		sl = slice(1, -1)
#
# 		for iy, ix in key_indices:
# 			# 预测值（注意边界填充）
# 			u_pred = u[0, 0, iy + 1, ix + 1]  # +1因为 padding
# 			v_pred = v[0, 0, iy + 1, ix + 1]
# 			p_pred = p[0, 0, iy + 1, ix + 1]
#
# 			# 参考值
# 			u_ref_point = U_ref[0, 0, iy, ix]
# 			v_ref_point = V_ref[0, 0, iy, ix]
# 			p_ref_point = P_ref[0, 0, iy, ix]
#
# 			loss_data += criterion(u_pred, u_ref_point)
# 			loss_data += criterion(v_pred, v_ref_point)
# 			loss_data += criterion(p_pred, p_ref_point)
#
#
#
#
#
# 		loss_pde = (
# 				0.3 * criterion(Rc[:, :, sl, sl], torch.zeros_like(Rc[:, :, sl, sl])) +
# 				0.2 * criterion(Ru[:, :, sl, sl], torch.zeros_like(Ru[:, :, sl, sl])) +
# 				0.4 * criterion(Rv[:, :, sl, sl], torch.zeros_like(Rv[:, :, sl, sl]))
# 		)
# 		# data_weight = max(0.5, 2.0 * (1 - epoch / nEpochs))   # 动态权重 初期 data权重大 后期小
# 		loss = loss_pde + loss_data
#
# 		# 计算全局L2误差（内部点）
# 		with torch.no_grad():
# 			# 获取内部点预测值
# 			u_internal = u[0, 0, 1:-1, 1:-1]
# 			v_internal = v[0, 0, 1:-1, 1:-1]
# 			p_internal = p[0, 0, 1:-1, 1:-1]
#
# 			# 参考值（去掉边界）
# 			u_ref_internal = U_ref[0, 0, 1:-1, 1:-1]
# 			v_ref_internal = V_ref[0, 0, 1:-1, 1:-1]
# 			p_ref_internal = P_ref[0, 0, 1:-1, 1:-1]
#
# 			# 计算L2相对误差
# 			l2_error_u = torch.sqrt(torch.mean((u_internal - u_ref_internal) ** 2)) / torch.sqrt(
# 				torch.mean(u_ref_internal ** 2))
# 			l2_error_v = torch.sqrt(torch.mean((v_internal - v_ref_internal) ** 2)) / torch.sqrt(
# 				torch.mean(v_ref_internal ** 2))
# 			l2_error_p = torch.sqrt(torch.mean((p_internal - p_ref_internal) ** 2)) / torch.sqrt(
# 				torch.mean(p_ref_internal ** 2))
#
# 		loss.backward()
# 		optimizer.step()
#
#
#
#
#
#
# 	if epoch % 100 == 0:
# 		print(f"[Epoch {epoch:4d}] Loss = {loss.item():.2e}")
# 		print(f"L2 Errors: U={l2_error_u.item():.4f}, V={l2_error_v.item():.4f}, P={l2_error_p.item():.4f}")
#
# 		# 使用整个网格而不是内部点切片
# 		X = coord[0, 0, :, :].cpu().numpy()  # 整个网格的X坐标
# 		Y = coord[0, 1, :, :].cpu().numpy()  # 整个网格的Y坐标
# 		U_p = u[0, 0, :, :].cpu().detach().numpy()  # 整个网格的U预测
# 		V_p = v[0, 0, :, :].cpu().detach().numpy()  # 整个网格的V预测
# 		P_p = p[0, 0, :, :].cpu().detach().numpy()  # 整个网格的P预测
# 		U_r = U_ref[0, 0, :, :].cpu().numpy()  # 整个网格的U参考
# 		V_r = V_ref[0, 0, :, :].cpu().numpy()  # 整个网格的V参考
# 		P_r = P_ref[0, 0, :, :].cpu().numpy()  # 整个网格的P参考
#
# 		# 计算全局范围
# 		u_min = min(U_p.min(), U_r.min())
# 		u_max = max(U_p.max(), U_r.max())
# 		v_min = min(V_p.min(), V_r.min())
# 		v_max = max(V_p.max(), V_r.max())
# 		p_min = min(P_p.min(), P_r.min())
# 		p_max = max(P_p.max(), P_r.max())
#
# 		# 为每个物理量设置统一范围
# 		ranges = {
# 			'U': (u_min, u_max),
# 			'V': (v_min, v_max),
# 			'P': (p_min, p_max)
# 		}
#
# 		# 使用相同的颜色映射
# 		cmap = 'viridis'
#
# 		fig, axes = plt.subplots(3, 3, figsize=(15, 12))  # 3行：预测、参考、误差
# 		for i, (pred, ref, name) in enumerate(zip(
# 				[U_p, V_p, P_p], [U_r, V_r, P_r], ['U', 'V', 'P']
# 		)):
# 			vmin, vmax = ranges[name]
#
# 			# 预测解 (第1行)
# 			im0 = axes[0, i].contourf(X, Y, pred, levels=20, vmin=vmin, vmax=vmax, cmap=cmap)
# 			axes[0, i].set_title(f'Predicted {name}')
# 			plt.colorbar(im0, ax=axes[0, i])
#
# 			# 参考解 (第2行)
# 			im1 = axes[1, i].contourf(X, Y, ref, levels=20, vmin=vmin, vmax=vmax, cmap=cmap)
# 			axes[1, i].set_title(f'Reference {name}')
# 			plt.colorbar(im1, ax=axes[1, i])
#
# 			# 绝对误差 (第3行)
# 			error = np.abs(pred - ref)
# 			im2 = axes[2, i].contourf(X, Y, error, levels=20, cmap='hot')
# 			axes[2, i].set_title(f'Abs Error {name}')
# 			plt.colorbar(im2, ax=axes[2, i])
#
# 			# ================== 添加关键点标记 ==================
# 			for point_idx, (iy, ix) in enumerate(key_indices):
# 				# 转换为绘图坐标 - 直接使用原始索引
# 				x_val = X[iy, ix]
# 				y_val = Y[iy, ix]
#
# 				# 在预测图上标记
# 				axes[0, i].plot(x_val, y_val, 'ro', markersize=6, markeredgewidth=1, markerfacecolor='none')
# 				axes[0, i].text(x_val, y_val, str(point_idx + 1), color='white', fontsize=8, ha='center', va='center')
#
# 				# 在参考图上标记
# 				axes[1, i].plot(x_val, y_val, 'ro', markersize=6, markeredgewidth=1, markerfacecolor='none')
# 				axes[1, i].text(x_val, y_val, str(point_idx + 1), color='white', fontsize=8, ha='center', va='center')
#
# 				# 在误差图上标记
# 				axes[2, i].plot(x_val, y_val, 'ro', markersize=6, markeredgewidth=1, markerfacecolor='none')
# 				axes[2, i].text(x_val, y_val, str(point_idx + 1), color='white', fontsize=8, ha='center', va='center')
#
# 		# ================== 添加关键点标记结束 ==================
#
# 		# 添加标题说明
# 		plt.suptitle(f'Epoch {epoch}: Key Points (red circles)', fontsize=16)
# 		plt.tight_layout()
# 		plt.savefig(f'epoch_{epoch:04d}.png', dpi=150)
# 		plt.close()  # 关闭图形，避免内存累积
# 	return loss.item(), l2_error_u.item(), l2_error_v.item(), l2_error_p.item()

def train(epoch):
	global U_ref, V_ref, P_ref
	loss_data = 0
	startTime=time.time()
	xRes=0
	yRes=0
	mRes=0
	eU=0
	eV=0
	eP=0
	for iteration, batch in enumerate(training_data_loader):
		[JJInv, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta,
		 sol_u, sol_v, sol_p, voronoi_input] = to4DTensor(batch)

		# J(1,1,19,84),JJInv(1,2,19,84),Jinv(1,1,19,84)，coord(1,2,19,84),dxdeta（1,1,19,84），dxdxi(1,1,19,84)，dydeta(1,1,19,84)，dydxi(1,1,19,84)，output（1，1，17，82），output_pad（1，1，19，84）
		# 二阶中心差分（dxdxi, dydxi, dxdeta, dydeta）只在不靠边的内部点定义   因为要去掉最外面一圈，所以形状从 (19,84) 减少两行两列，变成 (17, 82)。
		# 由此得到的雅可比 J, Jinv 也是 (17, 82)。
		# DataLoader (batch_size=1) 会把它们堆成形状 (1, 19, 84) —— 在最前面加一个“批次”维度。
		# to4DTensor 检测到这是一个三维张量，就再加一个“通道”维度，变成 (1, 1, 19, 84)。
		# 如果原来就是二维通道（比如 coord=[x,y] 或 InvariantInput=[J,Jinv]，NumPy 先给你 (2,19,84)），
		# DataLoader 会先堆成 (1,2,19,84)，to4DTensor 发现已经有 4 维就不变，仍然是 (1,2,19,84)。

		optimizer.zero_grad()
		output = model(coord, voronoi_input)           # (1,1,17,82)
		output_pad=udfpad(output)    # 输出后套一层  用法？？？？？？？？？求偏导？？？？？  (1,1,19,84)
		# outputV=output_pad[:,0,:,:].reshape(output_pad.shape[0],1,           # (1,1,19,84) 取出你想要的某个物理量（这里是网络输出 output_pad 的“第 0 通道”），
		# 	                                output_pad.shape[2],
		# 	                                output_pad.shape[3])
		# ── 分离通道 & 硬编码边界 ──
		u = output_pad[:, 0:1, :, :]
		v = output_pad[:, 1:2, :, :]
		p = output_pad[:, 2:3, :, :]

		# u, v = apply_boundary_conditions(u, v)
		key_mask = voronoi_input[:, 3:4] > 0.5
		u, v, p = apply_boundary_conditions(u, v, p, voronoi_input) # mask通道 > 0.5的被标记为 True

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

		Rc = ux + vy # 连续性
		Ru = u * ux + v * uy + px - nu * (uxx + uyy) # x方向动量方程
		Rv = u * vx + v * vy + py - nu * (vxx + vyy) # y方向动量方程



		sl = slice(1, -1)

		for iy, ix in key_indices: 								# 关键点被硬编码   loss_data = 0
			# 预测值（注意边界填充）
			u_pred = u[0, 0, iy, ix] # +1因为 padding, 不需要＋1？
			v_pred = v[0, 0, iy, ix]
			p_pred = p[0, 0, iy, ix]

			# 参考值
			u_ref_point = U_ref[0, 0, iy, ix]
			v_ref_point = V_ref[0, 0, iy, ix]
			p_ref_point = P_ref[0, 0, iy, ix]

			loss_data += criterion(u_pred, u_ref_point)
			loss_data += criterion(v_pred, v_ref_point)
			loss_data += criterion(p_pred, p_ref_point)





		loss_pde = (
				0.3 * criterion(Rc[:, :, sl, sl], torch.zeros_like(Rc[:, :, sl, sl])) +
				0.2 * criterion(Ru[:, :, sl, sl], torch.zeros_like(Ru[:, :, sl, sl])) +
				0.4 * criterion(Rv[:, :, sl, sl], torch.zeros_like(Rv[:, :, sl, sl]))
		)
		# data_weight = max(0.5, 2.0 * (1 - epoch / nEpochs))   # 动态权重 初期 data权重大 后期小
		loss = loss_pde + loss_data

		# 计算全局 L2误差（内部点）
		with torch.no_grad():
			# 获取内部点预测值
			u_internal = u[0, 0, 1:-1, 1:-1]
			v_internal = v[0, 0, 1:-1, 1:-1]
			p_internal = p[0, 0, 1:-1, 1:-1]

			# 参考值（去掉边界）
			u_ref_internal = U_ref[0, 0, 1:-1, 1:-1]
			v_ref_internal = V_ref[0, 0, 1:-1, 1:-1]
			p_ref_internal = P_ref[0, 0, 1:-1, 1:-1]

			# 计算L2相对误差
			l2_error_u = torch.sqrt(torch.mean((u_internal - u_ref_internal) ** 2)) / torch.sqrt(
				torch.mean(u_ref_internal ** 2))
			l2_error_v = torch.sqrt(torch.mean((v_internal - v_ref_internal) ** 2)) / torch.sqrt(
				torch.mean(v_ref_internal ** 2))
			l2_error_p = torch.sqrt(torch.mean((p_internal - p_ref_internal) ** 2)) / torch.sqrt(
				torch.mean(p_ref_internal ** 2))

		loss.backward()
		optimizer.step()






	if epoch % 100 == 0:
		print(f"[Epoch {epoch:4d}] Loss = {loss.item():.2e}")
		print(f"L2 Errors: U={l2_error_u.item():.4f}, V={l2_error_v.item():.4f}, P={l2_error_p.item():.4f}")

		# 使用整个网格而不是内部点切片
		X = coord[0, 0, :, :].cpu().numpy()  # 整个网格的X坐标
		Y = coord[0, 1, :, :].cpu().numpy()  # 整个网格的Y坐标
		U_p = u[0, 0, :, :].cpu().detach().numpy()  # 整个网格的U预测
		V_p = v[0, 0, :, :].cpu().detach().numpy()  # 整个网格的V预测
		Mag_p = np.sqrt(U_p ** 2 + V_p ** 2)  # 计算预测的速度幅值
		P_p = p[0, 0, :, :].cpu().detach().numpy()  # 整个网格的P预测
		U_r = U_ref[0, 0, :, :].cpu().numpy()  # 整个网格的U参考
		V_r = V_ref[0, 0, :, :].cpu().numpy()  # 整个网格的V参考
		Mag_r = Mag_ref[0, 0, :, :].cpu().numpy()  # 参考速度幅值
		P_r = P_ref[0, 0, :, :].cpu().numpy()  # 整个网格的P参考

		# 计算全局范围
		u_min = min(U_p.min(), U_r.min())
		u_max = max(U_p.max(), U_r.max())
		v_min = min(V_p.min(), V_r.min())
		v_max = max(V_p.max(), V_r.max())
		mag_min = min(Mag_p.min(), Mag_r.min())
		mag_max = max(Mag_p.max(), Mag_r.max())
		p_min = min(P_p.min(), P_r.min())
		p_max = max(P_p.max(), P_r.max())

		# 为每个物理量设置统一范围
		ranges = {
			'U': (u_min, u_max),
			'V': (v_min, v_max),
			'P': (p_min, p_max),
			'Velocity Magnitude': (mag_min, mag_max)
		}

		# 使用相同的颜色映射
		cmap = 'viridis'

		fig, axes = plt.subplots(3, 4, figsize=(20, 12))  # 3行：预测、参考、误差
		for i, (pred, ref, name) in enumerate(zip(
				[U_p, V_p, P_p, Mag_p], [U_r, V_r, P_r, Mag_r], ['U', 'V', 'P', 'Velocity Magnitude']
		)):
			vmin, vmax = ranges[name]

			# 预测解 (第1行)
			im0 = axes[0, i].contourf(X, Y, pred, levels=20, vmin=vmin, vmax=vmax, cmap=cmap)
			axes[0, i].set_title(f'Predicted {name}')
			plt.colorbar(im0, ax=axes[0, i])

			# 参考解 (第2行)
			im1 = axes[1, i].contourf(X, Y, ref, levels=20, vmin=vmin, vmax=vmax, cmap=cmap)
			axes[1, i].set_title(f'Reference {name}')
			plt.colorbar(im1, ax=axes[1, i])

			# 绝对误差 (第3行)
			error = np.abs(pred - ref)
			im2 = axes[2, i].contourf(X, Y, error, levels=20, cmap='hot')
			axes[2, i].set_title(f'Abs Error {name}')
			plt.colorbar(im2, ax=axes[2, i])

			# ================== 添加关键点标记 ==================
			for point_idx, (iy, ix) in enumerate(key_indices):
				# 转换为绘图坐标 - 直接使用原始索引
				x_val = X[iy, ix]
				y_val = Y[iy, ix]

				# 在预测图上标记
				axes[0, i].plot(x_val, y_val, 'ro', markersize=6, markeredgewidth=1, markerfacecolor='none')
				axes[0, i].text(x_val, y_val, str(point_idx + 1), color='white', fontsize=8, ha='center', va='center')

				# 在参考图上标记
				axes[1, i].plot(x_val, y_val, 'ro', markersize=6, markeredgewidth=1, markerfacecolor='none')
				axes[1, i].text(x_val, y_val, str(point_idx + 1), color='white', fontsize=8, ha='center', va='center')

				# 在误差图上标记
				axes[2, i].plot(x_val, y_val, 'ro', markersize=6, markeredgewidth=1, markerfacecolor='none')
				axes[2, i].text(x_val, y_val, str(point_idx + 1), color='white', fontsize=8, ha='center', va='center')

		# ================== 添加关键点标记结束 ==================

		# 添加标题说明
		plt.suptitle(f'Epoch {epoch}: Key Points (red circles)', fontsize=16)
		plt.tight_layout()
		plt.savefig(f'./output/完善/有数据点 - 硬Voronoi/epoch_{epoch:04d}.png', dpi=150)
		plt.close()  # 关闭图形，避免内存累积
	return loss.item(), l2_error_u.item(), l2_error_v.item(), l2_error_p.item()


if __name__ == '__main__':
	# 初始化列表记录训练过程
	losses = []
	errors_u = []
	errors_v = []
	errors_p = []

	t0 = time.time()
	for ep in range(1, nEpochs + 1):
		# 接收多个返回值
		loss, err_u, err_v, err_p = train(ep)

		# 记录损失和误差
		losses.append(loss)
		errors_u.append(err_u)
		errors_v.append(err_v)
		errors_p.append(err_p)

		# 定期保存模型和训练状态
		if ep % 100 == 0 or ep == nEpochs:
			print(f"保存模型: model_epoch_{ep}.pth")
			torch.save(model.state_dict(), f"./output/完善/有数据点 - 硬Voronoi/model_epoch_{ep}.pth")

			# 保存训练数据
			np.savetxt(f'./output/完善/有数据点 - 硬Voronoi/training_data_epoch_{ep}.csv',
					   np.column_stack([losses, errors_u, errors_v, errors_p]),
					   delimiter=',',
					   header='loss,err_u,err_v,err_p')

	# 计算总训练时间
	dt = time.time() - t0
	print(f"总训练时间: {dt:.2f} 秒")

	# 保存最终训练数据
	np.savetxt('./output/final_training_data.csv',
			   np.column_stack([losses, errors_u, errors_v, errors_p]),
			   delimiter=',',
			   header='loss,err_u,err_v,err_p')

	# 保存训练时间
	np.savetxt('./output/training_time.txt', [dt])

	# 绘制收敛曲线
	plt.figure(figsize=(12, 8))

	# 损失曲线
	plt.subplot(2, 1, 1)
	plt.semilogy(losses, 'b-', label='PDE Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Training Loss Convergence')
	plt.legend()
	plt.grid(True)

	# 误差曲线
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
	plt.savefig('training_convergence.png', dpi=300)
	plt.show()

	print("训练完成!")