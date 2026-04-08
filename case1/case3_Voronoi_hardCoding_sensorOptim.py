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
from pyMesh import hcubeMesh, visualize2D, plotBC, plotMesh,setAxisLabel,\
                   np2cuda
from model import USCNN
from voronoi_utils import VoronoiEnhancedUSCNN,to4DTensor,VaryGeoDataset_PairedSolution, LearnableKeyPointsOLD, generate_voronoi_input_torch
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
key_indices = [(b, a) for (a, b) in key_indices]
initial_positions = torch.tensor(
    [(y / 100.0, x / 100.0) for y, x in key_indices],
    dtype=torch.float32
)

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
NvarInput=2
NvarOutput=3
nEpochs=6000

lr = 0.0005
Ns=1
nu=0.01


model = VoronoiEnhancedUSCNN(h, nx, ny, NvarInput, NvarOutput).to('cuda')
key_points_model = LearnableKeyPointsOLD(initial_positions).to('cuda')

optimizer = optim.Adam([
    {'params': model.parameters()},
    {'params': key_points_model.parameters(), 'lr': 5e-4}
], lr=lr)
criterion = nn.MSELoss()
padSingleSide=1
udfpad=nn.ConstantPad2d([padSingleSide,padSingleSide,padSingleSide,padSingleSide],0)
MeshList=[]
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

training_data_loader=DataLoader(dataset=train_set,
	                            batch_size=batchSize)

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

	grid_size = (ny, nx)


	key_positions = key_points_model.positions

	indices_tensor = key_points_model(grid_size)
	key_indices_dyn = [(int(x), int(y))
					   for x, y in indices_tensor.detach().cpu().numpy()]


	for iteration, batch in enumerate(training_data_loader):
		[JJInv, coord_tensor, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta,
		 sol_u, sol_v, sol_p] = to4DTensor(batch)

		coord_2d = coord_tensor[0]
		sol_u_2d = sol_u[0, 0]
		sol_v_2d = sol_v[0, 0]
		sol_p_2d = sol_p[0, 0]

		voronoi_input = generate_voronoi_input_torch(
			coord_2d, sol_u_2d, sol_v_2d, sol_p_2d, key_positions, grid_size
		)

		optimizer.zero_grad()
		output = model(coord_tensor, voronoi_input)
		output_pad=udfpad(output)

		u = output_pad[:, 0:1, :, :]
		v = output_pad[:, 1:2, :, :]
		p = output_pad[:, 2:3, :, :]

		key_mask = voronoi_input[:, 3:4] > 0.5
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

		for iy, ix in key_indices_dyn:

			u_pred = u[0, 0, iy, ix]
			v_pred = v[0, 0, iy, ix]
			p_pred = p[0, 0, iy, ix]


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

		loss = loss_pde + loss_data

		with torch.no_grad():

			u_internal = u[0, 0, 1:-1, 1:-1]
			v_internal = v[0, 0, 1:-1, 1:-1]
			p_internal = p[0, 0, 1:-1, 1:-1]


			u_ref_internal = U_ref[0, 0, 1:-1, 1:-1]
			v_ref_internal = V_ref[0, 0, 1:-1, 1:-1]
			p_ref_internal = P_ref[0, 0, 1:-1, 1:-1]

			l2_error_u = torch.sqrt(torch.mean((u_internal - u_ref_internal) ** 2)) / torch.sqrt(
				torch.mean(u_ref_internal ** 2))
			l2_error_v = torch.sqrt(torch.mean((v_internal - v_ref_internal) ** 2)) / torch.sqrt(
				torch.mean(v_ref_internal ** 2))
			l2_error_p = torch.sqrt(torch.mean((p_internal - p_ref_internal) ** 2)) / torch.sqrt(
				torch.mean(p_ref_internal ** 2))

		loss.backward()
		optimizer.step()

	if epoch % 100 == 0:
		print("Optimized key points (normalized):")
		for i, pos in enumerate(key_points_model.positions.detach().cpu().numpy()):
			actual_pos = pos * [nx - 1, ny - 1]
			print(f"Point {i + 1}: ({actual_pos[0]:.2f}, {actual_pos[1]:.2f})")
		print(f"[Epoch {epoch:4d}] Loss = {loss.item():.2e}")
		print(f"L2 Errors: U={l2_error_u.item():.4f}, V={l2_error_v.item():.4f}, P={l2_error_p.item():.4f}")


		X = coord_tensor[0, 0, :, :].cpu().numpy()
		Y = coord_tensor[0, 1, :, :].cpu().numpy()
		U_p = u[0, 0, :, :].cpu().detach().numpy()
		V_p = v[0, 0, :, :].cpu().detach().numpy()
		Mag_p = np.sqrt(U_p ** 2 + V_p ** 2)
		P_p = p[0, 0, :, :].cpu().detach().numpy()
		U_r = U_ref[0, 0, :, :].cpu().numpy()
		V_r = V_ref[0, 0, :, :].cpu().numpy()
		Mag_r = Mag_ref[0, 0, :, :].cpu().numpy()
		P_r = P_ref[0, 0, :, :].cpu().numpy()


		u_min = min(U_p.min(), U_r.min())
		u_max = max(U_p.max(), U_r.max())
		v_min = min(V_p.min(), V_r.min())
		v_max = max(V_p.max(), V_r.max())
		mag_min = min(Mag_p.min(), Mag_r.min())
		mag_max = max(Mag_p.max(), Mag_r.max())
		p_min = min(P_p.min(), P_r.min())
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
			axes[0, i].set_title(f'Predicted {name}')
			plt.colorbar(im0, ax=axes[0, i])


			im1 = axes[1, i].contourf(X, Y, ref, levels=20, vmin=vmin, vmax=vmax, cmap=cmap)
			axes[1, i].set_title(f'Reference {name}')
			plt.colorbar(im1, ax=axes[1, i])


			error = np.abs(pred - ref)
			im2 = axes[2, i].contourf(X, Y, error, levels=20, cmap='hot')
			axes[2, i].set_title(f'Abs Error {name}')
			plt.colorbar(im2, ax=axes[2, i])


			for point_idx, (iy, ix) in enumerate(key_indices_dyn):

				x_val = X[iy, ix]
				y_val = Y[iy, ix]


				axes[0, i].plot(x_val, y_val, 'ro', markersize=6, markeredgewidth=1, markerfacecolor='none')
				axes[0, i].text(x_val, y_val, str(point_idx + 1), color='white', fontsize=8, ha='center', va='center')


				axes[1, i].plot(x_val, y_val, 'ro', markersize=6, markeredgewidth=1, markerfacecolor='none')
				axes[1, i].text(x_val, y_val, str(point_idx + 1), color='white', fontsize=8, ha='center', va='center')


				axes[2, i].plot(x_val, y_val, 'ro', markersize=6, markeredgewidth=1, markerfacecolor='none')
				axes[2, i].text(x_val, y_val, str(point_idx + 1), color='white', fontsize=8, ha='center', va='center')




		plt.suptitle(f'Epoch {epoch}: Key Points (red circles)', fontsize=16)
		plt.tight_layout()
		plt.savefig(f'./output/epoch_{epoch:04d}.png', dpi=150)
		plt.close()
	return loss.item(), l2_error_u.item(), l2_error_v.item(), l2_error_p.item()


if __name__ == '__main__':

	losses = []
	errors_u = []
	errors_v = []
	errors_p = []

	t0 = time.time()
	for ep in range(1, nEpochs + 1):

		loss, err_u, err_v, err_p = train(ep)

		losses.append(loss)
		errors_u.append(err_u)
		errors_v.append(err_v)
		errors_p.append(err_p)


		if ep % 100 == 0 or ep == nEpochs:
			print(f"保存模型: model_epoch_{ep}.pth")
			torch.save(model.state_dict(), f"./output/model_epoch_{ep}.pth")


			np.savetxt(f'./output/training_data_epoch_{ep}.csv',
					   np.column_stack([losses, errors_u, errors_v, errors_p]),
					   delimiter=',',
					   header='loss,err_u,err_v,err_p')

	dt = time.time() - t0
	print(f"总训练时间: {dt:.2f} 秒")


	np.savetxt('./output/final_training_data.csv',
			   np.column_stack([losses, errors_u, errors_v, errors_p]),
			   delimiter=',',
			   header='loss,err_u,err_v,err_p')

	np.savetxt('./output/training_time.txt', [dt])

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
	plt.savefig('training_convergence.png', dpi=300)
	plt.show()

	print("训练完成!")