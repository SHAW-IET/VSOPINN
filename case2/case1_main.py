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
                   np2cuda,to4DTensor
from model import USCNN,USCNNSepPhi,USCNNSep,DDBasic
from readOF import convertOFMeshToImage,convertOFMeshToImage_StructuredMesh
from sklearn.metrics import mean_squared_error as calMSE
import Ofpp
from pathlib import Path

# save_dir = Path(r"./output/origin/")

save_dir = Path(r"./output/2026.1.12完善/Re=250/1 无数据点")



h=0.01
OFBCCoord=Ofpp.parse_boundary_field('TemplateCase_simpleVessel/3200/C')            #  坐标 dict5
OFLOWC=OFBCCoord[b'low'][b'value']			# (49, 3)  x, y坐标, 第三个是 0 不用管
OFUPC=OFBCCoord[b'up'][b'value']			# (49, 3)
OFLEFTC=OFBCCoord[b'left'][b'value']		# (77, 3)
OFRIGHTC=OFBCCoord[b'rifht'][b'value']		# (77, 3)

leftX=OFLEFTC[:,0];leftY=OFLEFTC[:,1]
lowX=OFLOWC[:,0];lowY=OFLOWC[:,1]
rightX=OFRIGHTC[:,0];rightY=OFRIGHTC[:,1]
upX=OFUPC[:,0];upY=OFUPC[:,1]				# 分别取出 x， y 坐标
ny=len(leftX);nx=len(lowX)
myMesh=hcubeMesh(leftX,leftY,rightX,rightY,
	             lowX,lowY,upX,upY,h,True,True,
	             tolMesh=1e-10,tolJoint=1e-2)
####
batchSize=1
NvarInput=2
NvarOutput=1			# 分开预测了所以是 1， 一起的话是 3
nEpochs=30000
lr=0.001
Ns=1              # 没用


# nu=0.0008				# 粘度 ################################################# 记得改！！！！！！！！！！！！！！！！！！！！！！！   Re = 250
nu= 0.02 / 45.00			# 粘度 ################################################# 记得改！！！！！！！！！！！！！！！！！！！！！！！   Re = 450
# nu=0.002				# 粘度 ################################################# 记得改！！！！！！！！！！！！！！！！！！！！！！！   Re = 100
# nu=0.01				# 粘度 ################################################# 记得改！！！！！！！！！！！！！！！！！！！！！！！   Re = 20（原算例）

model=USCNNSep(h,nx,ny,NvarInput,NvarOutput,'ortho').to('cuda')
# model=torch.load('./Result/15000.pth')
model=model.to('cuda')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=lr)
padSingleSide=1
udfpad=nn.ConstantPad2d([padSingleSide,padSingleSide,padSingleSide,padSingleSide],0)
####
MeshList=[]
MeshList.append(myMesh)
train_set=VaryGeoDataset(MeshList)
training_data_loader=DataLoader(dataset=train_set,
	                            batch_size=batchSize)
OFPic=convertOFMeshToImage_StructuredMesh(nx,ny,'TemplateCase_simpleVessel/3200/C',
	                                           ['Re250/3200/U',
	                                            'Re250/3200/p'],
	                                            [0,1,0,1],0.0,False)							# (77, 49, 5)
OFX=OFPic[:,:,0]				# (77, 49)
OFY=OFPic[:,:,1]
OFU=OFPic[:,:,2]
OFV=OFPic[:,:,3]
OFP=OFPic[:,:,4]
OFU_sb=np.zeros(OFU.shape)
OFV_sb=np.zeros(OFV.shape)
OFP_sb=np.zeros(OFP.shape)
fcnn_P=np.zeros(OFU.shape)
fcnn_U=np.zeros(OFV.shape)
fcnn_V=np.zeros(OFP.shape)
fcnn=np.load('comparison_160000iter.npz')		 # (77, 49)  用于对比的值
fcnn_P_=fcnn['p_NN'].reshape(OFU_sb.shape)
fcnn_U_=fcnn['u_NN'].reshape(OFU_sb.shape)
fcnn_V_=fcnn['v_NN'].reshape(OFU_sb.shape)
fcnn_X=fcnn['x_coord'].reshape(OFU_sb.shape)
fcnn_Y=fcnn['y_coord'].reshape(OFU_sb.shape)

for i in range(nx):					# 最近邻插值到当前网格  fcnn解
	for j in range(ny):
		dist=(myMesh.x[j,i]-fcnn_X)**2+(myMesh.y[j,i]-fcnn_Y)**2
		idx_min=np.where(dist == dist.min())
		fcnn_U[j,i]=fcnn_U_[idx_min]
		fcnn_V[j,i]=fcnn_V_[idx_min]
		fcnn_P[j,i]=fcnn_P_[idx_min]

for i in range(nx):					# 最近邻插值到当前网格 OpenFOAM解
	for j in range(ny):
		dist=(myMesh.x[j,i]-OFX)**2+(myMesh.y[j,i]-OFY)**2
		idx_min=np.where(dist == dist.min())
		OFU_sb[j,i]=OFU[idx_min]
		OFV_sb[j,i]=OFV[idx_min]
		OFP_sb[j,i]=OFP[idx_min]


OFMag_sb = np.sqrt(OFU_sb ** 2 + OFV_sb ** 2)

U_ref = torch.tensor(OFU_sb.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)
V_ref = torch.tensor(OFV_sb.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)
Mag_ref = torch.tensor(OFMag_sb.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)
P_ref = torch.tensor(OFP_sb.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)

SolutionList = [np.stack([OFU_sb, OFV_sb, OFP_sb], axis=-1)]  # 组合成 (101, 101, 3)




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
	startTime=time.time()
	xRes=0
	yRes=0
	mRes=0
	eU=0
	eV=0
	eP=0
	for iteration, batch in enumerate(training_data_loader):
		[JJInv,coord,xi,eta,J,Jinv,dxdxi,dydxi,dxdeta,dydeta]=to4DTensor(batch)
		optimizer.zero_grad()
		output=model(coord)			# (1, 3, 75, 47)
		output_pad=udfpad(output)			# (1, 3, 77, 49)

		outputU=output_pad[:,0,:,:].reshape(output_pad.shape[0],1,
			                                output_pad.shape[2],
			                                output_pad.shape[3])			# (1, 1, 77, 49) 取出 u v p
		outputV=output_pad[:,1,:,:].reshape(output_pad.shape[0],1,
			                                output_pad.shape[2],
			                                output_pad.shape[3])			# (1, 1, 77, 49)
		outputP=output_pad[:,2,:,:].reshape(output_pad.shape[0],1,
			                                output_pad.shape[2],
			                                output_pad.shape[3])			# (1, 1, 77, 49)

		# XR=torch.zeros([batchSize,1,ny,nx]).to('cuda')
		# YR=torch.zeros([batchSize,1,ny,nx]).to('cuda')
		# MR=torch.zeros([batchSize,1,ny,nx]).to('cuda')


		for j in range(batchSize):			# 边界条件 下边界速度 (0, 1)，左右是无滑移边界条件，上边界 v·n = 0和 p = 0， batchsize = 1
			outputU[j,0,-padSingleSide:,padSingleSide:-padSingleSide]=output[j,0,-1,:].reshape(1,nx-2*padSingleSide) # 上边界 (y_max) 取模型输出的上边界，自由出口？
			outputU[j,0,:padSingleSide,padSingleSide:-padSingleSide]=0  # 下边界 (y_min) u = 0
			outputU[j,0,padSingleSide:-padSingleSide,-padSingleSide:]=0 # 右边界 (x_max) u = 0
			outputU[j,0,padSingleSide:-padSingleSide,0:padSingleSide]=0 # 左边界 (x_min) u = 0
			outputU[j,0,0,0]=1*(outputU[j,0,0,1])			# 左下角点 (0,0)  避免角点奇异
			outputU[j,0,0,-1]=1*(outputU[j,0,0,-2])			# 右下角点 (0,-1)
			outputV[j,0,-padSingleSide:,padSingleSide:-padSingleSide]=output[j,1,-1,:].reshape(1,nx-2*padSingleSide) # 上边界 (y_max) 取模型输出的上边界，自由出口？
			outputV[j,0,:padSingleSide,padSingleSide:-padSingleSide]=1				# 下边界 (y_min) v = 1
			outputV[j,0,padSingleSide:-padSingleSide,-padSingleSide:]=0 			# 右边界 (x_max) v = 0
			outputV[j,0,padSingleSide:-padSingleSide,0:padSingleSide]=0 			# 左边界 (x_min) v = 0
			outputV[j,0,0,0]=1*(outputV[j,0,0,1])		# 左下角点 (0,0)
			outputV[j,0,0,-1]=1*(outputV[j,0,0,-2])		# 右下角点 (0,-1)
			outputP[j,0,-padSingleSide:,padSingleSide:-padSingleSide]=0 			# 上边界 (y_max) p = 0 参考大气压力为 0
			outputP[j,0,:padSingleSide,padSingleSide:-padSingleSide]=output[j,2,0,:].reshape(1,nx-2*padSingleSide)     # 下边界 使用模型预测的第一行(0,:)
			outputP[j,0,padSingleSide:-padSingleSide,-padSingleSide:]=output[j,2,:,-1].reshape(ny-2*padSingleSide,1)    	# 右边界 使用模型预测的边界列(:,-1和:,0)
			outputP[j,0,padSingleSide:-padSingleSide,0:padSingleSide]=output[j,2,:,0].reshape(ny-2*padSingleSide,1)     # 左边界 使用模型预测的边界列(:,-1和:,0)
			outputP[j,0,0,0]=1*(outputP[j,0,0,1])		# 左下角点 (0,0)
			outputP[j,0,0,-1]=1*(outputP[j,0,0,-2])		# 右下角点 (0,-1)


			#边界位置	速度u	速度v	压力p	物理意义
			#上边界	自由滑移	自由滑移	p=0	出口/自由表面
			#下边界	u=0	v=1	∂p/∂y=0	入口(均匀来流)
			#左边界	u=0	v=0	∂p/∂x=0	固壁(无滑移)
			#右边界	u=0	v=0	∂p/∂x=0	固壁(无滑移)

		# 计算的导数
		dudx=dfdx(outputU,dydeta,dydxi,Jinv)
		d2udx2=dfdx(dudx,dydeta,dydxi,Jinv)
		dudy=dfdy(outputU,dxdxi,dxdeta,Jinv)
		d2udy2=dfdy(dudy,dxdxi,dxdeta,Jinv)
		dvdx=dfdx(outputV,dydeta,dydxi,Jinv)
		d2vdx2=dfdx(dvdx,dydeta,dydxi,Jinv)
		dvdy=dfdy(outputV,dxdxi,dxdeta,Jinv)
		d2vdy2=dfdy(dvdy,dxdxi,dxdeta,Jinv)
		dpdx=dfdx(outputP,dydeta,dydxi,Jinv)
		d2pdx2=dfdx(dpdx,dydeta,dydxi,Jinv)
		dpdy=dfdy(outputP,dxdxi,dxdeta,Jinv)
		d2pdy2=dfdy(dpdy,dxdxi,dxdeta,Jinv)

		# 物理方程
		continuity=dudx+dvdy;
		momentumX=outputU*dudx+outputV*dudy
		forceX=-dpdx+nu*(d2udx2+d2udy2)

		Xresidual=momentumX-forceX   
		momentumY=outputU*dvdx+outputV*dvdy

		forceY=-dpdy+nu*(d2vdx2+d2vdy2)
		Yresidual=momentumY-forceY

		loss=(criterion(Xresidual,Xresidual*0)+\
		  criterion(Yresidual,Yresidual*0)+\
		  criterion(continuity,continuity*0))

		# sl = slice(1, -1)
		# loss = (
		# 		criterion(Xresidual[:, :, sl, sl], torch.zeros_like(Xresidual[:, :, sl, sl])) +
		# 		criterion(Yresidual[:, :, sl, sl], torch.zeros_like(Yresidual[:, :, sl, sl])) +
		# 		criterion(continuity[:, :, sl, sl], torch.zeros_like(continuity[:, :, sl, sl]))
		# )

		with torch.no_grad():
			Mag_pred = torch.sqrt(outputU ** 2 + outputV ** 2)
			# 计算L2相对误差
			l2_error_u = torch.sqrt(torch.mean((outputU - U_ref) ** 2)) / torch.sqrt(
				torch.mean(U_ref ** 2))
			l2_error_v = torch.sqrt(torch.mean((outputV - V_ref) ** 2)) / torch.sqrt(
				torch.mean(V_ref ** 2))
			l2_error_p = torch.sqrt(torch.mean((outputP - P_ref) ** 2)) / torch.sqrt(
				torch.mean(P_ref ** 2))
			l2_error_mag = torch.sqrt(torch.mean((Mag_pred - Mag_ref) ** 2)) / torch.sqrt(
				torch.mean(Mag_ref ** 2))
			l2_error_av = (l2_error_u + l2_error_v + l2_error_p + l2_error_mag) / 4.0  # 平均值


		loss.backward()
		optimizer.step()


	# 	loss_xm=criterion(Xresidual, Xresidual*0)			# x动量
	# 	loss_ym=criterion(Yresidual, Yresidual*0)			# y动量
	# 	loss_mass=criterion(continuity, continuity*0)			# 连续性
	#
	#
	# 	xRes+=loss_xm.item()
	# 	yRes+=loss_ym.item()
	# 	mRes+=loss_mass.item()
	#
	# 	CNNUNumpy=outputU[0,0,:,:].cpu().detach().numpy()
	# 	CNNVNumpy=outputV[0,0,:,:].cpu().detach().numpy()
	# 	CNNPNumpy=outputP[0,0,:,:].cpu().detach().numpy()
	#
	# 	eU=eU+np.sqrt(calMSE(OFU_sb,CNNUNumpy)/calMSE(OFU_sb,OFU_sb*0))
	# 	eV=eV+np.sqrt(calMSE(OFV_sb,CNNVNumpy)/calMSE(OFV_sb,OFV_sb*0))
	# 	eP=eP+np.sqrt(calMSE(OFP_sb,CNNPNumpy)/calMSE(OFP_sb,OFP_sb*0))
	#
	# 	# RMSE OF  0-1 velocity magnitude
	# 	eVmag=np.sqrt(calMSE(np.sqrt(OFU_sb**2+OFV_sb**2),np.sqrt(CNNUNumpy**2+CNNVNumpy**2))/calMSE(np.sqrt(OFU_sb**2+OFV_sb**2),np.sqrt(OFU_sb**2+OFV_sb**2)*0))
	# 	# RMSE FCNN   0-1 velocity magnitude
	# 	eVmag_FCNN=np.sqrt(calMSE(np.sqrt(OFU_sb**2+OFV_sb**2),np.sqrt(fcnn_U**2+fcnn_V**2))/calMSE(np.sqrt(OFU_sb**2+OFV_sb**2),np.sqrt(OFU_sb**2+OFV_sb**2)*0))
	#
	# 	print('VelMagError_CNN=',eVmag)
	# 	print('VelMagError_FCNN=',eVmag_FCNN)
	# 	print('P_err_CNN=',np.sqrt(calMSE(OFP_sb,CNNPNumpy)/calMSE(OFP_sb,OFP_sb*0)))
	# 	print('P_err_FCNN=',np.sqrt(calMSE(OFP_sb,fcnn_P)/calMSE(OFP_sb,OFP_sb*0)))
	#
	# print('Epoch is ',epoch)
	# print("xRes Loss is", (xRes/len(training_data_loader)))		# x动量残差
	# print("yRes Loss is", (yRes/len(training_data_loader)))		# y动量残差
	# print("mRes Loss is", (mRes/len(training_data_loader)))		# 连续性残差
	# print("eU Loss is", (eU/len(training_data_loader)))			# U V P 的归一化 RMSE
	# print("eV Loss is", (eV/len(training_data_loader)))
	# print("eP Loss is", (eP/len(training_data_loader)))



	if epoch%100 ==0:
		print(f"[Epoch {epoch:4d}] Loss = {loss.item():.2e}")
		print(
			f"L2 Errors: U={l2_error_u.item():.4f}, V={l2_error_v.item():.4f}, P={l2_error_p.item():.4f}, Mag={l2_error_mag.item():.4f}, AV={l2_error_av.item():.4f}")

	if epoch%5000==0 or epoch%nEpochs==0:
		# torch.save(model, str(epoch)+'.pth')
		ckpt_path = save_dir / f"epoch_{epoch:04d}.pth"
		torch.save(model, ckpt_path)

		fig0=plt.figure()
		ax=plt.subplot(2,3,1)
		_,cbar=visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           np.sqrt(fcnn_U[1:-1,1:-1]**2+\
			           		   fcnn_V[1:-1,1:-1]**2),'vertical',[0,1.3])
		cbar.set_ticks([0,0.3,0.6,0.9,1.3])
		setAxisLabel(ax,'p')
		ax.set_title('FCNN '+'Velocity')

		ax=plt.subplot(2,3,2)
		_,cbar=visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           np.sqrt(outputU[0,0,1:-1,1:-1].cpu().detach().numpy()**2+\
			           		   outputV[0,0,1:-1,1:-1].cpu().detach().numpy()**2),'vertical',[0,1.3])
		setAxisLabel(ax,'p')
		ax.set_title('PhyGeoNet '+'Velocity')
		cbar.set_ticks([0,0.3,0.6,0.9,1.3])

		ax=plt.subplot(2,3,3)
		_,cbar=visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           np.sqrt(OFU_sb[1:-1,1:-1]**2+\
			           		   OFV_sb[1:-1,1:-1]**2),'vertical',[0,1.3])
		cbar.set_ticks([0,0.3,0.6,0.9,1.3])
		setAxisLabel(ax,'p')
		ax.set_title('CFD '+'Velocity')
		

		ax=plt.subplot(2,3,4)
		visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           fcnn_P[1:-1,1:-1],'vertical',[0,0.5])
		setAxisLabel(ax,'p')
		ax.set_title('FCNN '+'Pressure')
		
		ax=plt.subplot(2,3,5)
		visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           outputP[0,0,1:-1,1:-1].cpu().detach().numpy(),'vertical',[0,0.5])
		setAxisLabel(ax,'p')
		ax.set_title('PhyGeoNet '+'Pressure')

		ax=plt.subplot(2,3,6)
		visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           OFP_sb[1:-1,1:-1],'vertical',[0,0.5])
		setAxisLabel(ax,'p')
		ax.set_title('CFD '+'Pressure')
		fig0.tight_layout(pad=1)
		# fig0.savefig(str(epoch)+'VelMagAndPressureFCNN.pdf',bbox_inches='tight')
		fig0_path = save_dir / f"epoch_{epoch:04d}_VelMagAndPressureFCNN.pdf"
		fig0.savefig(fig0_path, bbox_inches='tight')

		plt.close(fig0)

		fig0=plt.figure()
		ax=plt.subplot(2,2,1)
		_,cbar=visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           np.sqrt(outputU[0,0,1:-1,1:-1].cpu().detach().numpy()**2+\
			           		   outputV[0,0,1:-1,1:-1].cpu().detach().numpy()**2),'vertical',[0,1.3])
		setAxisLabel(ax,'p')
		ax.set_title('PhyGeoNet '+'Velocity')
		ax.set_aspect(1.3)
		cbar.set_ticks([0,0.3,0.6,0.9,1.3])		
		ax=plt.subplot(2,2,2)
		visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           outputP[0,0,1:-1,1:-1].cpu().detach().numpy(),'vertical',[0,0.5])
		setAxisLabel(ax,'p')
		ax.set_title('PhyGeoNet '+'Pressure')
		ax.set_aspect(1.3)
		ax=plt.subplot(2,2,3)
		_,cbar=visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           np.sqrt(OFU_sb[1:-1,1:-1]**2+\
			           		   OFV_sb[1:-1,1:-1]**2),'vertical',[0,1.3])
		setAxisLabel(ax,'p')
		ax.set_title('CFD '+'Velocity')
		ax.set_aspect(1.3)
		cbar.set_ticks([0,0.3,0.6,0.9,1.3])
		ax=plt.subplot(2,2,4)
		visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           OFP_sb[1:-1,1:-1],'vertical',[0,0.5])
		setAxisLabel(ax,'p')
		ax.set_title('CFD '+'Pressure')
		ax.set_aspect(1.3)
		fig0.tight_layout(pad=1)
		# fig0.savefig(str(epoch)+'VelMagAndPressureCNN.pdf',bbox_inches='tight')
		fig1_path = save_dir / f"epoch_{epoch:04d}_VelMagAndPressureCNN.pdf"
		fig0.savefig(fig1_path, bbox_inches='tight')
		plt.close(fig0)
	return loss.item(), l2_error_u.item(), l2_error_v.item(), l2_error_p.item(), l2_error_mag.item(), l2_error_av.item()


# return (xRes / len(training_data_loader)), (yRes / len(training_data_loader)), \
#     (mRes / len(training_data_loader)), (eU / len(training_data_loader)), \
#     (eV / len(training_data_loader)), (eP / len(training_data_loader))



if __name__ == '__main__':
	# 初始化列表记录训练过程
	losses = []
	errors_u = []
	errors_v = []
	errors_p = []
	errors_mag = []
	errors_av = []

	t0 = time.time()
	for ep in range(1, nEpochs + 1):
		# 接收多个返回值
		loss, err_u, err_v, err_p, err_mag, err_av = train(ep)

		# 记录损失和误差
		losses.append(loss)
		errors_u.append(err_u)
		errors_v.append(err_v)
		errors_p.append(err_p)
		errors_mag.append(err_mag)
		errors_av.append(err_av)

		# 定期保存模型和训练状态
		if ep % 100 == 0 or ep == nEpochs:
			# print(f"保存模型: model_epoch_{ep}.pth")
			# torch.save(model.state_dict(), f"./output/有Voronoi/model_epoch_{ep}.pth")

			# 保存训练数据
			data_path = save_dir / f"data_epoch_{ep}.csv"
			np.savetxt(data_path,
					   np.column_stack([losses, errors_u, errors_v, errors_p, errors_mag,errors_av]),
					   delimiter=',',
					   header='loss,err_u,err_v,err_p,err_mag, err_av')

	# 计算总训练时间
	dt = time.time() - t0
	print(f"总训练时间: {dt:.2f} 秒")

	# 保存最终训练数据
	fina_data_path = save_dir / f"final_data.csv"
	np.savetxt(fina_data_path,
			   np.column_stack([losses, errors_u, errors_v, errors_p, errors_mag, errors_av]),
			   delimiter=',',
			   header='loss,err_u,err_v,err_p,err_mag, err_av')

	# 保存训练时间
	time_path = save_dir / f"time.txt"
	np.savetxt(time_path, [dt])

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
	convergence_path = save_dir / f"training_convergence.png"

	plt.savefig(convergence_path, dpi=300)
	plt.show()





	###########最小值计算
	# —— 1. 全部 epoch 中的最小值 —— #
	# 构造 ndarray 和对应的 epoch 数组
	epochs = np.arange(1, len(errors_u) + 1)  # [1, 2, …, num_epochs]
	u_arr = np.array(errors_u)
	v_arr = np.array(errors_v)
	p_arr = np.array(errors_p)
	mag_arr = np.array(errors_mag)
	av_arr = np.array(errors_av)


	# 定义一个小函数打印最小值及所在 epoch
	def print_min(name, arr, epochs):
		idx = arr.argmin()  # 最小值索引
		print(f"{name} 最小 L2 误差：{arr[idx]:.4f}，出现在 epoch {epochs[idx]}")


	# 依次打印
	print("\n—— 全部 epoch 的最小误差 ——")
	print_min("U", u_arr, epochs)  # U 最小
	print_min("V", v_arr, epochs)  # V 最小
	print_min("P", p_arr, epochs)  # P 最小
	print_min("Mag", mag_arr, epochs)  # Mag 最小
	print_min("AV", av_arr, epochs)  # AV 最小

	# —— 2. 仅考虑 100 倍数 epoch 的最小值 —— #
	mask = (epochs % 100 == 0)
	mask_epochs = epochs[mask]  # [100, 200, …]
	print("\n—— 100 倍数 epoch 的最小误差 ——")
	print_min("U", u_arr[mask], mask_epochs)
	print_min("V", v_arr[mask], mask_epochs)
	print_min("P", p_arr[mask], mask_epochs)
	print_min("Mag", mag_arr[mask], mask_epochs)
	print_min("AV", av_arr[mask], mask_epochs)

	# 打开文件写入
	min_errors_summary_path = save_dir / f"min_errors_summary.txt"
	with open(min_errors_summary_path, 'w', encoding='utf-8') as f:
		f.write('—— 全部 epoch 的最小误差 ——\n')
		for name, arr in [('U', u_arr), ('V', v_arr), ('P', p_arr), ('Mag', mag_arr), ('AV', av_arr)]:
			idx = arr.argmin()
			f.write(f"{name} 最小 L2 误差：{arr[idx]:.4f}，出现在 epoch {epochs[idx]}\n")

		f.write('\n—— 100 倍数 epoch 的最小误差 ——\n')
		mask = (epochs % 100 == 0)
		mask_epochs = epochs[mask]
		for name, arr in [('U', u_arr), ('V', v_arr), ('P', p_arr), ('Mag', mag_arr), ('AV', av_arr)]:
			sub = arr[mask]
			idx = sub.argmin()
			f.write(f"{name} 最小 L2 误差：{sub[idx]:.4f}，出现在 epoch {mask_epochs[idx]}\n")

	print(f"已将最小误差结果保存到 {min_errors_summary_path}")

	print("训练完成!")





























# XRes=[];YRes=[];MRes=[]
# EU=[];EV=[];EP=[]
# TotalstartTime=time.time()
#
# for epoch in range(1,nEpochs+1):
# 	xres,yres,mres,eu,ev,ep=train(epoch)
# 	XRes.append(xres)
# 	YRes.append(yres)
# 	MRes.append(mres)
# 	EU.append(eu)
# 	EV.append(ev)
# 	EP.append(ep)
# TimeSpent=time.time()-TotalstartTime
#
# plt.figure()
# plt.plot(XRes,'-o',label='X-momentum Residual')
# plt.plot(YRes,'-x',label='Y-momentum Residual')
# plt.plot(MRes,'-*',label='Continuity Residual')
# plt.xlabel('Epoch')
# plt.ylabel('Residual')
# plt.legend()
# plt.yscale('log')
# plt.savefig('./output/origin/convergence.pdf',bbox_inches='tight')
# # tikzplotlib.save('convergence.tikz')
#
#
# plt.figure()
# plt.plot(EU,'-o',label=r'$u$')
# plt.plot(EV,'-x',label=r'$v$')
# plt.plot(EP,'-*',label=r'$p$')
# plt.xlabel('Epoch')
# plt.ylabel('Error')
# plt.legend()
# plt.yscale('log')
# plt.savefig('./output/origin/error.pdf',bbox_inches='tight')
# # tikzplotlib.save('error.tikz')
#
# EU=np.asarray(EU)
# EV=np.asarray(EV)
# EP=np.asarray(EP)
# XRes=np.asarray(XRes)
# YRes=np.asarray(YRes)
# MRes=np.asarray(MRes)
# np.savetxt('./output/origin/EU.txt',EU)
# np.savetxt('./output/origin/EV.txt',EV)
# np.savetxt('./output/origin/EP.txt',EP)
# np.savetxt('./output/origin/XRes.txt',XRes)
# np.savetxt('./output/origin/YRes.txt',YRes)
# np.savetxt('./output/origin/MRes.txt',MRes)
# np.savetxt('./output/origin/TimeSpent.txt',np.zeros([2,2])+TimeSpent)























































'''
			dudx=Jinv[j:j+1,0:1,:,:]*(model.convdxi(outputU[j:j+1,0:1,:,:])*dydeta[j:j+1,0:1,:,:]-\
			     model.convdeta(outputU[j:j+1,0:1,:,:])*dydxi[j:j+1,0:1,:,:])
			d2udx2=Jinv[j:j+1,0:1,2:-2,2:-2]*(model.convdxi(dudx)*dydeta[j:j+1,0:1,2:-2,2:-2]-\
			       model.convdeta(dudx)*dydxi[j:j+1,0:1,2:-2,2:-2])
			dvdx=Jinv[j:j+1,0:1,:,:]*(model.convdxi(outputV[j:j+1,0:1,:,:])*dydeta[j:j+1,0:1,:,:]-\
			     model.convdeta(outputV[j:j+1,0:1,:,:])*dydxi[j:j+1,0:1,:,:])
			d2vdx2=Jinv[j:j+1,0:1,2:-2,2:-2]*(model.convdxi(dvdx)*dydeta[j:j+1,0:1,2:-2,2:-2]-\
			       model.convdeta(dvdx)*dydxi[j:j+1,0:1,2:-2,2:-2])

			dudy=Jinv[j:j+1,0:1,:,:]*(model.convdeta(outputU[j:j+1,0:1,:,:])*dxdxi[j:j+1,0:1,:,:]-\
			     model.convdxi(outputU[j:j+1,0:1,:,:])*dxdeta[j:j+1,0:1,:,:])
			d2udy2=Jinv[j:j+1,0:1,2:-2,2:-2]*(model.convdeta(dudy)*dxdxi[j:j+1,0:1,2:-2,2:-2]-\
			     model.convdxi(dudy)*dxdeta[j:j+1,0:1,2:-2,2:-2])
			dvdy=Jinv[j:j+1,0:1,:,:]*(model.convdeta(outputV[j:j+1,0:1,:,:])*dxdxi[j:j+1,0:1,:,:]-\
			     model.convdxi(outputV[j:j+1,0:1,:,:])*dxdeta[j:j+1,0:1,:,:])
			d2vdy2=Jinv[j:j+1,0:1,2:-2,2:-2]*(model.convdeta(dvdy)*dxdxi[j:j+1,0:1,2:-2,2:-2]-\
			     model.convdxi(dvdy)*dxdeta[j:j+1,0:1,2:-2,2:-2])

			dpdx=Jinv[j:j+1,0:1,:,:]*(model.convdxi(outputP[j:j+1,0:1,:,:])*dydeta[j:j+1,0:1,:,:]-\
			     model.convdeta(outputP[j:j+1,0:1,:,:])*dydxi[j:j+1,0:1,:,:])
			dpdy=Jinv[j:j+1,0:1,:,:]*(model.convdeta(outputP[j:j+1,0:1,:,:])*dxdxi[j:j+1,0:1,:,:]-\
			     model.convdxi(outputP[j:j+1,0:1,:,:])*dxdeta[j:j+1,0:1,:,:])

			continuity=dudx[:,:,2:-2,2:-2]+dudy[:,:,2:-2,2:-2];
			#u*dudx+v*dudy
			momentumX=outputU[j:j+1,:,2:-2,2:-2]*dudx+\
			          outputV[j:j+1,:,2:-2,2:-2]*dvdx
			#-dpdx+nu*lap(u)
			forceX=-dpdx[0:,0:,2:-2,2:-2]+nu*(d2udx2+d2udy2)
			# Xresidual
			Xresidual=momentumX[0:,0:,2:-2,2:-2]-forceX   

			#u*dvdx+v*dvdy
			momentumY=outputU[j:j+1,:,2:-2,2:-2]*dvdx+\
			          outputV[j:j+1,:,2:-2,2:-2]*dvdy
			#-dpdy+nu*lap(v)
			forceY=-dpdy[0:,0:,2:-2,2:-2]+nu*(d2vdx2+d2vdy2)
			# Yresidual
			Yresidual=momentumY[0:,0:,2:-2,2:-2]-forceY 
			'''



#####  修改操作
# 去掉 pth载入 用定义的模型
# 去掉 save tikz
# 修改 epoch
# loss.backward()启动