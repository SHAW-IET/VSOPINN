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


save_dir = Path(r"./output/2026.1.12_improved/Re=250/1 no_data_points")


h=0.01
OFBCCoord=Ofpp.parse_boundary_field('TemplateCase_simpleVessel/3200/C')
OFLOWC=OFBCCoord[b'low'][b'value']
OFUPC=OFBCCoord[b'up'][b'value']
OFLEFTC=OFBCCoord[b'left'][b'value']
OFRIGHTC=OFBCCoord[b'rifht'][b'value']

leftX=OFLEFTC[:,0];leftY=OFLEFTC[:,1]
lowX=OFLOWC[:,0];lowY=OFLOWC[:,1]
rightX=OFRIGHTC[:,0];rightY=OFRIGHTC[:,1]
upX=OFUPC[:,0];upY=OFUPC[:,1]
ny=len(leftX);nx=len(lowX)
myMesh=hcubeMesh(leftX,leftY,rightX,rightY,
              lowX,lowY,upX,upY,h,True,True,
              tolMesh=1e-10,tolJoint=1e-2)

batchSize=1
NvarInput=2
NvarOutput=1
nEpochs=30000
lr=0.001
Ns=1


nu= 0.02 / 45.00


model=USCNNSep(h,nx,ny,NvarInput,NvarOutput,'ortho').to('cuda')

model=model.to('cuda')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=lr)
padSingleSide=1
udfpad=nn.ConstantPad2d([padSingleSide,padSingleSide,padSingleSide,padSingleSide],0)

MeshList=[]
MeshList.append(myMesh)
train_set=VaryGeoDataset(MeshList)
training_data_loader=DataLoader(dataset=train_set,
                             batch_size=batchSize)
OFPic=convertOFMeshToImage_StructuredMesh(nx,ny,'TemplateCase_simpleVessel/3200/C',
                                            ['Re250/3200/U',
                                             'Re250/3200/p'],
                                             [0,1,0,1],0.0,False)
OFX=OFPic[:,:,0]
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
fcnn=np.load('comparison_160000iter.npz')
fcnn_P_=fcnn['p_NN'].reshape(OFU_sb.shape)
fcnn_U_=fcnn['u_NN'].reshape(OFU_sb.shape)
fcnn_V_=fcnn['v_NN'].reshape(OFU_sb.shape)
fcnn_X=fcnn['x_coord'].reshape(OFU_sb.shape)
fcnn_Y=fcnn['y_coord'].reshape(OFU_sb.shape)

for i in range(nx):
	for j in range(ny):
		dist=(myMesh.x[j,i]-fcnn_X)**2+(myMesh.y[j,i]-fcnn_Y)**2
		idx_min=np.where(dist == dist.min())
		fcnn_U[j,i]=fcnn_U_[idx_min]
		fcnn_V[j,i]=fcnn_V_[idx_min]
		fcnn_P[j,i]=fcnn_P_[idx_min]

for i in range(nx):
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

SolutionList = [np.stack([OFU_sb, OFV_sb, OFP_sb], axis=-1)]


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
		output=model(coord)
		output_pad=udfpad(output)

		outputU=output_pad[:,0,:,:].reshape(output_pad.shape[0],1,
		                                 output_pad.shape[2],
		                                 output_pad.shape[3])
		outputV=output_pad[:,1,:,:].reshape(output_pad.shape[0],1,
		                                 output_pad.shape[2],
		                                 output_pad.shape[3])
		outputP=output_pad[:,2,:,:].reshape(output_pad.shape[0],1,
		                                 output_pad.shape[2],
		                                 output_pad.shape[3])


		for j in range(batchSize):
			outputU[j,0,-padSingleSide:,padSingleSide:-padSingleSide]=output[j,0,-1,:].reshape(1,nx-2*padSingleSide)
			outputU[j,0,:padSingleSide,padSingleSide:-padSingleSide]=0
			outputU[j,0,padSingleSide:-padSingleSide,-padSingleSide:]=0
			outputU[j,0,padSingleSide:-padSingleSide,0:padSingleSide]=0
			outputU[j,0,0,0]=1*(outputU[j,0,0,1])
			outputU[j,0,0,-1]=1*(outputU[j,0,0,-2])
			outputV[j,0,-padSingleSide:,padSingleSide:-padSingleSide]=output[j,1,-1,:].reshape(1,nx-2*padSingleSide)
			outputV[j,0,:padSingleSide,padSingleSide:-padSingleSide]=1
			outputV[j,0,padSingleSide:-padSingleSide,-padSingleSide:]=0
			outputV[j,0,padSingleSide:-padSingleSide,0:padSingleSide]=0
			outputV[j,0,0,0]=1*(outputV[j,0,0,1])
			outputV[j,0,0,-1]=1*(outputV[j,0,0,-2])
			outputP[j,0,-padSingleSide:,padSingleSide:-padSingleSide]=0
			outputP[j,0,:padSingleSide,padSingleSide:-padSingleSide]=output[j,2,0,:].reshape(1,nx-2*padSingleSide)
			outputP[j,0,padSingleSide:-padSingleSide,-padSingleSide:]=output[j,2,:,-1].reshape(ny-2*padSingleSide,1)
			outputP[j,0,padSingleSide:-padSingleSide,0:padSingleSide]=output[j,2,:,0].reshape(ny-2*padSingleSide,1)
			outputP[j,0,0,0]=1*(outputP[j,0,0,1])
			outputP[j,0,0,-1]=1*(outputP[j,0,0,-2])


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


		with torch.no_grad():
			Mag_pred = torch.sqrt(outputU ** 2 + outputV ** 2)

			l2_error_u = torch.sqrt(torch.mean((outputU - U_ref) ** 2)) / torch.sqrt(
			 torch.mean(U_ref ** 2))
			l2_error_v = torch.sqrt(torch.mean((outputV - V_ref) ** 2)) / torch.sqrt(
			 torch.mean(V_ref ** 2))
			l2_error_p = torch.sqrt(torch.mean((outputP - P_ref) ** 2)) / torch.sqrt(
			 torch.mean(P_ref ** 2))
			l2_error_mag = torch.sqrt(torch.mean((Mag_pred - Mag_ref) ** 2)) / torch.sqrt(
			 torch.mean(Mag_ref ** 2))
			l2_error_av = (l2_error_u + l2_error_v + l2_error_p + l2_error_mag) / 4.0


		loss.backward()
		optimizer.step()


	if epoch%100 ==0:
		print(f"[Epoch {epoch:4d}] Loss = {loss.item():.2e}")
		print(
		 f"L2 Errors: U={l2_error_u.item():.4f}, V={l2_error_v.item():.4f}, P={l2_error_p.item():.4f}, Mag={l2_error_mag.item():.4f}, AV={l2_error_av.item():.4f}")

	if epoch%5000==0 or epoch%nEpochs==0:

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

		fig1_path = save_dir / f"epoch_{epoch:04d}_VelMagAndPressureCNN.pdf"
		fig0.savefig(fig1_path, bbox_inches='tight')
		plt.close(fig0)
	return loss.item(), l2_error_u.item(), l2_error_v.item(), l2_error_p.item(), l2_error_mag.item(), l2_error_av.item()


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


			data_path = save_dir / f"data_epoch_{ep}.csv"
			np.savetxt(data_path,
			     np.column_stack([losses, errors_u, errors_v, errors_p, errors_mag,errors_av]),
			     delimiter=',',
			     header='loss,err_u,err_v,err_p,err_mag, err_av')


	dt = time.time() - t0
	print(f"Total training time: {dt:.2f} seconds")


	fina_data_path = save_dir / f"final_data.csv"
	np.savetxt(fina_data_path,
	     np.column_stack([losses, errors_u, errors_v, errors_p, errors_mag, errors_av]),
	     delimiter=',',
	     header='loss,err_u,err_v,err_p,err_mag, err_av')


	time_path = save_dir / f"time.txt"
	np.savetxt(time_path, [dt])


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
	convergence_path = save_dir / f"training_convergence.png"

	plt.savefig(convergence_path, dpi=300)
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
	print_min("U", u_arr[mask], mask_epochs)
	print_min("V", v_arr[mask], mask_epochs)
	print_min("P", p_arr[mask], mask_epochs)
	print_min("Mag", mag_arr[mask], mask_epochs)
	print_min("AV", av_arr[mask], mask_epochs)


	min_errors_summary_path = save_dir / f"min_errors_summary.txt"
	with open(min_errors_summary_path, 'w', encoding='utf-8') as f:
		f.write('—— minimum errors for all epochs ——\n')
		for name, arr in [('U', u_arr), ('V', v_arr), ('P', p_arr), ('Mag', mag_arr), ('AV', av_arr)]:
			idx = arr.argmin()
			f.write(f"{name} minimum L2 error：{arr[idx]:.4f}，at epoch {epochs[idx]}\n")

		f.write('\n—— minimum errors for epochs divisible by 100 ——\n')
		mask = (epochs % 100 == 0)
		mask_epochs = epochs[mask]
		for name, arr in [('U', u_arr), ('V', v_arr), ('P', p_arr), ('Mag', mag_arr), ('AV', av_arr)]:
			sub = arr[mask]
			idx = sub.argmin()
			f.write(f"{name} minimum L2 error：{sub[idx]:.4f}，at epoch {mask_epochs[idx]}\n")

	print(f"Saved minimum error results to {min_errors_summary_path}")

	print("Training completed!")


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


