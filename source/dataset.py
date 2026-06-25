from torch.utils.data import Dataset, DataLoader
import pdb
import numpy as np
from scipy.interpolate import griddata
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class VaryGeoDataset(Dataset):

	def __init__(self,MeshList):
		self.MeshList=MeshList
	def __len__(self):
		return len(self.MeshList)
	def __getitem__(self,idx):
		mesh=self.MeshList[idx]
		x=mesh.x
		y=mesh.y
		xi=mesh.xi
		eta=mesh.eta
		J=mesh.J_ho
		Jinv=mesh.Jinv_ho
		dxdxi=mesh.dxdxi_ho
		dydxi=mesh.dydxi_ho
		dxdeta=mesh.dxdeta_ho
		dydeta=mesh.dydeta_ho
		cord=np.zeros([2,x.shape[0],x.shape[1]])
		cord[0,:,:]=x; cord[1,:,:]=y
		InvariantInput=np.zeros([2,J.shape[0],J.shape[1]])
		InvariantInput[0,:,:]=J
		InvariantInput[1,:,:]=Jinv
		return [InvariantInput,cord,xi,eta,J,
		        Jinv,dxdxi,dydxi,
		        dxdeta,dydeta]


class FixGeoDataset(Dataset):

	def __init__(self,ParaList,mesh,OFSolutionList):
		self.ParaList=ParaList
		self.mesh=mesh
		self.OFSolutionList=OFSolutionList
	def __len__(self):
		return len(self.ParaList)
	def __getitem__(self,idx):
		mesh=self.mesh
		x=mesh.x
		y=mesh.y
		xi=mesh.xi
		eta=mesh.eta
		J=mesh.J_ho
		Jinv=mesh.Jinv_ho
		dxdxi=mesh.dxdxi_ho
		dydxi=mesh.dydxi_ho
		dxdeta=mesh.dxdeta_ho
		dydeta=mesh.dydeta_ho
		cord=np.zeros([2,x.shape[0],x.shape[1]])
		cord[0,:,:]=x; cord[1,:,:]=y


		ParaStart=np.ones(x.shape[0])*self.ParaList[idx]
		ParaEnd=np.zeros(x.shape[0])
		Para=np.linspace(ParaStart,ParaEnd,x.shape[1]).T
		return [Para,cord,xi,eta,J,
		        Jinv,dxdxi,dydxi,
		        dxdeta,dydeta,self.OFSolutionList[idx]]

class FixGeoDatasetVoronoi(Dataset):

	def __init__(self,ParaList,mesh,OFSolutionList, key_indices):
		self.ParaList=ParaList
		self.mesh=mesh
		self.OFSolutionList=OFSolutionList
		self.key_indices = key_indices
	def __len__(self):
		return len(self.ParaList)
	def __getitem__(self,idx):
		mesh=self.mesh
		x=mesh.x
		y=mesh.y
		xi=mesh.xi
		eta=mesh.eta
		J=mesh.J_ho
		Jinv=mesh.Jinv_ho
		dxdxi=mesh.dxdxi_ho
		dydxi=mesh.dydxi_ho
		dxdeta=mesh.dxdeta_ho
		dydeta=mesh.dydeta_ho
		cord=np.zeros([2,x.shape[0],x.shape[1]])
		cord[0,:,:]=x; cord[1,:,:]=y


		ParaStart=np.ones(x.shape[0])*self.ParaList[idx]
		ParaEnd=np.zeros(x.shape[0])
		Para=np.linspace(ParaStart,ParaEnd,x.shape[1]).T


		sol = self.OFSolutionList[idx]
		if sol.shape[0] == 3:
			sol = np.transpose(sol, (1, 2, 0))

		ny, nx = x.shape
		voronoi_input = np.zeros([2, ny, nx])


		mask = np.zeros((ny, nx))
		for (y_idx, x_idx) in self.key_indices:
			mask[y_idx, x_idx] = 1.0
		voronoi_input[1] = mask


		key_points_coord = []
		key_physics = []
		for (y_idx, x_idx) in self.key_indices:

			key_points_coord.append([x[y_idx, x_idx], y[y_idx, x_idx]])

			key_physics.append(sol[y_idx, x_idx])


		grid_points = np.column_stack((x.ravel(), y.ravel()))


		T_vals = key_physics
		T_vor = griddata(key_points_coord, T_vals, grid_points, method='nearest').reshape(ny, nx)

		voronoi_input[0] = T_vor


		return [Para,cord,xi,eta,J,
		        Jinv,dxdxi,dydxi,
		        dxdeta,dydeta,self.OFSolutionList[idx],
		  voronoi_input]

class VaryGeoDataset_PairedSolution(Dataset):

	def __init__(self,MeshList,SolutionList):
		self.MeshList=MeshList
		self.SolutionList=SolutionList
	def __len__(self):
		return len(self.MeshList)
	def __getitem__(self,idx):
		mesh=self.MeshList[idx]
		x=mesh.x
		y=mesh.y
		xi=mesh.xi
		eta=mesh.eta
		J=mesh.J_ho
		Jinv=mesh.Jinv_ho
		dxdxi=mesh.dxdxi_ho
		dydxi=mesh.dydxi_ho
		dxdeta=mesh.dxdeta_ho
		dydeta=mesh.dydeta_ho
		cord=np.zeros([2,x.shape[0],x.shape[1]])
		cord[0,:,:]=x; cord[1,:,:]=y
		InvariantInput=np.zeros([2,J.shape[0],J.shape[1]])
		InvariantInput[0,:,:]=J
		InvariantInput[1,:,:]=Jinv
		return [InvariantInput,cord,xi,eta,J,
		        Jinv,dxdxi,dydxi,
		        dxdeta,dydeta,
		  self.SolutionList[idx][:,:,0],
		  self.SolutionList[idx][:,:,1],
		  self.SolutionList[idx][:,:,2]]
