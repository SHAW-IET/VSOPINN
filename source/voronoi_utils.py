import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import griddata
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from math import ceil, sqrt
from torch import isnan, isinf
import pdb


class VoronoiEnhancedPINN(nn.Module):
    def __init__(self, h, nx, ny, key_points, nVarOut=3):
        super().__init__()
        self.key_points = key_points
        self.nx, self.ny = nx, ny


        self.voronoi_branch = nn.Sequential(
            nn.Conv2d(4, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU()
        )


        self.physics_branch = nn.Sequential(
            nn.Conv2d(2, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU()
        )


        self.fusion = nn.Sequential(
            nn.Conv2d(128, 128, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, nVarOut, 5, padding=2)
        )


class LearnableKeyPointsOLD(nn.Module):
    def __init__(self, initial_positions):
        super().__init__()

        self.positions = nn.Parameter(initial_positions.clone())

    def forward(self, grid_size):


        return self.positions

class LearnableKeyPointsOLD2(nn.Module):
\
\
\

    def __init__(self, initial_positions):
        super().__init__()
        eps = 1e-6
        self.raw = nn.Parameter(
            torch.logit(initial_positions.clamp(eps, 1 - eps))
        )

    @property
    def positions(self):

        return torch.sigmoid(self.raw)

    def forward(self, grid_size):
        ny, nx = grid_size
        xy_cont = self.positions * torch.tensor(
            [ny - 1, nx - 1], device=self.raw.device
        )
        xy_disc = torch.round(xy_cont)
        xy_disc = torch.clamp(
            xy_disc,
            min=torch.tensor([0, 0], device=self.raw.device),
            max=torch.tensor([ny - 1, nx - 1], device=self.raw.device),
        )

        return xy_disc + (xy_cont - xy_cont.detach())

class LearnableKeyPointsOLD3(nn.Module):
\
\
\
\
\

    def __init__(self, initial_positions):
        super().__init__()
        eps = 1e-7
        init = torch.clamp(initial_positions, eps, 1 - eps)
        self.raw = nn.Parameter(torch.log(init / (1 - init)))


    def get_normalized(self):
        return torch.sigmoid(self.raw)


    def forward(self, grid_size):
        ny, nx = grid_size
        pos_norm = self.get_normalized()
        idx_cont = pos_norm * pos_norm.new_tensor([ny - 1, nx - 1])
        idx_round = torch.round(idx_cont)
        idx_ste   = idx_cont + (idx_round - idx_cont).detach()

        idx_ste[..., 0].clamp_(0, ny - 1)
        idx_ste[..., 1].clamp_(0, nx - 1)
        return idx_ste


class LearnableKeyPoints(nn.Module):
    def __init__(self, initial_positions):
        super().__init__()
        eps = 1e-6
        init = initial_positions.clamp(eps, 1 - eps)
        self.raw = nn.Parameter(torch.logit(init))

    def get_normalized(self):
        eps = 1e-6
        return torch.sigmoid(self.raw).clamp(eps, 1 - eps)

    def forward(self, grid_size):
        ny, nx = grid_size
        idx_float = self.get_normalized().clone()


        idx_ste = idx_float.clone()
        idx_ste[:, 0].mul_(ny-1).round_()
        idx_ste[:, 1].mul_(nx-1).round_()
        idx_ste[:, 0].clamp_(0, ny - 1)
        idx_ste[:, 1].clamp_(0, nx - 1)
        return idx_ste + (idx_float - idx_ste).detach()


class VaryGeoDataset_PairedSolutionOld(Dataset):


    def __init__(self, MeshList, SolutionList, key_indices):
        self.MeshList = MeshList
        self.SolutionList = SolutionList
        self.key_indices = key_indices

    def __len__(self):
        return len(self.MeshList)

    def __getitem__(self, idx):

        mesh = self.MeshList[idx]
        x = mesh.x
        y = mesh.y
        xi = mesh.xi
        eta = mesh.eta
        J = mesh.J_ho
        Jinv = mesh.Jinv_ho
        dxdxi = mesh.dxdxi_ho
        dydxi = mesh.dydxi_ho
        dxdeta = mesh.dxdeta_ho
        dydeta = mesh.dydeta_ho
        cord = np.zeros([2, x.shape[0], x.shape[1]])
        cord[0, :, :] = x;
        cord[1, :, :] = y
        InvariantInput = np.zeros([2, J.shape[0], J.shape[1]])
        InvariantInput[0, :, :] = J
        InvariantInput[1, :, :] = Jinv


        sol = self.SolutionList[idx]
        if sol.shape[0] == 3:
            sol = np.transpose(sol, (1, 2, 0))

        ny, nx = x.shape
        voronoi_input = np.zeros([4, ny, nx])


        mask = np.zeros((ny, nx))
        for (y_idx, x_idx) in self.key_indices:
            mask[y_idx, x_idx] = 1.0
        voronoi_input[3] = mask


        key_points_coord = []
        key_physics = []
        for (y_idx, x_idx) in self.key_indices:

            key_points_coord.append([x[y_idx, x_idx], y[y_idx, x_idx]])

            key_physics.append(sol[y_idx, x_idx])


        grid_points = np.column_stack((x.ravel(), y.ravel()))


        u_vals = [p[0] for p in key_physics]
        u_vor = griddata(key_points_coord, u_vals, grid_points, method='nearest').reshape(ny, nx)

        v_vals = [p[1] for p in key_physics]
        v_vor = griddata(key_points_coord, v_vals, grid_points, method='nearest').reshape(ny, nx)

        p_vals = [p[2] for p in key_physics]
        p_vor = griddata(key_points_coord, p_vals, grid_points, method='nearest').reshape(ny, nx)

        voronoi_input[0] = u_vor
        voronoi_input[1] = v_vor
        voronoi_input[2] = p_vor


        return [InvariantInput, cord, xi, eta, J,
                Jinv, dxdxi, dydxi,
                dxdeta, dydeta,
                self.SolutionList[idx][:, :, 0],
                self.SolutionList[idx][:, :, 1],
                self.SolutionList[idx][:, :, 2],
                voronoi_input]

class VaryGeoDataset_PairedSolution(Dataset):


    def __init__(self, MeshList, SolutionList, key_indices):
        self.MeshList = MeshList
        self.SolutionList = SolutionList


    def __len__(self):
        return len(self.MeshList)

    def __getitem__(self, idx):

        mesh = self.MeshList[idx]
        x = mesh.x
        y = mesh.y
        xi = mesh.xi
        eta = mesh.eta
        J = mesh.J_ho
        Jinv = mesh.Jinv_ho
        dxdxi = mesh.dxdxi_ho
        dydxi = mesh.dydxi_ho
        dxdeta = mesh.dxdeta_ho
        dydeta = mesh.dydeta_ho
        cord = np.zeros([2, x.shape[0], x.shape[1]])
        cord[0, :, :] = x;
        cord[1, :, :] = y
        InvariantInput = np.zeros([2, J.shape[0], J.shape[1]])
        InvariantInput[0, :, :] = J
        InvariantInput[1, :, :] = Jinv


        return [InvariantInput, cord, xi, eta, J,
                Jinv, dxdxi, dydxi,
                dxdeta, dydeta,
                self.SolutionList[idx][:, :, 0],
                self.SolutionList[idx][:, :, 1],
                self.SolutionList[idx][:, :, 2]]


def generate_voronoi_input(coord, sol_u, sol_v, sol_p, key_indices, grid_size):

    ny, nx = grid_size
    voronoi_input = np.zeros((4, ny, nx))


    mask = np.zeros((ny, nx))
    for (y_idx, x_idx) in key_indices:
        if 0 <= y_idx < ny and 0 <= x_idx < nx:
            mask[y_idx, x_idx] = 1.0
    voronoi_input[3] = mask


    points = []
    u_vals, v_vals, p_vals = [], [], []

    for (y_idx, x_idx) in key_indices:
        if 0 <= y_idx < ny and 0 <= x_idx < nx:
            points.append([coord[0, y_idx, x_idx], coord[1, y_idx, x_idx]])
            u_vals.append(sol_u[y_idx, x_idx])
            v_vals.append(sol_v[y_idx, x_idx])
            p_vals.append(sol_p[y_idx, x_idx])

    if not points:
        return torch.zeros(1, 4, ny, nx, dtype=torch.float32).to('cuda')

    grid_points = np.column_stack((coord[0].ravel(), coord[1].ravel()))


    u_vor = griddata(points, u_vals, grid_points, method='nearest').reshape(ny, nx)
    v_vor = griddata(points, v_vals, grid_points, method='nearest').reshape(ny, nx)
    p_vor = griddata(points, p_vals, grid_points, method='nearest').reshape(ny, nx)

    voronoi_input[0] = u_vor
    voronoi_input[1] = v_vor
    voronoi_input[2] = p_vor

    return torch.tensor(voronoi_input, dtype=torch.float32).unsqueeze(0).to('cuda')


def generate_voronoi_input_torch(coord, sol_u, sol_v, sol_p, key_positions, grid_size):
\
\
\
\
\
\

    ny, nx = grid_size
    device = coord.device
    N = key_positions.shape[0]


    grid_x, grid_y = torch.meshgrid(
        torch.arange(nx, device=device, dtype=torch.float),
        torch.arange(ny, device=device, dtype=torch.float),
        indexing='xy'
    )
    grid_points = torch.stack([grid_y, grid_x], dim=-1)


    key_points_grid = key_positions * torch.tensor([[nx - 1, ny - 1]], device=device)


    grid_expanded = grid_points.view(ny, nx, 1, 2)
    key_expanded = key_points_grid.view(1, 1, N, 2)


    distances = torch.norm(grid_expanded - key_expanded, dim=-1)


    _, nearest_indices = torch.min(distances, dim=2)


    mask = torch.zeros(ny, nx, device=device)

    rounded_indices = torch.round(key_points_grid).long()
    for i in range(N):
        y, x = rounded_indices[i]
        if 0 <= x < nx and 0 <= y < ny:
            mask[y, x] = 1.0


    key_values = torch.zeros(N, 3, device=device)
    for i in range(N):
        y, x = rounded_indices[i]
        if 0 <= x < nx and 0 <= y < ny:
            key_values[i, 0] = sol_u[y, x]
            key_values[i, 1] = sol_v[y, x]
            key_values[i, 2] = sol_p[y, x]


    voronoi_uvp = key_values[nearest_indices]


    u_vor = voronoi_uvp[..., 0]
    v_vor = voronoi_uvp[..., 1]
    p_vor = voronoi_uvp[..., 2]


    voronoi_input = torch.stack([u_vor, v_vor, p_vor, mask], dim=0)

    return voronoi_input.unsqueeze(0)

def softVoronoi(coord, sol_u, sol_v, sol_p, key_pos, grid_size, alpha):
    ny, nx = grid_size
    device = coord.device
    N = key_pos.shape[0]


    def dbg(name, t):
        t.register_hook(lambda g: print(f"[grad] {name}: finite={torch.isfinite(g).all()}, "
                                        f"max={g.abs().max().item():.2e}"))

    gy, gx = torch.meshgrid(
        torch.linspace(0, ny-1, ny, device=device),
        torch.linspace(0, nx-1, nx, device=device),
        indexing='ij')
    grid = torch.stack([gy, gx], -1).unsqueeze(2)


    key = key_pos * key_pos.new_tensor([ny-1, nx-1])
    key = key.view(1,1,N,2)

    d2 = ((grid - key) ** 2).sum(-1)
    diag2 = (ny - 1) ** 2 + (nx - 1) ** 2
    d2_norm = d2 / (diag2 + 1e-12)
    w  = torch.softmax(-alpha * d2_norm, dim=-1)


    grid_pts = torch.stack(
        [key_pos[:, 1] * 2 - 1,
         key_pos[:, 0] * 2 - 1], dim=-1
    ).view(1, -1, 1, 2)


    key_u = F.grid_sample(sol_u.unsqueeze(0).unsqueeze(0), grid_pts,
                          align_corners=True).squeeze()
    key_v = F.grid_sample(sol_v.unsqueeze(0).unsqueeze(0), grid_pts,
                          align_corners=True).squeeze()
    key_p = F.grid_sample(sol_p.unsqueeze(0).unsqueeze(0), grid_pts,
                          align_corners=True).squeeze()


    u_vor = (w * key_u.view(1, 1, -1)).sum(-1)
    v_vor = (w * key_v.view(1, 1, -1)).sum(-1)
    p_vor = (w * key_p.view(1, 1, -1)).sum(-1)


    d_min = torch.sqrt(d2)
    d_min = d_min.min(-1).values
    β = 5.0
    mask = torch.exp(-β * d_min / d_min.max())


    vor = torch.stack([u_vor, v_vor, p_vor, mask], 0)


    return vor.unsqueeze(0)


class VoronoiEnhancedUSCNN(nn.Module):
    def __init__(self, h, nx, ny, nVarIn=2, nVarOut=3, initWay=None, k=5, s=1, p=2):
        super(VoronoiEnhancedUSCNN, self).__init__()

        self.initWay = initWay
        self.nVarIn = nVarIn
        self.nVarOut = nVarOut
        self.k = k
        self.s = 1
        self.p = 2
        self.deltaX = h
        self.nx = nx
        self.ny = ny


        self.voronoi_branch = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=k, stride=s, padding=p),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=k, stride=s, padding=p),
            nn.ReLU()
        )


        self.coord_branch = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=k, stride=s, padding=p),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=k, stride=s, padding=p),
            nn.ReLU()
        )


        self.fusion_branch = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=k, stride=s, padding=p),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=k, stride=s, padding=p),
            nn.ReLU(),
            nn.Conv2d(32, nVarOut, kernel_size=k, stride=s, padding=p)
        )


        self.US = nn.Upsample(size=[self.ny - 2, self.nx - 2], mode='bicubic')
        self.pixel_shuffle = nn.PixelShuffle(1)


        dxFilter = torch.Tensor([[[[0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0.],
                                   [1., -8., 0., 8., -1.],
                                   [0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0.]]]]).to("cuda") / 12. / self.deltaX
        self.convdx = nn.Conv2d(1, 1, (5, 5), stride=1, padding=0, bias=None)
        self.convdx.weight = nn.Parameter(dxFilter, requires_grad=False)

        dyFilter = torch.Tensor([[[[0., 0., 1., 0., 0.],
                                   [0., 0., -8., 0., 0.],
                                   [0., 0., 0., 0., 0.],
                                   [0., 0., 8., 0., 0.],
                                   [0., 0., -1., 0., 0.]]]]).to("cuda") / 12. / self.deltaX
        self.convdy = nn.Conv2d(1, 1, (5, 5), stride=1, padding=0, bias=None)
        self.convdy.weight = nn.Parameter(dyFilter, requires_grad=False)

        lapFilter = torch.Tensor([[[[0., 0., -1., 0., 0.],
                                    [0., 0., 16., 0., 0.],
                                    [-1., 16., -60., 16., -1.],
                                    [0., 0., 16., 0., 0.],
                                    [0., 0., -1., 0., 0.]]]]).to("cuda") / 12. / self.deltaX / self.deltaX
        self.convlap = nn.Conv2d(1, 1, (5, 5), stride=1, padding=0, bias=None)
        self.convlap.weight = nn.Parameter(lapFilter, requires_grad=False)


    def forward(self, coord, voronoi_input):


        voronoi_feat = self.voronoi_branch(voronoi_input)


        coord_feat = self.coord_branch(coord)


        fused = torch.cat([voronoi_feat, coord_feat], dim=1)
        output = self.fusion_branch(fused)


        output = self.US(output)
        output = self.pixel_shuffle(output)

        return output


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)


        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + x

class VoronoiAttentionUSCNN(nn.Module):
    def __init__(self, h, nx, ny, nVarIn=2, nVarOut=3, initWay=None, k=5, s=1, p=2):
        super(VoronoiAttentionUSCNN, self).__init__()

        self.initWay = initWay
        self.nVarIn = nVarIn
        self.nVarOut = nVarOut
        self.k = k
        self.s = 1
        self.p = 2
        self.deltaX = h
        self.nx = nx
        self.ny = ny


        self.voronoi_branch = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=k, stride=s, padding=p),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=k, stride=s, padding=p),
            nn.ReLU()
        )


        self.coord_branch = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=k, stride=s, padding=p),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=k, stride=s, padding=p),
            nn.ReLU()
        )


        self.fusion_branch = nn.Sequential(

            nn.Conv2d(64, 64, kernel_size=k, stride=s, padding=p),
            nn.ReLU(),


            CBAM(64, ratio=16),


            nn.Conv2d(64, 32, kernel_size=k, stride=s, padding=p),
            nn.ReLU(),
            nn.Conv2d(32, nVarOut, kernel_size=k, stride=s, padding=p)
        )


        self.US = nn.Upsample(size=[self.ny - 2, self.nx - 2], mode='bicubic')
        self.pixel_shuffle = nn.PixelShuffle(1)


        dxFilter = torch.Tensor([[[[0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0.],
                                   [1., -8., 0., 8., -1.],
                                   [0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0.]]]]).to("cuda") / 12. / self.deltaX
        self.convdx = nn.Conv2d(1, 1, (5, 5), stride=1, padding=0, bias=None)
        self.convdx.weight = nn.Parameter(dxFilter, requires_grad=False)

        dyFilter = torch.Tensor([[[[0., 0., 1., 0., 0.],
                                   [0., 0., -8., 0., 0.],
                                   [0., 0., 0., 0., 0.],
                                   [0., 0., 8., 0., 0.],
                                   [0., 0., -1., 0., 0.]]]]).to("cuda") / 12. / self.deltaX
        self.convdy = nn.Conv2d(1, 1, (5, 5), stride=1, padding=0, bias=None)
        self.convdy.weight = nn.Parameter(dyFilter, requires_grad=False)

        lapFilter = torch.Tensor([[[[0., 0., -1., 0., 0.],
                                    [0., 0., 16., 0., 0.],
                                    [-1., 16., -60., 16., -1.],
                                    [0., 0., 16., 0., 0.],
                                    [0., 0., -1., 0., 0.]]]]).to("cuda") / 12. / self.deltaX / self.deltaX
        self.convlap = nn.Conv2d(1, 1, (5, 5), stride=1, padding=0, bias=None)
        self.convlap.weight = nn.Parameter(lapFilter, requires_grad=False)


    def forward(self, coord, voronoi_input):


        voronoi_feat = self.voronoi_branch(voronoi_input)


        coord_feat = self.coord_branch(coord)


        fused = torch.cat([voronoi_feat, coord_feat], dim=1)
        output = self.fusion_branch(fused)


        output = self.US(output)
        output = self.pixel_shuffle(output)

        return output

class VoronoiAttention2USCNN(nn.Module):
    def __init__(self, h, nx, ny, nVarIn=2, nVarOut=3, initWay=None, k=5, s=1, p=2):
        super(VoronoiAttention2USCNN, self).__init__()

        self.initWay = initWay
        self.nVarIn = nVarIn
        self.nVarOut = nVarOut
        self.k = k
        self.s = 1
        self.p = 2
        self.deltaX = h
        self.nx = nx
        self.ny = ny


        self.voronoi_branch = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=k, stride=s, padding=p),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=k, stride=s, padding=p),
            nn.ReLU()
        )


        self.coord_branch = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=k, stride=s, padding=p),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=k, stride=s, padding=p),
            nn.ReLU()
        )


        self.fusion_branch = nn.Sequential(
            CBAM(64, ratio=16),

            nn.Conv2d(64, 64, kernel_size=k, stride=s, padding=p),
            nn.ReLU(),


            nn.Conv2d(64, 32, kernel_size=k, stride=s, padding=p),
            nn.ReLU(),
            nn.Conv2d(32, nVarOut, kernel_size=k, stride=s, padding=p)
        )


        self.US = nn.Upsample(size=[self.ny - 2, self.nx - 2], mode='bicubic')
        self.pixel_shuffle = nn.PixelShuffle(1)


        dxFilter = torch.Tensor([[[[0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0.],
                                   [1., -8., 0., 8., -1.],
                                   [0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0.]]]]).to("cuda") / 12. / self.deltaX
        self.convdx = nn.Conv2d(1, 1, (5, 5), stride=1, padding=0, bias=None)
        self.convdx.weight = nn.Parameter(dxFilter, requires_grad=False)

        dyFilter = torch.Tensor([[[[0., 0., 1., 0., 0.],
                                   [0., 0., -8., 0., 0.],
                                   [0., 0., 0., 0., 0.],
                                   [0., 0., 8., 0., 0.],
                                   [0., 0., -1., 0., 0.]]]]).to("cuda") / 12. / self.deltaX
        self.convdy = nn.Conv2d(1, 1, (5, 5), stride=1, padding=0, bias=None)
        self.convdy.weight = nn.Parameter(dyFilter, requires_grad=False)

        lapFilter = torch.Tensor([[[[0., 0., -1., 0., 0.],
                                    [0., 0., 16., 0., 0.],
                                    [-1., 16., -60., 16., -1.],
                                    [0., 0., 16., 0., 0.],
                                    [0., 0., -1., 0., 0.]]]]).to("cuda") / 12. / self.deltaX / self.deltaX
        self.convlap = nn.Conv2d(1, 1, (5, 5), stride=1, padding=0, bias=None)
        self.convlap.weight = nn.Parameter(lapFilter, requires_grad=False)


    def forward(self, coord, voronoi_input):


        voronoi_feat = self.voronoi_branch(voronoi_input)


        coord_feat = self.coord_branch(coord)


        fused = torch.cat([voronoi_feat, coord_feat], dim=1)
        output = self.fusion_branch(fused)


        output = self.US(output)
        output = self.pixel_shuffle(output)

        return output


class VoronoiMultiUSCNN(nn.Module):
    def __init__(self, h, nx, ny, nVarIn=2, nVarOut=3, initWay=None, k=5, s=1, p=2, num_conditions=1):
        super(VoronoiMultiUSCNN, self).__init__()

        self.initWay = initWay
        self.nVarIn = nVarIn
        self.nVarOut = nVarOut
        self.k = k
        self.s = 1
        self.p = 2
        self.deltaX = h
        self.nx = nx
        self.ny = ny
        self.num_conditions = num_conditions


        self.voronoi_branch = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=k, stride=s, padding=p),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=k, stride=s, padding=p),
            nn.ReLU()
        )


        self.coord_branch = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=k, stride=s, padding=p),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=k, stride=s, padding=p),
            nn.ReLU()
        )


        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=k, stride=s, padding=p),
                nn.ReLU(),
                nn.Conv2d(64, 32, kernel_size=k, stride=s, padding=p),
                nn.ReLU(),
                nn.Conv2d(32, nVarOut, kernel_size=k, stride=s, padding=p)
            ) for _ in range(num_conditions)
        ])


        self.US = nn.Upsample(size=[self.ny - 2, self.nx - 2], mode='bicubic')
        self.pixel_shuffle = nn.PixelShuffle(1)


        dxFilter = torch.Tensor([[[[0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0.],
                                   [1., -8., 0., 8., -1.],
                                   [0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0.]]]]).to("cuda") / 12. / self.deltaX
        self.convdx = nn.Conv2d(1, 1, (5, 5), stride=1, padding=0, bias=None)
        self.convdx.weight = nn.Parameter(dxFilter, requires_grad=False)

        dyFilter = torch.Tensor([[[[0., 0., 1., 0., 0.],
                                   [0., 0., -8., 0., 0.],
                                   [0., 0., 0., 0., 0.],
                                   [0., 0., 8., 0., 0.],
                                   [0., 0., -1., 0., 0.]]]]).to("cuda") / 12. / self.deltaX
        self.convdy = nn.Conv2d(1, 1, (5, 5), stride=1, padding=0, bias=None)
        self.convdy.weight = nn.Parameter(dyFilter, requires_grad=False)


        lapFilter = torch.Tensor([[[[0., 0., -1., 0., 0.],
                                    [0., 0., 16., 0., 0.],
                                    [-1., 16., -60., 16., -1.],
                                    [0., 0., 16., 0., 0.],
                                    [0., 0., -1., 0., 0.]]]]).to("cuda") / 12. / self.deltaX / self.deltaX
        self.convlap = nn.Conv2d(1, 1, (5, 5), stride=1, padding=0, bias=None)
        self.convlap.weight = nn.Parameter(lapFilter, requires_grad=False)

    def forward(self, coord, voronoi_input, condition_idx=0):
\
\
\


        voronoi_feat = self.voronoi_branch(voronoi_input)
        coord_feat = self.coord_branch(coord)


        fused = torch.cat([voronoi_feat, coord_feat], dim=1)


        if not (0 <= condition_idx < self.num_conditions):
            raise ValueError(f"condition_idx {condition_idx} out of range (0-{self.num_conditions - 1})")

        output = self.decoders[condition_idx](fused)


        output = self.US(output)
        output = self.pixel_shuffle(output)

        return output


def compute_cvt_update(current_sensors, density_map, device='cuda'):
\
\
\
\
\
\
\

    H, W = density_map.shape
    K = current_sensors.shape[0]


    y_vals = torch.linspace(-1, 1, H, device=device)
    x_vals = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(y_vals, x_vals, indexing='ij')


    grid_coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    density_flat = density_map.flatten() + 1e-8


    dists = torch.cdist(grid_coords, current_sensors)
    labels = torch.argmin(dists, dim=1)


    weighted_coords_sum = torch.zeros((K, 2), device=device)
    weight_sum = torch.zeros((K, 1), device=device)


    weighted_coords = grid_coords * density_flat.unsqueeze(1)


    labels_expanded = labels.unsqueeze(1).expand(-1, 2)
    weighted_coords_sum.scatter_add_(0, labels_expanded, weighted_coords)
    weight_sum.scatter_add_(0, labels.unsqueeze(1), density_flat.unsqueeze(1))


    new_sensors = weighted_coords_sum / weight_sum


    new_sensors = torch.clamp(new_sensors, -1.0, 1.0)

    return new_sensors


def to4DTensor(myList, device='cuda'):
    out = []
    for item in myList:

        if isinstance(item, torch.Tensor):
            t = item.to(device=device, dtype=torch.float32)
        elif isinstance(item, np.ndarray):
            t = torch.from_numpy(item).to(device=device, dtype=torch.float32)
        else:
            t = torch.as_tensor(item, dtype=torch.float32, device=device)


        if t.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.ndim == 3:
            t = t.unsqueeze(0)
        elif t.ndim == 4:
            pass
        else:

            raise ValueError(f"Unsupported ndim {t.ndim} for item in to4DTensor")

        out.append(t)
    return out


def _ranges(ny, nx, pad=1, include_boundary=False):
    y0 = 0 if include_boundary else pad
    y1 = ny-1 if include_boundary else ny-1-pad
    x0 = 0 if include_boundary else pad
    x1 = nx-1 if include_boundary else nx-1-pad
    return y0, y1, x0, x1

def select_key_indices(mode="grid", N=64, ny=None, nx=None, pad=1,
                       include_boundary=False, seed=42,
                       n_theta=None, n_r=None, min_sep=None):
\
\
\
\
\
\
\
\
\
\

    assert ny is not None and nx is not None
    y0, y1, x0, x1 = _ranges(ny, nx, pad, include_boundary)
    H, W = (y1 - y0 + 1), (x1 - x0 + 1)

    def _dedup_round(pairs):

        S = set()
        out = []
        for y, x in pairs:
            iy = int(round(y))
            ix = int(round(x))
            if y0 <= iy <= y1 and x0 <= ix <= x1 and (iy, ix) not in S:
                S.add((iy, ix)); out.append((iy, ix))
        return out

    if mode == "grid":

        if n_theta is None or n_r is None:
            n_theta = max(1, int(round(sqrt(N * H / W))))
            n_r     = max(1, int(ceil(N / n_theta)))
        yy = np.linspace(y0, y1, n_theta)
        xx = np.linspace(x0, x1, n_r)
        pairs = [(y, x) for y in yy for x in xx]
        return _dedup_round(pairs)[:N]

    elif mode == "random":
        rng = np.random.default_rng(seed)
        ys = rng.integers(y0, y1+1, size=10*N)
        xs = rng.integers(x0, x1+1, size=10*N)
        pairs = list(zip(ys, xs))
        pairs = _dedup_round(pairs)

        rng.shuffle(pairs)
        return pairs[:N]

    elif mode == "halton":

        def van_der_corput(n, base):
            vdc, denom = 0.0, 1.0
            while n:
                n, rem = divmod(n, base)
                denom *= base
                vdc += rem / denom
            return vdc
        pts = []
        i = 1
        while len(pts) < N*3:
            u = van_der_corput(i, 2)
            v = van_der_corput(i, 3)
            y = y0 + u * (y1 - y0)
            x = x0 + v * (x1 - x0)
            pts.append((y, x))
            i += 1
        pairs = _dedup_round(pts)
        return pairs[:N]

    elif mode == "fps":


        grid_y, grid_x = np.mgrid[y0:y1+1, x0:x1+1]
        candidates = np.column_stack([grid_y.ravel(), grid_x.ravel()]).astype(float)


        rng = np.random.default_rng(seed)
        start_idx = rng.integers(0, len(candidates))
        chosen = [candidates[start_idx]]

        for _ in range(1, N*3):
            d2 = np.full(len(candidates), np.inf)
            for c in chosen:
                dy = candidates[:, 0] - c[0]
                dx = candidates[:, 1] - c[1]
                d2 = np.minimum(d2, dy*dy + dx*dx)
            j = int(np.argmax(d2))
            if min_sep is not None and sqrt(d2[j]) < min_sep:
                break
            chosen.append(candidates[j])
            if len(chosen) >= N:
                break
        return _dedup_round(chosen)[:N]

    elif mode == "boundary":

        Ni = max(1, int(round(N * 0.4)))
        No = max(1, int(round(N * 0.4)))
        Nb = max(1, N - Ni - No)


        ys = np.linspace(y0, y1, max(Ni, No), dtype=int)
        inner = [(iy, x0) for iy in ys[:Ni]]
        outer = [(iy, x1) for iy in ys[:No]]


        n_theta = max(1, int(round(sqrt(Nb * H / W))))
        n_r     = max(1, int(ceil(Nb / n_theta)))
        yy = np.linspace(y0, y1, n_theta, dtype=int)
        xx = np.linspace(x0, x1, n_r, dtype=int)
        bulk = []
        for y in yy:
            for x in xx:
                bulk.append((y, x))
                if len(bulk) >= Nb:
                    break
            if len(bulk) >= Nb:
                break

        pairs = inner + outer + bulk
        pairs = list(dict.fromkeys(pairs))
        return pairs[:N]

    else:
        raise ValueError(f"Unknown mode: {mode!r}")


def softVoronoi(coord, sol_u, sol_v, sol_p, key_pos, grid_size, alpha):
    ny, nx = grid_size
    device = coord.device
    N = key_pos.shape[0]


    def dbg(name, t):
        t.register_hook(lambda g: print(f"[grad] {name}: finite={torch.isfinite(g).all()}, "
                                        f"max={g.abs().max().item():.2e}"))

    gy, gx = torch.meshgrid(
        torch.linspace(0, ny-1, ny, device=device),
        torch.linspace(0, nx-1, nx, device=device),
        indexing='ij')
    grid = torch.stack([gy, gx], -1).unsqueeze(2)

    key = key_pos * key_pos.new_tensor([ny-1, nx-1])
    key = key.view(1,1,N,2)

    d2 = ((grid - key) ** 2).sum(-1)
    diag2 = (ny - 1) ** 2 + (nx - 1) ** 2
    d2_norm = d2 / (diag2 + 1e-12)
    w  = torch.softmax(-alpha * d2_norm, dim=-1)


    grid_pts = torch.stack(
        [key_pos[:, 1] * 2 - 1,
         key_pos[:, 0] * 2 - 1], dim=-1
    ).view(1, -1, 1, 2)


    key_u = F.grid_sample(sol_u.unsqueeze(0).unsqueeze(0), grid_pts,
                          align_corners=True).squeeze()
    key_v = F.grid_sample(sol_v.unsqueeze(0).unsqueeze(0), grid_pts,
                          align_corners=True).squeeze()
    key_p = F.grid_sample(sol_p.unsqueeze(0).unsqueeze(0), grid_pts,
                          align_corners=True).squeeze()


    u_vor = (w * key_u.view(1, 1, -1)).sum(-1)
    v_vor = (w * key_v.view(1, 1, -1)).sum(-1)
    p_vor = (w * key_p.view(1, 1, -1)).sum(-1)

    d_min = torch.sqrt(d2)
    d_min = d_min.min(-1).values
    β = 5.0
    mask = torch.exp(-β * d_min / d_min.max())
    vor = torch.stack([u_vor, v_vor, p_vor, mask], 0)
    return vor.unsqueeze(0)
