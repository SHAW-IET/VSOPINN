import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pdb
from torch.utils.data import DataLoader
import time
from voronoi_utils import VaryGeoDataset_PairedSolutionOld, to4DTensor, LearnableKeyPoints, VaryGeoDataset_PairedSolution, softVoronoi, select_key_indices, compute_cvt_update
from view import view_cfd_fields, view_cfd_fields_physical, view_compare_uvp_mag_physical
from scipy.interpolate import interp1d
import tikzplotlib

sys.path.insert(0, '../source')
from dataset import VaryGeoDataset
from pyMesh import hcubeMesh, visualize2D, plotBC, plotMesh, setAxisLabel,\
    np2cuda
from model import USCNN, USCNNSepPhi, USCNNSep, DDBasic, VoronoiEnhancedUSCNNSep
from readOF import convertOFMeshToImage, convertOFMeshToImage_StructuredMesh
from sklearn.metrics import mean_squared_error as calMSE
import Ofpp
from pathlib import Path


save_dir = Path(r"./output/improved_1.28_data_records/5 optimized_sensors_CVT")
save_dir.mkdir(parents=True, exist_ok=True)


h=0.01
r=0.5
R=1
dtheta=0

OFBCCoord = Ofpp.parse_boundary_field('discRe350_0.01nu/0/C')

OFLEFTC=OFBCCoord[b'left'][b'value']
OFRIGHTC=OFBCCoord[b'right'][b'value']
leftX=r*np.cos(np.linspace(dtheta,2*np.pi-dtheta,276))
leftY=r*np.sin(np.linspace(dtheta,2*np.pi-dtheta,276))
rightX=R*np.cos(np.linspace(dtheta,2*np.pi-dtheta,276))
rightY=R*np.sin(np.linspace(dtheta,2*np.pi-dtheta,276))
lowX=np.linspace(leftX[0],rightX[0],49);
lowY=lowX*0+np.sin(dtheta)
upX=np.linspace(leftX[-1],rightX[-1],49);
upY=upX*0-np.sin(dtheta)
ny=len(leftX);nx=len(lowX)
myMesh=hcubeMesh(leftX,leftY,rightX,rightY,
                lowX,lowY,upX,upY,h,True,True,
                tolMesh=1e-10,tolJoint=0.01)


key_indices = select_key_indices("grid", ny=ny, nx=nx, N=12, pad=1)


initial_positions = torch.tensor(
    [(y / (ny - 1), x / (nx - 1)) for y, x in key_indices],
    dtype=torch.float32
)

print("Key point indices:", key_indices)

key_history = []


batchSize = 1
NvarInput = 2
NvarOutput = 1
nEpochs = 30000
lr = 0.001
Ns = 1
Omega = 14
xc = 0
yc = 0


nu = 1e-02

CLIP_NORM = 1.0


model = VoronoiEnhancedUSCNNSep(h, nx, ny, nVorCh=4, initWay='kaiming').to('cuda')
key_points_model = LearnableKeyPoints(initial_positions).to('cuda')

model = model.to('cuda')

criterion = nn.MSELoss()

optimizer = torch.optim.Adam([
    {'params': model.parameters(), 'lr': 1e-3},
    {'params': key_points_model.parameters(), 'lr': 1e-2}
], weight_decay=1e-7)


padSingleSide = 1
udfpad = nn.ConstantPad2d([padSingleSide, padSingleSide, padSingleSide, padSingleSide], 0)

MeshList = []
MeshList.append(myMesh)

OFPic=convertOFMeshToImage_StructuredMesh(nx,ny,'discRe350_0.01nu/0/C',
                                              ['discRe350_0.01nu/2969/U',
                                               'discRe350_0.01nu/2969/p'],
                                               [0,1,0,1],0.0,False)
OFX = OFPic[:, :, 0]
OFY = OFPic[:, :, 1]
OFU = OFPic[:, :, 2]
OFV = OFPic[:, :, 3]
OFP = OFPic[:, :, 4]
OFMag = np.sqrt(OFU ** 2 + OFV ** 2)

OFU_sb = np.zeros(OFU.shape)
OFV_sb = np.zeros(OFV.shape)
OFP_sb = np.zeros(OFP.shape)
OFMag_sb = np.zeros(OFMag.shape)

for i in range(nx):
    for j in range(ny):
        dist = (myMesh.x[j, i] - OFX) ** 2 + (myMesh.y[j, i] - OFY) ** 2
        idx_min = np.where(dist == dist.min())
        OFU_sb[j, i] = OFU[idx_min]
        OFV_sb[j, i] = OFV[idx_min]
        OFP_sb[j, i] = OFP[idx_min]

OFMag_sb = np.sqrt(OFU_sb ** 2 + OFV_sb ** 2)

U_ref = torch.tensor(OFU_sb.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)
V_ref = torch.tensor(OFV_sb.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)
Mag_ref = torch.tensor(OFMag_sb.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)
P_ref = torch.tensor(OFP_sb.copy(), dtype=torch.float32).to('cuda').reshape(1, 1, ny, nx)


P_ref_zm = P_ref - P_ref.mean()

SolutionList = [np.stack([OFU_sb, OFV_sb, OFP_sb], axis=-1)]

train_set = VaryGeoDataset_PairedSolution(MeshList, SolutionList, key_indices)
training_data_loader = DataLoader(dataset=train_set,
                                  batch_size=batchSize)


view_cfd_fields_physical(OFU_sb, OFV_sb, OFP_sb, myMesh, OFMag=OFMag_sb)


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


def apply_annulus_rotating_bc(outputU, outputV, outputP, myMesh, Omega, xc, yc, padSingleSide=1):
\
\

    B, C, H, W = outputU.shape
    assert C == 1

    ii = slice(padSingleSide, W - padSingleSide)
    jj = slice(padSingleSide, H - padSingleSide)

    device = outputU.device
    dtype = outputU.dtype

    X = torch.as_tensor(myMesh.x, dtype=dtype, device=device)
    Y = torch.as_tensor(myMesh.y, dtype=dtype, device=device)


    xL = X[jj, 0:1];
    yL = Y[jj, 0:1]
    xR = X[jj, -1:];
    yR = Y[jj, -1:]


    outputU[:, 0, :padSingleSide, ii] = outputU[:, 0, -padSingleSide - 1:-padSingleSide, ii]
    outputV[:, 0, :padSingleSide, ii] = outputV[:, 0, -padSingleSide - 1:-padSingleSide, ii]
    outputP[:, 0, :padSingleSide, ii] = outputP[:, 0, -padSingleSide - 1:-padSingleSide, ii]


    outputU[:, 0, -padSingleSide:, ii] = outputU[:, 0, padSingleSide:padSingleSide + 1, ii]
    outputV[:, 0, -padSingleSide:, ii] = outputV[:, 0, padSingleSide:padSingleSide + 1, ii]
    outputP[:, 0, -padSingleSide:, ii] = outputP[:, 0, padSingleSide:padSingleSide + 1, ii]


    outputU[:, 0, jj, -padSingleSide:] = 0.0
    outputV[:, 0, jj, -padSingleSide:] = 0.0


    uL = -Omega * (yL - yc)
    vL = Omega * (xL - xc)

    outputU[:, 0, jj, 0:padSingleSide] = uL.unsqueeze(0).expand(B, -1, -1)
    outputV[:, 0, jj, 0:padSingleSide] = vL.unsqueeze(0).expand(B, -1, -1)


    outputP[:, 0] -= outputP[:, 0].mean(dim=(-2, -1), keepdim=True)

    return outputU, outputV, outputP


def train(epoch):
    startTime = time.time()
    loss_data = 0
    eU = 0;
    eV = 0;
    eP = 0

    grid_size = (ny, nx)

    key_positions = key_points_model.get_normalized()

    idx_float = key_points_model(grid_size)

    for iteration, batch in enumerate(training_data_loader):
        [JJInv, coord_tensor, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta,
         sol_u, sol_v, sol_p] = to4DTensor(batch)

        alpha0, alpha_max = 2.0, 80.0
        alpha_t = min(alpha_max, alpha0 * (1 + epoch / 300) ** 2)

        coord_2d = coord_tensor[0]
        sol_u_2d = sol_u[0, 0]
        sol_v_2d = sol_v[0, 0]
        sol_p_2d = sol_p[0, 0]

        voronoi_input = softVoronoi(
            coord_2d, sol_u_2d, sol_v_2d, sol_p_2d,
            key_positions,
            grid_size, alpha=alpha_t)

        optimizer.zero_grad()
        output = model(coord_tensor, voronoi_input)
        output_pad = udfpad(output)

        outputU = output_pad[:, 0, :, :].reshape(output_pad.shape[0], 1,
                                                 output_pad.shape[2],
                                                 output_pad.shape[3])
        outputV = output_pad[:, 1, :, :].reshape(output_pad.shape[0], 1,
                                                 output_pad.shape[2],
                                                 output_pad.shape[3])
        outputP = output_pad[:, 2, :, :].reshape(output_pad.shape[0], 1,
                                                 output_pad.shape[2],
                                                 output_pad.shape[3])


        outputU, outputV, outputP = apply_annulus_rotating_bc(
            outputU, outputV, outputP,
            myMesh=myMesh, Omega=Omega, xc=xc, yc=yc,
            padSingleSide=padSingleSide
        )

        Pm_pred = outputP.mean(dim=(-2, -1), keepdim=True)
        outputP = outputP - Pm_pred


        dudx = dfdx(outputU, dydeta, dydxi, Jinv)
        d2udx2 = dfdx(dudx, dydeta, dydxi, Jinv)
        dudy = dfdy(outputU, dxdxi, dxdeta, Jinv)
        d2udy2 = dfdy(dudy, dxdxi, dxdeta, Jinv)
        dvdx = dfdx(outputV, dydeta, dydxi, Jinv)
        d2vdx2 = dfdx(dvdx, dydeta, dydxi, Jinv)
        dvdy = dfdy(outputV, dxdxi, dxdeta, Jinv)
        d2vdy2 = dfdy(dvdy, dxdxi, dxdeta, Jinv)
        dpdx = dfdx(outputP, dydeta, dydxi, Jinv)
        dpdy = dfdy(outputP, dxdxi, dxdeta, Jinv)


        continuity = dudx + dvdy;
        momentumX = outputU * dudx + outputV * dudy
        forceX = -dpdx + nu * (d2udx2 + d2udy2)
        Xresidual = momentumX - forceX

        momentumY = outputU * dvdx + outputV * dvdy
        forceY = -dpdy + nu * (d2vdx2 + d2vdy2)
        Yresidual = momentumY - forceY


        sl = slice(1, -1)
        loss_pde = (
                criterion(Xresidual[:, :, sl, sl], torch.zeros_like(Xresidual[:, :, sl, sl])) +
                criterion(Yresidual[:, :, sl, sl], torch.zeros_like(Yresidual[:, :, sl, sl])) +
                criterion(continuity[:, :, sl, sl], torch.zeros_like(continuity[:, :, sl, sl]))
        )

        grid_pts = torch.stack([
            2 * idx_float[:, 1] - 1,
            2 * idx_float[:, 0] - 1
        ], dim=-1).view(1, -1, 1, 2)

        pred_uvp = F.grid_sample(output_pad, grid_pts,
                                 mode='bilinear', align_corners=True).view(3, -1)
        u_pred, v_pred, p_pred = pred_uvp


        ref_uvp = F.grid_sample(torch.cat([U_ref, V_ref, P_ref_zm], dim=1),
                                grid_pts, align_corners=True).view(3, -1)
        u_ref, v_ref, p_ref = ref_uvp

        loss_data = (criterion(u_pred, u_ref) +
                     criterion(v_pred, v_ref) +
                     criterion(p_pred, p_ref))

        loss = loss_pde + loss_data

        with torch.no_grad():
            Mag_pred = torch.sqrt(outputU ** 2 + outputV ** 2)
            l2_error_u = torch.sqrt(torch.mean((outputU - U_ref) ** 2)) / torch.sqrt(torch.mean(U_ref ** 2))
            l2_error_v = torch.sqrt(torch.mean((outputV - V_ref) ** 2)) / torch.sqrt(torch.mean(V_ref ** 2))
            l2_error_p = torch.sqrt(torch.mean((outputP - P_ref_zm) ** 2)) / torch.sqrt(torch.mean(P_ref_zm ** 2))
            l2_error_mag = torch.sqrt(torch.mean((Mag_pred - Mag_ref) ** 2)) / torch.sqrt(torch.mean(Mag_ref ** 2))
            l2_error_av = (l2_error_u + l2_error_v + l2_error_p + l2_error_mag) / 4.0

        loss.backward()


        for name, param in list(model.named_parameters()) + list(key_points_model.named_parameters()):
            if param.grad is not None and not torch.all(torch.isfinite(param.grad)):
                print(f"[WARN] Detected non-finite grad in {name}; zeroing it.")
                param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        torch.nn.utils.clip_grad_norm_(key_points_model.parameters(), CLIP_NORM)

        optimizer.step()


        CVT_INTERVAL = 500
        device = 'cuda'

        if epoch > 0 and epoch % CVT_INTERVAL == 0 and iteration == len(training_data_loader) - 1:
            print(f"\n>>> [CVT] Epoch {epoch}: Performing Sensor Optimization...")

            with torch.no_grad():


                curr_ref = torch.cat([sol_u, sol_v, P_ref_zm], dim=1).to(device)
                curr_pred = torch.cat([outputU, outputV, outputP], dim=1).detach()

                diff = curr_pred - curr_ref
                error_sq = torch.sum(diff ** 2, dim=1)
                density_map = torch.sqrt(error_sq).mean(dim=0)

                current_pos_01 = key_points_model.get_normalized()


                current_pos_xy_01 = torch.flip(current_pos_01, dims=[1])
                current_pos_xy_11 = 2.0 * current_pos_xy_01 - 1.0

                new_pos_xy_11 = compute_cvt_update(current_pos_xy_11, density_map, device=device)


                new_pos_xy_01 = (new_pos_xy_11 + 1.0) / 2.0
                new_pos_yx_01 = torch.flip(new_pos_xy_01, dims=[1])

                new_pos_yx_01 = torch.clamp(new_pos_yx_01, 0.01, 0.99)
                new_raw = torch.logit(new_pos_yx_01)

                key_points_model.raw.data.copy_(new_raw)

                shift = torch.norm(new_pos_yx_01 - current_pos_01, dim=1).mean().item()
                print(f"    -> Density Peak: {density_map.max().item():.5f}")
                print(f"    -> Sensors Updated. Mean Shift: {shift:.5f}")


        with torch.no_grad():
            key_points_model.raw.clamp_(-4.6, 4.6)

    if epoch % 100 == 0 or epoch % nEpochs == 0:
        print(f"[Epoch {epoch:4d}] Loss = {loss.item():.2e}")
        print(
            f"L2 Errors: U={l2_error_u.item():.4f}, V={l2_error_v.item():.4f}, P={l2_error_p.item():.4f}, Mag={l2_error_mag.item():.4f}, AV={l2_error_av.item():.4f}")

        key_history.append(key_points_model.get_normalized().detach().cpu().numpy())

    return loss.item(), l2_error_u.item(), l2_error_v.item(), l2_error_p.item(), l2_error_mag.item(), l2_error_av.item()


if __name__ == '__main__':
    losses = []
    errors_u = []
    errors_v = []
    errors_p = []
    errors_mag = []
    errors_av = []

    global_min_mag_error = 1e9
    best_epoch_mag = -1
    best_model_path = save_dir / "best_model_mag.pth"
    best_keypoints_path = save_dir / "best_keypoints_mag.pth"

    t0 = time.time()
    for ep in range(1, nEpochs + 1):
        loss, err_u, err_v, err_p, err_mag, err_av = train(ep)

        losses.append(loss)
        errors_u.append(err_u)
        errors_v.append(err_v)
        errors_p.append(err_p)
        errors_mag.append(err_mag)
        errors_av.append(err_av)


        if err_mag < global_min_mag_error:
            global_min_mag_error = err_mag
            best_epoch_mag = ep
            print(f"    >>> [New Record] Best Mag Error: {err_mag:.6f} at Epoch {ep}")

            torch.save(model.state_dict(), best_model_path)
            torch.save(key_points_model.state_dict(), best_keypoints_path)


        if ep % 100 == 0:
            data_path = save_dir / "training_log.csv"
            np.savetxt(data_path,
                       np.column_stack([losses, errors_u, errors_v, errors_p, errors_mag, errors_av]),
                       delimiter=',',
                       header='loss,err_u,err_v,err_p,err_mag, err_av',
                       comments='')

    dt = time.time() - t0
    print(f"Total training time: {dt:.2f} seconds")


    np.savetxt(save_dir / "final_data.csv",
               np.column_stack([losses, errors_u, errors_v, errors_p, errors_mag, errors_av]),
               delimiter=',', header='loss,err_u,err_v,err_p,err_mag, err_av')

    key_hist_arr = np.stack(key_history, axis=0)
    np.save(save_dir / "keypoints_history.npy", key_hist_arr)
    np.savetxt(save_dir / "time.txt", [dt])


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
    plt.savefig(save_dir / "training_convergence.png", dpi=300)
    plt.show()


    print(f"\nTraining Finished. Loading best model from Epoch {best_epoch_mag} to generate final results...")

    if best_epoch_mag != -1:
        model.load_state_dict(torch.load(best_model_path))
        key_points_model.load_state_dict(torch.load(best_keypoints_path))
    model.eval()


    first_batch = next(iter(training_data_loader))
    [_, coord_tensor, _, _, _, _, _, _, _, _, _, _, _] = to4DTensor(first_batch)

    key_positions = key_points_model.get_normalized()
    alpha_final = 80.0


    voronoi_input_final = softVoronoi(
        coord_tensor[0],
        U_ref[0, 0], V_ref[0, 0], P_ref_zm[0, 0],
        key_positions,
        (ny, nx), alpha=alpha_final)

    with torch.no_grad():
        output = model(coord_tensor, voronoi_input_final)
        output_pad = udfpad(output)

        outputU = output_pad[:, 0, :, :].reshape(output_pad.shape[0], 1, output_pad.shape[2], output_pad.shape[3])
        outputV = output_pad[:, 1, :, :].reshape(output_pad.shape[0], 1, output_pad.shape[2], output_pad.shape[3])
        outputP = output_pad[:, 2, :, :].reshape(output_pad.shape[0], 1, output_pad.shape[2], output_pad.shape[3])


        outputU, outputV, outputP = apply_annulus_rotating_bc(
            outputU, outputV, outputP,
            myMesh=myMesh, Omega=Omega, xc=xc, yc=yc,
            padSingleSide=padSingleSide
        )


        Pm_pred = outputP.mean(dim=(-2, -1), keepdim=True)
        outputP = outputP - Pm_pred


    u_pred = output_pad[0, 0, 1:-1, 1:-1].cpu().numpy()
    v_pred = output_pad[0, 1, 1:-1, 1:-1].cpu().numpy()
    p_pred = output_pad[0, 2, 1:-1, 1:-1].cpu().numpy()
    mag_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)


    u_true = OFU_sb[1:-1, 1:-1]
    v_true = OFV_sb[1:-1, 1:-1]
    p_true = P_ref_zm[0, 0, 1:-1, 1:-1].cpu().numpy()
    mag_true = OFMag_sb[1:-1, 1:-1]

    x_plot = coord_tensor[0, 0, 1:-1, 1:-1].cpu().numpy()
    y_plot = coord_tensor[0, 1, 1:-1, 1:-1].cpu().numpy()


    print("Generating best result plots...")
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))

    fields = [
        ('U-Velocity', u_pred, u_true),
        ('V-Velocity', v_pred, v_true),
        ('Pressure', p_pred, p_true),
        ('Magnitude', mag_pred, mag_true)
    ]

    for row, (name, pred, true) in enumerate(fields):
        vmin = min(pred.min(), true.min())
        vmax = max(pred.max(), true.max())
        err = np.abs(pred - true)

        ax = axes[row, 0]
        cf = ax.contourf(x_plot, y_plot, pred, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f"Predicted {name}")
        plt.colorbar(cf, ax=ax)
        ax.axis('scaled')

        ax = axes[row, 1]
        cf = ax.contourf(x_plot, y_plot, true, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f"Reference {name}")
        plt.colorbar(cf, ax=ax)
        ax.axis('scaled')

        ax = axes[row, 2]
        cf = ax.contourf(x_plot, y_plot, err, levels=50, cmap='inferno')
        ax.set_title(f"Absolute Error {name}")
        plt.colorbar(cf, ax=ax)
        ax.axis('scaled')

    plt.tight_layout()
    plt.savefig(save_dir / "best_result_fields.png", dpi=300)
    plt.close()


    print("Saving best sensor positions...")
    best_pos_norm = key_positions.detach().cpu().numpy()

    with open(save_dir / "best_sensors_location.txt", "w") as f:
        f.write(f"Best Epoch: {best_epoch_mag}\n")
        f.write(f"Best Mag Error: {global_min_mag_error:.6f}\n")
        f.write("-" * 30 + "\n")
        f.write("Sensor Index (y, x) | Normalized (y, x)\n")
        for i, pos in enumerate(best_pos_norm):
            iy = pos[0] * (ny - 1)
            ix = pos[1] * (nx - 1)
            f.write(f"Point {i + 1}: ({iy:.2f}, {ix:.2f}) | ({pos[0]:.4f}, {pos[1]:.4f})\n")


    u_arr = np.array(errors_u)
    v_arr = np.array(errors_v)
    p_arr = np.array(errors_p)
    mag_arr = np.array(errors_mag)
    av_arr = np.array(errors_av)
    ep_arr = np.arange(1, nEpochs + 1)

    summary_path = save_dir / "final_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=== Minimum Errors Summary ===\n")


        def write_min(name, arr):
            idx = np.argmin(arr)
            f.write(f"{name} Min Error: {arr[idx]:.6f} at Epoch {ep_arr[idx]}\n")


        write_min("U", u_arr)
        write_min("V", v_arr)
        write_min("P", p_arr)
        write_min("Mag", mag_arr)
        write_min("AV", av_arr)

        f.write("\n=== Best Model Saved Based on Mag Error ===\n")
        f.write(f"Saved Epoch: {best_epoch_mag}\n")
        f.write(f"Saved Mag Error: {global_min_mag_error:.6f}\n")

    print(f"All results saved to {save_dir}")
