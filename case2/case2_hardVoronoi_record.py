import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from torch.utils.data import DataLoader
import time
from voronoi_utils import VaryGeoDataset_PairedSolutionOld, to4DTensor, generate_voronoi_input
from view import view_cfd_fields, view_cfd_fields_physical
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
from scipy.interpolate import griddata


OUT_DIR = "./output/2026.1.28_improved_data_records/3 hard_Voronoi"
os.makedirs(OUT_DIR, exist_ok=True)


h = 0.01

key_indices = [
    (8, 10),
    (15, 65),
    (35, 50),
    (45, 5)
]


key_indices = [(b, a) for (a, b) in key_indices]


print("Key point indices:", key_indices)

OFBCCoord = Ofpp.parse_boundary_field('TemplateCase_simpleVessel/3200/C')
OFLOWC = OFBCCoord[b'low'][b'value']
OFUPC = OFBCCoord[b'up'][b'value']
OFLEFTC = OFBCCoord[b'left'][b'value']
OFRIGHTC = OFBCCoord[b'rifht'][b'value']

leftX = OFLEFTC[:, 0];
leftY = OFLEFTC[:, 1]
lowX = OFLOWC[:, 0];
lowY = OFLOWC[:, 1]
rightX = OFRIGHTC[:, 0];
rightY = OFRIGHTC[:, 1]
upX = OFUPC[:, 0];
upY = OFUPC[:, 1]
ny = len(leftX);
nx = len(lowX)
myMesh = hcubeMesh(leftX, leftY, rightX, rightY,
                   lowX, lowY, upX, upY, h, True, True,
                   tolMesh=1e-10, tolJoint=1e-2)

batchSize = 1
NvarInput = 2
NvarOutput = 1
nEpochs = 30000
lr = 0.001
Ns = 1


nu=0.02/45.00


model = VoronoiEnhancedUSCNNSep(h, nx, ny, nVorCh=4, initWay='ortho').to('cuda')

model = model.to('cuda')

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
padSingleSide = 1
udfpad = nn.ConstantPad2d([padSingleSide, padSingleSide, padSingleSide, padSingleSide], 0)

MeshList = []
MeshList.append(myMesh)


OFPic = convertOFMeshToImage_StructuredMesh(nx, ny, 'TemplateCase_simpleVessel/3200/C',
                                            ['Re450/3200/U',
                                             'Re450/3200/p'],
                                            [0, 1, 0, 1], 0.0, False)
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

fcnn_P = np.zeros(OFU.shape)
fcnn_U = np.zeros(OFV.shape)
fcnn_V = np.zeros(OFP.shape)
fcnn_Mag = np.zeros(OFMag.shape)

fcnn = np.load('comparison_160000iter.npz')
fcnn_P_ = fcnn['p_NN'].reshape(OFU_sb.shape)
fcnn_U_ = fcnn['u_NN'].reshape(OFU_sb.shape)
fcnn_V_ = fcnn['v_NN'].reshape(OFU_sb.shape)
fcnn_Mag_ = np.sqrt(fcnn_U_ ** 2 + fcnn_V_ ** 2)
fcnn_X = fcnn['x_coord'].reshape(OFU_sb.shape)
fcnn_Y = fcnn['y_coord'].reshape(OFU_sb.shape)

for i in range(nx):
    for j in range(ny):
        dist = (myMesh.x[j, i] - fcnn_X) ** 2 + (myMesh.y[j, i] - fcnn_Y) ** 2
        idx_min = np.where(dist == dist.min())
        fcnn_U[j, i] = fcnn_U_[idx_min]
        fcnn_V[j, i] = fcnn_V_[idx_min]
        fcnn_P[j, i] = fcnn_P_[idx_min]
        fcnn_Mag[j, i] = fcnn_Mag_[idx_min]

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

SolutionList = [np.stack([OFU_sb, OFV_sb, OFP_sb], axis=-1)]

train_set = VaryGeoDataset_PairedSolutionOld(MeshList, SolutionList, key_indices)
training_data_loader = DataLoader(dataset=train_set,
                                  batch_size=batchSize)


coord_np = np.stack([myMesh.x, myMesh.y], axis=0)

try:
    vor = generate_voronoi_input(coord_np, OFU_sb, OFV_sb, OFP_sb, key_indices, (ny, nx))

    print(f"Voronoi input generation succeeded。")
except Exception as e:
    print(f"Voronoi generation warning: {e}")


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
    startTime = time.time()
    loss_data = 0
    xRes = 0
    yRes = 0
    mRes = 0
    eU = 0
    eV = 0
    eP = 0
    for iteration, batch in enumerate(training_data_loader):


        [JJInv, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta,
         sol_u, sol_v, sol_p, voronoi_input] = to4DTensor(batch)

        optimizer.zero_grad()
        output = model(coord, voronoi_input)
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


        for j in range(batchSize):
            outputU[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = output[j, 0, -1, :].reshape(1,
                                                                                                       nx - 2 * padSingleSide)
            outputU[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = 0
            outputU[j, 0, padSingleSide:-padSingleSide, -padSingleSide:] = 0
            outputU[j, 0, padSingleSide:-padSingleSide, 0:padSingleSide] = 0
            outputU[j, 0, 0, 0] = 1 * (outputU[j, 0, 0, 1])
            outputU[j, 0, 0, -1] = 1 * (outputU[j, 0, 0, -2])
            outputV[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = output[j, 1, -1, :].reshape(1,
                                                                                                       nx - 2 * padSingleSide)
            outputV[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = 1
            outputV[j, 0, padSingleSide:-padSingleSide, -padSingleSide:] = 0
            outputV[j, 0, padSingleSide:-padSingleSide, 0:padSingleSide] = 0
            outputV[j, 0, 0, 0] = 1 * (outputV[j, 0, 0, 1])
            outputV[j, 0, 0, -1] = 1 * (outputV[j, 0, 0, -2])
            outputP[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = 0
            outputP[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = output[j, 2, 0, :].reshape(1,
                                                                                                     nx - 2 * padSingleSide)
            outputP[j, 0, padSingleSide:-padSingleSide, -padSingleSide:] = output[j, 2, :, -1].reshape(
                ny - 2 * padSingleSide, 1)
            outputP[j, 0, padSingleSide:-padSingleSide, 0:padSingleSide] = output[j, 2, :, 0].reshape(
                ny - 2 * padSingleSide, 1)
            outputP[j, 0, 0, 0] = 1 * (outputP[j, 0, 0, 1])
            outputP[j, 0, 0, -1] = 1 * (outputP[j, 0, 0, -2])


        dudx = dfdx(outputU, dydeta, dydxi, Jinv)
        d2udx2 = dfdx(dudx, dydeta, dydxi, Jinv)
        dudy = dfdy(outputU, dxdxi, dxdeta, Jinv)
        d2udy2 = dfdy(dudy, dxdxi, dxdeta, Jinv)
        dvdx = dfdx(outputV, dydeta, dydxi, Jinv)
        d2vdx2 = dfdx(dvdx, dydeta, dydxi, Jinv)
        dvdy = dfdy(outputV, dxdxi, dxdeta, Jinv)
        d2vdy2 = dfdy(dvdy, dxdxi, dxdeta, Jinv)
        dpdx = dfdx(outputP, dydeta, dydxi, Jinv)
        d2pdx2 = dfdx(dpdx, dydeta, dydxi, Jinv)
        dpdy = dfdy(outputP, dxdxi, dxdeta, Jinv)
        d2pdy2 = dfdy(dpdy, dxdxi, dxdeta, Jinv)


        continuity = dudx + dvdy;
        momentumX = outputU * dudx + outputV * dudy
        forceX = -dpdx + nu * (d2udx2 + d2udy2)

        Xresidual = momentumX - forceX
        momentumY = outputU * dvdx + outputV * dvdy

        forceY = -dpdy + nu * (d2vdx2 + d2vdy2)
        Yresidual = momentumY - forceY

        loss_pde = (criterion(Xresidual, Xresidual * 0) +\
                    criterion(Yresidual, Yresidual * 0) +\
                    criterion(continuity, continuity * 0))


        for iy, ix in key_indices:

            u_pred = outputU[0, 0, iy, ix]
            v_pred = outputV[0, 0, iy, ix]
            p_pred = outputP[0, 0, iy, ix]


            u_ref_point = U_ref[0, 0, iy, ix]
            v_ref_point = V_ref[0, 0, iy, ix]
            p_ref_point = P_ref[0, 0, iy, ix]

            loss_data += criterion(u_pred, u_ref_point)
            loss_data += criterion(v_pred, v_ref_point)
            loss_data += criterion(p_pred, p_ref_point)

        loss = loss_pde + loss_data

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

    if epoch % 100 == 0:
        print(f"[Epoch {epoch:4d}] Loss = {loss.item():.2e} | Mag Err = {l2_error_mag.item():.5f}")

    return loss.item(), l2_error_u.item(), l2_error_v.item(), l2_error_p.item(), l2_error_mag.item(), l2_error_av.item()


if __name__ == '__main__':

    losses = []
    errors_u = []
    errors_v = []
    errors_p = []
    errors_mag = []
    errors_av = []

    best_mag_error = float('inf')
    best_epoch = -1

    t0 = time.time()
    for ep in range(1, nEpochs + 1):

        loss, err_u, err_v, err_p, err_mag, err_av = train(ep)


        losses.append(loss)
        errors_u.append(err_u)
        errors_v.append(err_v)
        errors_p.append(err_p)
        errors_mag.append(err_mag)
        errors_av.append(err_av)


        if err_mag < best_mag_error:
            best_mag_error = err_mag
            best_epoch = ep


            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pth"))
            print(f"[*] New Best Found at Epoch {ep}: Mag Err = {best_mag_error:.5f}")


        if ep % 100 == 0 or ep == nEpochs:
            try:
                np.savetxt(os.path.join(OUT_DIR, 'training_log.csv'),
                           np.column_stack([losses, errors_u, errors_v, errors_p, errors_mag, errors_av]),
                           delimiter=',',
                           header='loss,err_u,err_v,err_p,err_mag, err_av')
            except PermissionError:
                pass


    dt = time.time() - t0
    print(f"Total training time: {dt:.2f} seconds")
    print(f"Best Mag Error: {best_mag_error:.5f} (Epoch {best_epoch})")


    print("\nGenerating best result report...")


    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_model.pth")))
    model.eval()


    with torch.no_grad():
        batch = next(iter(training_data_loader))
        [JJInv, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta,
         sol_u, sol_v, sol_p, voronoi_input] = to4DTensor(batch)

        output = model(coord, voronoi_input)
        output_pad = udfpad(output)

        outputU = output_pad[:, 0, :, :].reshape(1, 1, ny, nx)
        outputV = output_pad[:, 1, :, :].reshape(1, 1, ny, nx)
        outputP = output_pad[:, 2, :, :].reshape(1, 1, ny, nx)


        j = 0
        outputU[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = output[j, 0, -1, :].reshape(1,
                                                                                                   nx - 2 * padSingleSide)
        outputU[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = 0
        outputU[j, 0, padSingleSide:-padSingleSide, -padSingleSide:] = 0
        outputU[j, 0, padSingleSide:-padSingleSide, 0:padSingleSide] = 0
        outputU[j, 0, 0, 0] = 1 * (outputU[j, 0, 0, 1])
        outputU[j, 0, 0, -1] = 1 * (outputU[j, 0, 0, -2])
        outputV[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = output[j, 1, -1, :].reshape(1,
                                                                                                   nx - 2 * padSingleSide)
        outputV[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = 1
        outputV[j, 0, padSingleSide:-padSingleSide, -padSingleSide:] = 0
        outputV[j, 0, padSingleSide:-padSingleSide, 0:padSingleSide] = 0
        outputV[j, 0, 0, 0] = 1 * (outputV[j, 0, 0, 1])
        outputV[j, 0, 0, -1] = 1 * (outputV[j, 0, 0, -2])
        outputP[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = 0
        outputP[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = output[j, 2, 0, :].reshape(1,
                                                                                                 nx - 2 * padSingleSide)
        outputP[j, 0, padSingleSide:-padSingleSide, -padSingleSide:] = output[j, 2, :, -1].reshape(
            ny - 2 * padSingleSide, 1)
        outputP[j, 0, padSingleSide:-padSingleSide, 0:padSingleSide] = output[j, 2, :, 0].reshape(
            ny - 2 * padSingleSide, 1)
        outputP[j, 0, 0, 0] = 1 * (outputP[j, 0, 0, 1])
        outputP[j, 0, 0, -1] = 1 * (outputP[j, 0, 0, -2])


        X_np = coord[0, 0, :, :].cpu().numpy()
        Y_np = coord[0, 1, :, :].cpu().numpy()

        U_pred = outputU[0, 0, :, :].cpu().numpy()
        V_pred = outputV[0, 0, :, :].cpu().numpy()
        P_pred = outputP[0, 0, :, :].cpu().numpy()
        Mag_pred = np.sqrt(U_pred ** 2 + V_pred ** 2)


        U_true = U_ref[0, 0, :, :].cpu().numpy()
        V_true = V_ref[0, 0, :, :].cpu().numpy()
        P_true = P_ref[0, 0, :, :].cpu().numpy()
        Mag_true = Mag_ref[0, 0, :, :].cpu().numpy()

        Err_mag = np.abs(Mag_pred - Mag_true)
        Err_p = np.abs(P_pred - P_true)


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


    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    im0 = axes[0, 0].contourf(X_np, Y_np, Mag_pred, levels=50, cmap='viridis')
    axes[0, 0].set_title(f"Predicted Mag (Ep {best_epoch})");
    plt.colorbar(im0, ax=axes[0, 0])
    im1 = axes[0, 1].contourf(X_np, Y_np, Mag_true, levels=50, cmap='viridis')
    axes[0, 1].set_title(f"Reference Mag");
    plt.colorbar(im1, ax=axes[0, 1])
    im2 = axes[0, 2].contourf(X_np, Y_np, Err_mag, levels=50, cmap='hot')
    axes[0, 2].set_title(f"Mag Abs Error");
    plt.colorbar(im2, ax=axes[0, 2])
    im3 = axes[1, 0].contourf(X_np, Y_np, P_pred, levels=50, cmap='viridis')
    axes[1, 0].set_title(f"Predicted Pressure");
    plt.colorbar(im3, ax=axes[1, 0])
    im4 = axes[1, 1].contourf(X_np, Y_np, P_true, levels=50, cmap='viridis')
    axes[1, 1].set_title(f"Reference Pressure");
    plt.colorbar(im4, ax=axes[1, 1])
    im5 = axes[1, 2].contourf(X_np, Y_np, Err_p, levels=50, cmap='hot')
    axes[1, 2].set_title(f"Pressure Abs Error");
    plt.colorbar(im5, ax=axes[1, 2])


    for i, (iy, ix) in enumerate(key_indices):
        axes[0, 0].plot(X_np[iy, ix], Y_np[iy, ix], 'ro', markersize=4)
        axes[0, 1].plot(X_np[iy, ix], Y_np[iy, ix], 'ro', markersize=4)
        axes[0, 2].plot(X_np[iy, ix], Y_np[iy, ix], 'ro', markersize=4)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Best_Comparison.png"), dpi=100)
    plt.close(fig)


    sensor_coords = []
    for (iy, ix) in key_indices:
        sensor_coords.extend([myMesh.x[iy, ix], myMesh.y[iy, ix]])
    np.savetxt(os.path.join(OUT_DIR, "fixed_sensors.csv"),
               np.array(sensor_coords).reshape(1, -1),
               delimiter=',', header="x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6", comments='')


    with open(os.path.join(OUT_DIR, "final_summary.txt"), "w") as f:
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Minimum Mag Error: {best_mag_error:.6f}\n")
        f.write(f"Total Training Time: {dt:.2f}s\n")


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

    print(f"All results saved to: {OUT_DIR}")
