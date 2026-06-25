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
from pyMesh import hcubeMesh, visualize2D, plotBC, plotMesh, setAxisLabel,\
    np2cuda, to4DTensor
from model import USCNN, USCNNSepPhi, USCNNSep, DDBasic
from readOF import convertOFMeshToImage, convertOFMeshToImage_StructuredMesh
from sklearn.metrics import mean_squared_error as calMSE
import Ofpp
from pathlib import Path
from view import view_compare_uvp_mag_physical


save_dir = Path(r"./output/improved_1.28_data_records/1 no_data_points")
save_dir.mkdir(parents=True, exist_ok=True)


h = 0.01
r = 0.5
R = 1
dtheta = 0

OFBCCoord = Ofpp.parse_boundary_field('discRe350_0.01nu/0/C')

OFLEFTC = OFBCCoord[b'left'][b'value']
OFRIGHTC = OFBCCoord[b'right'][b'value']
leftX = r * np.cos(np.linspace(dtheta, 2 * np.pi - dtheta, 276))
leftY = r * np.sin(np.linspace(dtheta, 2 * np.pi - dtheta, 276))
rightX = R * np.cos(np.linspace(dtheta, 2 * np.pi - dtheta, 276))
rightY = R * np.sin(np.linspace(dtheta, 2 * np.pi - dtheta, 276))
lowX = np.linspace(leftX[0], rightX[0], 49);
lowY = lowX * 0 + np.sin(dtheta)
upX = np.linspace(leftX[-1], rightX[-1], 49);
upY = upX * 0 - np.sin(dtheta)
ny = len(leftX);
nx = len(lowX)
myMesh = hcubeMesh(leftX, leftY, rightX, rightY,
                   lowX, lowY, upX, upY, h, True, True,
                   tolMesh=1e-10, tolJoint=0.01)


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

model = USCNNSep(h, nx, ny, NvarInput, NvarOutput, 'ortho').to('cuda')


model = model.to('cuda')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
padSingleSide = 1
udfpad = nn.ConstantPad2d([padSingleSide, padSingleSide, padSingleSide, padSingleSide], 0)

MeshList = []
MeshList.append(myMesh)
train_set = VaryGeoDataset(MeshList)
training_data_loader = DataLoader(dataset=train_set,
                                  batch_size=batchSize)
OFPic = convertOFMeshToImage_StructuredMesh(nx, ny, 'discRe350_0.01nu/0/C',
                                            ['discRe350_0.01nu/2969/U',
                                             'discRe350_0.01nu/2969/p'],
                                            [0, 1, 0, 1], 0.0, False)
OFX = OFPic[:, :, 0]
OFY = OFPic[:, :, 1]
OFU = OFPic[:, :, 2]
OFV = OFPic[:, :, 3]
OFP = OFPic[:, :, 4]
OFU_sb = np.zeros(OFU.shape)
OFV_sb = np.zeros(OFV.shape)
OFP_sb = np.zeros(OFP.shape)

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
\
\
\
\
\

    B, C, H, W = outputU.shape
    assert C == 1, "Expected channel dimension to be 1 (processed channel by channel)"


    ii = slice(padSingleSide, W - padSingleSide)
    jj = slice(padSingleSide, H - padSingleSide)


    device = outputU.device
    dtype = outputU.dtype

    X = torch.as_tensor(myMesh.x, dtype=dtype, device=device)
    Y = torch.as_tensor(myMesh.y, dtype=dtype, device=device)


    xL = X[jj, 0:1]
    yL = Y[jj, 0:1]

    xR = X[jj, -1:]
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
    xRes = 0
    yRes = 0
    mRes = 0
    eU = 0
    eV = 0
    eP = 0
    for iteration, batch in enumerate(training_data_loader):
        [JJInv, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta] = to4DTensor(batch)
        optimizer.zero_grad()
        output = model(coord)
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
        Pm_ref = P_ref.mean(dim=(-2, -1), keepdim=True)
        P_ref_zm = P_ref - Pm_ref


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

        loss = (criterion(Xresidual, Xresidual * 0) +\
                criterion(Yresidual, Yresidual * 0) +\
                criterion(continuity, continuity * 0))

        with torch.no_grad():
            Mag_pred = torch.sqrt(outputU ** 2 + outputV ** 2)

            l2_error_u = torch.sqrt(torch.mean((outputU - U_ref) ** 2)) / torch.sqrt(
                torch.mean(U_ref ** 2))
            l2_error_v = torch.sqrt(torch.mean((outputV - V_ref) ** 2)) / torch.sqrt(
                torch.mean(V_ref ** 2))


            l2_error_p = torch.sqrt(torch.mean((outputP - P_ref_zm) ** 2)) /\
                         torch.sqrt(torch.mean(P_ref_zm ** 2))
            l2_error_mag = torch.sqrt(torch.mean((Mag_pred - Mag_ref) ** 2)) / torch.sqrt(
                torch.mean(Mag_ref ** 2))
            l2_error_av = (l2_error_u + l2_error_v + l2_error_p + l2_error_mag) / 4.0

        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        print(f"[Epoch {epoch:4d}] Loss = {loss.item():.2e}")
        print(
            f"L2 Errors: U={l2_error_u.item():.4f}, V={l2_error_v.item():.4f}, P={l2_error_p.item():.4f}, Mag={l2_error_mag.item():.4f}, AV={l2_error_av.item():.4f}")

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

        if ep % 100 == 0:

            data_path = save_dir / "training_log.csv"
            np.savetxt(
                data_path,
                np.column_stack([losses, errors_u, errors_v, errors_p, errors_mag, errors_av]),
                delimiter=',',
                header='loss,err_u,err_v,err_p,err_mag,err_av',
                comments=''
            )


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


    print(f"\nTraining Finished. Loading best model from Epoch {best_epoch_mag} to generate final results...")


    if best_epoch_mag != -1:
        model.load_state_dict(torch.load(best_model_path))
    model.eval()


    first_batch = next(iter(training_data_loader))

    [_, coord_tensor, _, _, _, _, _, _, _, _] = to4DTensor(first_batch)

    with torch.no_grad():
        output = model(coord_tensor)
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


    u_pred = outputU[0, 0, 1:-1, 1:-1].cpu().numpy()
    v_pred = outputV[0, 0, 1:-1, 1:-1].cpu().numpy()
    p_pred = outputP[0, 0, 1:-1, 1:-1].cpu().numpy()
    mag_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)


    p_ref_val = OFP_sb[1:-1, 1:-1]
    p_ref_val = p_ref_val - np.mean(p_ref_val)

    u_true = OFU_sb[1:-1, 1:-1]
    v_true = OFV_sb[1:-1, 1:-1]
    p_true = p_ref_val
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
        f.write("\nNote: Case3 uses fixed grid (no sensor optimization).\n")

    print(f"All results saved to {save_dir}")
