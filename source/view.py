
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from pyMesh import visualize2D, setAxisLabel
from matplotlib.colors import Normalize
import matplotlib as mpl

def view_cfd_fields(OFU, OFV, OFP, myMesh, OFMag=None, *,
                    cmap='coolwarm', interpolation='nearest',
                    save_path=None, show=True):
\
\
\
\
\


    ny, nx = myMesh.x.shape
    for name, arr in [('OFU', OFU), ('OFV', OFV), ('OFP', OFP)]:
        assert arr.shape == (ny, nx), f"{name}  shape {arr.shape}  is inconsistent with grid {(ny, nx)} inconsistent"


    if OFMag is None:
        OFMag = np.hypot(OFU, OFV)


    x_min, x_max = float(myMesh.x.min()), float(myMesh.x.max())
    y_min, y_max = float(myMesh.y.min()), float(myMesh.y.max())
    extent = [x_min, x_max, y_min, y_max]


    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)

    im0 = axes[0].imshow(OFU, origin='lower', extent=extent,
                         cmap=cmap, interpolation=interpolation)
    axes[0].set_title('OFU (u-component)')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(OFV, origin='lower', extent=extent,
                         cmap=cmap, interpolation=interpolation)
    axes[1].set_title('OFV (v-component)')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(OFP, origin='lower', extent=extent,
                         cmap=cmap, interpolation=interpolation)
    axes[2].set_title('OFP (pressure)')
    axes[2].set_xlabel('x'); axes[2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[2])

    im3 = axes[3].imshow(OFMag, origin='lower', extent=extent,
                         cmap=cmap, interpolation=interpolation)
    axes[3].set_title('Velocity Magnitude |V|')
    axes[3].set_xlabel('x'); axes[3].set_ylabel('y')
    plt.colorbar(im3, ax=axes[3])


    for ax in axes:
        ax.set_aspect('equal', adjustable='box')

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def view_cfd_fields_physical(OFU, OFV, OFP, myMesh, OFMag=None, *,
                             cmap='coolwarm', save_path=None, show=True):
\
\

    ny, nx = myMesh.x.shape

    assert OFU.shape == (ny, nx), f"OFU shape {OFU.shape} != {(ny, nx)}"
    assert OFV.shape == (ny, nx), f"OFV shape {OFV.shape} != {(ny, nx)}"
    assert OFP.shape == (ny, nx), f"OFP shape {OFP.shape} != {(ny, nx)}"

    if OFMag is None:
        OFMag = np.hypot(OFU, OFV)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)


    ax = axes[0]
    _, cbar = visualize2D(ax, myMesh.x, myMesh.y, OFU, colorbarPosition='vertical')
    ax.set_title('OFU (u-component)')
    setAxisLabel(ax, 'p')
    ax.set_aspect('equal', adjustable='box')


    ax = axes[1]
    _, cbar = visualize2D(ax, myMesh.x, myMesh.y, OFV, colorbarPosition='vertical')
    ax.set_title('OFV (v-component)')
    setAxisLabel(ax, 'p')
    ax.set_aspect('equal', adjustable='box')


    ax = axes[2]
    _, cbar = visualize2D(ax, myMesh.x, myMesh.y, OFP, colorbarPosition='vertical')
    ax.set_title('OFP (pressure)')
    setAxisLabel(ax, 'p')
    ax.set_aspect('equal', adjustable='box')


    ax = axes[3]
    _, cbar = visualize2D(ax, myMesh.x, myMesh.y, OFMag, colorbarPosition='vertical')
    ax.set_title('Velocity Magnitude |V|')
    setAxisLabel(ax, 'p')
    ax.set_aspect('equal', adjustable='box')

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_voronoi_uvp_mask(voronoi_input, myMesh, *,
                          key_indices=None,
                          cmap='coolwarm',
                          save_path=None,
                          save_split_dir=None,
                          show=False):
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
\
\
\
\
\
\
\
\


    if isinstance(voronoi_input, torch.Tensor):
        v = voronoi_input.detach().cpu().numpy()
    else:
        v = np.asarray(voronoi_input)

    if v.ndim == 4 and v.shape[0] == 1:
        v = v[0]
    assert v.ndim == 3 and v.shape[0] == 4, f"Expected [4,ny,nx]，got {v.shape}"

    ny, nx = v.shape[1], v.shape[2]
    extent = [float(myMesh.x.min()), float(myMesh.x.max()),
              float(myMesh.y.min()), float(myMesh.y.max())]

    titles = ['u_Voronoi', 'v_Voronoi', 'p_Voronoi', 'Mask']


    uvp = v[0:3]
    vmin, vmax = float(uvp.min()), float(uvp.max())

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
    for i, ax in enumerate(axes):
        if i < 3:
            im = ax.imshow(v[i], origin='lower', extent=extent,
                           cmap=cmap, interpolation='nearest',
                           vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax)
        else:
            im = ax.imshow(v[3], origin='lower', extent=extent,
                           cmap='gray', interpolation='nearest',
                           vmin=0.0, vmax=1.0)
            plt.colorbar(im, ax=ax)

        ax.set_title(titles[i])
        ax.set_xlabel('ξ'); ax.set_ylabel('η')
        ax.set_aspect('equal', adjustable='box')


        if key_indices is not None and i == 3:
            xs = [myMesh.x[y, x] for (y, x) in key_indices]
            ys = [myMesh.y[y, x] for (y, x) in key_indices]
            ax.plot(xs, ys, 'ro', markersize=4, markerfacecolor='none', markeredgewidth=1)


    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')


    if save_split_dir:
        os.makedirs(save_split_dir, exist_ok=True)
        for i, name in enumerate(titles):
            plt.imsave(
                os.path.join(save_split_dir, f'{name}.png'),
                v[i],
                origin='lower',
                cmap=(cmap if i < 3 else 'gray'),
                vmin=(vmin if i < 3 else 0.0),
                vmax=(vmax if i < 3 else 1.0)
            )

    if show:
        plt.show()
    else:
        plt.close(fig)


def view_compare_uvp_mag_physical(OFU, OFV, OFP, predU, predV, predP, myMesh,
                                  padSingleSide=1, share_scale=True,
                                  cmap_field='coolwarm', cmap_mag='viridis',
                                  save_path=None, show=True,
                                  key_indices_init=None, key_indices_now=None
                                  ):
\
\
\
\
\


    def _tonp(a):
        try:
            import torch
            if isinstance(a, torch.Tensor):
                a = a.detach().cpu().numpy()
        except Exception:
            pass
        a = np.asarray(a)
        if a.ndim > 2:
            a = np.squeeze(a)
        assert a.ndim == 2, f"expect 2D array after squeeze, got {a.shape}"
        return a


    OFU, OFV, OFP = map(_tonp, (OFU, OFV, OFP))
    pU,  pV,  pP  = map(_tonp, (predU, predV, predP))
    X = _tonp(getattr(myMesh, 'x')); Y = _tonp(getattr(myMesh, 'y'))
    ny, nx = X.shape

    def _valid_idx(y, x):
        if not (0 <= y < ny and 0 <= x < nx):
            return False
        if padSingleSide and padSingleSide > 0:
            return (padSingleSide <= y < ny - padSingleSide) and (padSingleSide <= x < nx - padSingleSide)
        return True

    def _scatter_keys(ax, keys, *, marker='o', face='none', edge='orange', color=None, lw=1.2, size=22):
        if not keys:
            return
        xs, ys = [], []
        ny, nx = X.shape

        for iy, ix in keys:
            iy = int(round(iy));
            ix = int(round(ix))

            iy = min(max(iy, 0), ny - 1)
            ix = min(max(ix, 0), nx - 1)

            if padSingleSide and padSingleSide > 0:
                iy = min(max(iy, padSingleSide), ny - 1 - padSingleSide)
                ix = min(max(ix, padSingleSide), nx - 1 - padSingleSide)

            xs.append(X[iy, ix]);
            ys.append(Y[iy, ix])

        if not xs:
            return
        if marker == 'x':
            ax.scatter(xs, ys, s=size, marker=marker, c=color if color else edge, linewidths=lw)
        else:
            ax.scatter(xs, ys, s=size, marker=marker, facecolors=face, edgecolors=edge, linewidths=lw)

    if padSingleSide and padSingleSide > 0:
        s = slice(padSingleSide, -padSingleSide)
        OFU, OFV, OFP = OFU[s, s], OFV[s, s], OFP[s, s]
        pU,  pV,  pP  =  pU[s, s],  pV[s, s],  pP[s, s]
        X,   Y        =   X[s, s],   Y[s, s]


    OFM = np.hypot(OFU, OFV)
    pM  = np.hypot(pU,  pV)


    def _range_pair(a, b):
        vmin = float(np.nanmin([a.min(), b.min()]))
        vmax = float(np.nanmax([a.max(), b.max()]))
        if np.isclose(vmin, vmax):
            eps = 1e-12 if vmax == 0 else 1e-3*abs(vmax)
            vmin -= eps; vmax += eps
        return vmin, vmax

    if share_scale:
        umin, umax = _range_pair(OFU, pU)
        vmin, vmax = _range_pair(OFV, pV)
        pmin, pmax = _range_pair(OFP, pP)
        mmin, mmax = _range_pair(OFM, pM)
    else:
        umin, umax = OFU.min(), OFU.max()
        vmin, vmax = OFV.min(), OFV.max()
        pmin, pmax = OFP.min(), OFP.max()
        mmin, mmax = OFM.min(), OFM.max()


    def _draw(ax, X, Y, C, title, cmap, vmin, vmax):
        used_vis2d = False
        if 'visualize2D' in globals():
            try:
                visualize2D(X, Y, C, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)
                used_vis2d = True
            except Exception:
                used_vis2d = False
        if not used_vis2d:

            ax.pcolormesh(X, Y, C, shading='auto', cmap=cmap,
                          norm=Normalize(vmin=vmin, vmax=vmax))
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel('x'); ax.set_ylabel('y')


    fig, axes = plt.subplots(4, 2, figsize=(11, 14), constrained_layout=True)


    _draw(axes[0,0], X, Y, OFU, 'CFD: u', cmap_field, umin, umax)


    _draw(axes[0,1], X, Y, pU,  'Pred: u', cmap_field, umin, umax)


    _draw(axes[1,0], X, Y, OFV, 'CFD: v', cmap_field, vmin, vmax)


    _draw(axes[1,1], X, Y, pV,  'Pred: v', cmap_field, vmin, vmax)


    _draw(axes[2,0], X, Y, OFP, 'CFD: p', cmap_field, pmin, pmax)


    _draw(axes[2,1], X, Y, pP,  'Pred: p', cmap_field, pmin, pmax)


    _draw(axes[3,0], X, Y, OFM, 'CFD: |V|', cmap_mag, mmin, mmax)


    _draw(axes[3,1], X, Y, pM,  'Pred: |V|', cmap_mag, mmin, mmax)


    for ax in axes.flat:
        _scatter_keys(ax, key_indices_init, marker='o', face='none', edge='orange', lw=1.2, size=22)
        _scatter_keys(ax, key_indices_now, marker='x', color='cyan', lw=1.2, size=24)


    def _row_cbar(axrow, cmap, vmin, vmax):
        sm = mpl.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=axrow, orientation='horizontal', fraction=0.05, pad=0.06)

    _row_cbar(axes[0,:], cmap_field, umin, umax)
    _row_cbar(axes[1,:], cmap_field, vmin, vmax)
    _row_cbar(axes[2,:], cmap_field, pmin, pmax)
    _row_cbar(axes[3,:], cmap_mag,   mmin, mmax)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
