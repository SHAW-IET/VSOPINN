

\
\
\
\
\
\
\
\
\


import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import map_coordinates


def set_sci_rcparams(font_size: int = 10) -> None:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.labelsize'] = font_size + 1
    plt.rcParams['xtick.labelsize'] = font_size - 1
    plt.rcParams['ytick.labelsize'] = font_size - 1
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'


def natural_key(path: Path):
    name = path.name
    digits = ''
    for ch in name:
        if ch.isdigit():
            digits += ch
        else:
            break
    return (int(digits) if digits else 10**9, name)


def load_case_csv(
    csv_path: Path,
    grid_res: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    r_inner: float,
    r_outer: float,
) -> dict:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    required = [
        'x', 'y',
        'u_true', 'v_true', 'mag_true',
        'u_pred', 'v_pred', 'mag_pred'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{csv_path.name}] missing columns: {missing}")

    x = df['x'].to_numpy(dtype=float)
    y = df['y'].to_numpy(dtype=float)

    u_true = df['u_true'].to_numpy(dtype=float)
    v_true = df['v_true'].to_numpy(dtype=float)
    mag_true = df['mag_true'].to_numpy(dtype=float)

    u_pred = df['u_pred'].to_numpy(dtype=float)
    v_pred = df['v_pred'].to_numpy(dtype=float)
    mag_pred = df['mag_pred'].to_numpy(dtype=float)


    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)


    err_u = np.abs(u_pred - u_true)
    err_v = np.abs(v_pred - v_true)
    if 'abs_error_mag' in df.columns:
        err_mag = df['abs_error_mag'].to_numpy(dtype=float)
    else:
        err_mag = np.abs(mag_pred - mag_true)


    xi = np.linspace(xlim[0], xlim[1], grid_res)
    yi = np.linspace(ylim[0], ylim[1], grid_res)
    X, Y = np.meshgrid(xi, yi)


    pts = np.column_stack([x, y])
    tri = Delaunay(pts)

    vals_mat = np.column_stack([
        u_true, v_true, mag_true,
        u_pred, v_pred, mag_pred,
        err_u, err_v, err_mag
    ])

    interp = LinearNDInterpolator(tri, vals_mat, fill_value=np.nan)
    q = np.column_stack([X.ravel(), Y.ravel()])
    out = interp(q).reshape(X.shape + (9,))

    data = {
        'csv_path': csv_path,
        'X': X,
        'Y': Y,
        'u_true': out[..., 0],
        'v_true': out[..., 1],
        'mag_true': out[..., 2],
        'u_pred': out[..., 3],
        'v_pred': out[..., 4],
        'mag_pred': out[..., 5],
        'err_u': out[..., 6],
        'err_v': out[..., 7],
        'err_mag': out[..., 8],
        'raw': {
            'r': r,
            'theta': theta,
            'u_true': u_true,
            'v_true': v_true,
            'mag_true': mag_true,
            'u_pred': u_pred,
            'v_pred': v_pred,
            'mag_pred': mag_pred,
            'err_u': err_u,
            'err_v': err_v,
            'err_mag': err_mag,
        }
    }


    Rg = np.sqrt(X * X + Y * Y)
    mask = (Rg < r_inner) | (Rg > r_outer)
    for k in ['u_true', 'v_true', 'mag_true', 'u_pred', 'v_pred', 'mag_pred', 'err_u', 'err_v', 'err_mag']:
        arr = np.array(data[k], copy=True)
        arr[mask] = np.nan
        data[k] = arr

    data['mask'] = mask
    return data


def nanminmax(arr_list: list[np.ndarray]) -> tuple[float, float]:
    stacked = np.array([a for a in arr_list if a is not None], dtype=float)
    return float(np.nanmin(stacked)), float(np.nanmax(stacked))


def compute_global_ranges(datasets: list[dict]) -> dict:
    u_list, v_list, m_list = [], [], []
    for d in datasets:
        u_list.extend([d['u_true'], d['u_pred']])
        v_list.extend([d['v_true'], d['v_pred']])
        m_list.extend([d['mag_true'], d['mag_pred']])
    min_u, max_u = nanminmax(u_list)
    min_v, max_v = nanminmax(v_list)
    min_m, max_m = nanminmax(m_list)

    max_eu = max(float(np.nanmax(d['err_u'])) for d in datasets) * 1.05
    max_ev = max(float(np.nanmax(d['err_v'])) for d in datasets) * 1.05
    max_em = max(float(np.nanmax(d['err_mag'])) for d in datasets) * 1.05

    return {
        'u': (min_u, max_u),
        'v': (min_v, max_v),
        'mag': (min_m, max_m),
        'err_u': (0.0, max_eu),
        'err_v': (0.0, max_ev),
        'err_mag': (0.0, max_em),
    }


def decorate_annulus(ax, r_inner: float, r_outer: float, xlim, ylim) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.add_patch(Circle((0, 0), r_outer, fill=False, linewidth=0.6, edgecolor='k'))
    ax.add_patch(Circle((0, 0), r_inner, fill=False, linewidth=0.6, edgecolor='k'))


def set_compact_ticks(ax, row: int, col: int, nrows: int, ncols: int) -> None:
    if row == nrows - 1:
        ax.set_xlabel('$x$')
    else:
        ax.set_xticklabels([])
    if col == 0:
        ax.set_ylabel('$y$')
    else:
        ax.set_yticklabels([])


def add_colorbar_column(fig, ranges, cmaps, labels, x0, x1, top=0.92, bottom=0.08, pad=0.03):
    n = len(ranges)
    total_h = top - bottom
    bar_h = (total_h - pad * (n - 1)) / n
    for i in range(n):
        y0 = top - (i + 1) * bar_h - i * pad
        cax = fig.add_axes([x0, y0, x1 - x0, bar_h])
        sm = ScalarMappable(norm=Normalize(*ranges[i]), cmap=cmaps[i])
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label(labels[i])
        cb.ax.tick_params(labelsize=8)


def plot_fields_merged(
    datasets: list[dict],
    model_names: list[str],
    out_dir: Path,
    tag: str,
    r_inner: float,
    r_outer: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    dpi: int,
) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)
    ranges = compute_global_ranges(datasets)

    vars_list = ['u', 'v', 'mag']
    var_titles = [r'$u$', r'$v$', r'$||\mathbf{u}||$']
    cmaps_val = ['coolwarm', 'coolwarm', 'coolwarm']
    cmap_err = 'hot'

    nrows = 1 + len(datasets)
    ncols = 6


    row_labels = ['Ground Truth'] + model_names
    row_markers = [f'({chr(ord("a")+i)})' for i in range(nrows)]


    data_ratio = (ylim[1] - ylim[0]) / max(1e-9, (xlim[1] - xlim[0]))
    subplot_width = 2.2
    subplot_height = subplot_width * data_ratio
    fig_width = subplot_width * 7.0
    fig_height = subplot_height * (nrows + 0.5)

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(
        nrows, ncols, figure=fig,
        wspace=0.08, hspace=0.08,
        left=0.10, right=0.88, top=0.95, bottom=0.05
    )

    def set_ticks(ax, row_idx: int, col_idx: int) -> None:
        if row_idx == nrows - 1:
            ax.set_xlabel('$x$')
        else:
            ax.set_xticklabels([])
        if col_idx == 0:
            ax.set_ylabel('$y$')
        else:
            ax.set_yticklabels([])

    for row in range(nrows):
        d = datasets[0] if row == 0 else datasets[row - 1]

        for col in range(ncols):
            var_idx = col // 2
            is_err = (col % 2 == 1)
            var = vars_list[var_idx]

            ax = fig.add_subplot(gs[row, col])


            if row == 0 and is_err:
                ax.axis('off')
                continue

            if row == 0 and not is_err:
                key = f'{var}_true'
                vmin, vmax = ranges[var]
                cmap = cmaps_val[var_idx]
            else:
                if is_err:
                    key = {'u': 'err_u', 'v': 'err_v', 'mag': 'err_mag'}[var]
                    vmin, vmax = ranges[key]
                    cmap = cmap_err
                else:
                    key = f'{var}_pred'
                    vmin, vmax = ranges[var]
                    cmap = cmaps_val[var_idx]

            ax.contourf(d['X'], d['Y'], d[key], levels=120, cmap=cmap, vmin=vmin, vmax=vmax)
            decorate_annulus(ax, r_inner, r_outer, xlim, ylim)
            set_ticks(ax, row, col)


            if row == 0 and not is_err:
                ax.set_title(var_titles[var_idx], fontsize=14, fontweight='bold', pad=8)


            if col == 0:
                ax.text(-0.35, 0.95, row_markers[row], transform=ax.transAxes,
                        fontsize=12, fontweight='bold', va='top', ha='right')
                ax.text(-0.35, 0.50, row_labels[row], transform=ax.transAxes,
                        fontsize=12, fontweight='bold', va='center', ha='right', rotation=0)


    total_h = 0.95 - 0.05
    cbar_h = total_h / 6 * 0.7
    spacing = total_h / 6

    for i in range(6):
        y_pos = 0.95 - (i + 1) * spacing + (spacing - cbar_h) / 2
        cbar_ax = fig.add_axes([0.90, y_pos, 0.015, cbar_h])

        var_idx = i // 2
        is_err = (i % 2 == 1)
        var = vars_list[var_idx]

        if is_err:
            key = {'u': 'err_u', 'v': 'err_v', 'mag': 'err_mag'}[var]
            vmin, vmax = ranges[key]
            cmap = cmap_err
        else:
            vmin, vmax = ranges[var]
            cmap = cmaps_val[var_idx]

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax)

        if abs(vmax) < 0.01 or abs(vmax) > 1000:
            cb.formatter.set_powerlimits((0, 0))

    fig.savefig(out_dir / f'case3_{tag}_merged_uvmag_interleave.jpg', dpi=dpi, bbox_inches='tight')
    fig.savefig(out_dir / f'case3_{tag}_merged_uvmag_interleave.pdf', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
def sample_bilinear_on_uniform_grid(Z: np.ndarray, x: np.ndarray, y: np.ndarray,
                                   xlim: tuple[float, float], ylim: tuple[float, float]) -> np.ndarray:

    ny, nx = Z.shape
    col = (x - xlim[0]) / (xlim[1] - xlim[0]) * (nx - 1)
    row = (y - ylim[0]) / (ylim[1] - ylim[0]) * (ny - 1)
    coords = np.vstack([row, col])
    return map_coordinates(Z, coords, order=1, mode='constant', cval=np.nan)


def plot_radial_profile(
    datasets: list[dict],
    model_names: list[str],
    out_dir: Path,
    tag: str,
    r_inner: float,
    r_outer: float,
    theta_deg: float,
    theta_window_deg: float,
    nbins: int,
    dpi: int,
    value: str = 'mag',
    xlim: tuple[float, float] = (-1.0, 1.0),
    ylim: tuple[float, float] = (-1.0, 1.0),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    theta0 = math.radians(theta_deg)
    n_samples = int(max(50, nbins))
    eps = 1e-6 * (r_outer - r_inner)
    r_line = np.linspace(r_inner + eps, r_outer - eps, n_samples)
    x_line = r_line * np.cos(theta0)
    y_line = r_line * np.sin(theta0)

    gt_key = f'{value}_true'
    pred_key = f'{value}_pred'

    fig = plt.figure(figsize=(7.0, 3.6))
    ax = fig.add_subplot(111)

    gt_vals = sample_bilinear_on_uniform_grid(datasets[0][gt_key], x_line, y_line, xlim=xlim, ylim=ylim)
    ax.plot(r_line, gt_vals, 'k-', linewidth=2.2, label='Ground Truth')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    styles = ['--', '--', '-.', '-.', ':']

    for i, d in enumerate(datasets):
        v = sample_bilinear_on_uniform_grid(d[pred_key], x_line, y_line, xlim=xlim, ylim=ylim)
        ax.plot(r_line, v, color=colors[i % len(colors)], linestyle=styles[i % len(styles)],
                linewidth=1.5, label=model_names[i])

    ax.set_xlim(r_inner, r_outer)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlabel(r'$r$')
    ylab = {'u': r'$u$', 'v': r'$v$', 'mag': r'$||\mathbf{u}||$'}[value]
    ax.set_ylabel(ylab)
    ax.set_title(f'Radial Profile at $\\theta={theta_deg:.0f}^\\circ$ (grid-sampled)',
                 fontsize=11, fontweight='bold')
    ax.legend(ncol=3, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.22), frameon=False)

    fig.savefig(out_dir / f'case3_{tag}_radial_{value}_theta{int(theta_deg)}.jpg', dpi=dpi, bbox_inches='tight')
    fig.savefig(out_dir / f'case3_{tag}_radial_{value}_theta{int(theta_deg)}.pdf', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
def main():
    parser = argparse.ArgumentParser(description="Case3 annulus plotting (merged figure, no pressure p).")
    parser.add_argument('--data_dir', type=str, default='.', help='Directory containing csv files.')
    parser.add_argument('--pattern', type=str, default='*.csv', help='Glob pattern to find csvs.')
    parser.add_argument('--csvs', type=str, nargs='*', default=None, help='Explicit list of csv files (overrides pattern).')
    parser.add_argument('--tag', type=str, default='Re350', help='Tag used in output filenames.')
    parser.add_argument('--out_dir', type=str, default='fig_case3', help='Output directory.')
    parser.add_argument('--grid_res', type=int, default=400, help='Grid resolution for interpolation (>=300 recommended).')
    parser.add_argument('--xlim', type=float, nargs=2, default=[-1.0, 1.0], help='x limits.')
    parser.add_argument('--ylim', type=float, nargs=2, default=[-1.0, 1.0], help='y limits.')
    parser.add_argument('--r_inner', type=float, default=0.5, help='Inner radius (hole).')
    parser.add_argument('--r_outer', type=float, default=1.0, help='Outer radius.')
    parser.add_argument('--dpi', type=int, default=1000, help='Save dpi for PNG.')


    parser.add_argument('--make_profile', action='store_true', help='Also generate radial profile figure.')
    parser.add_argument('--theta_deg', type=float, default=0.0, help='Radial line angle in degrees.')
    parser.add_argument('--theta_window_deg', type=float, default=2.0, help='Angular window half-width in degrees.')
    parser.add_argument('--profile_bins', type=int, default=220, help='Number of radial bins.')
    parser.add_argument('--profile_value', type=str, default='mag', choices=['u', 'v', 'mag'],
                        help='Which quantity to plot in radial profile.')

    args = parser.parse_args()
    set_sci_rcparams(font_size=10)

    data_dir = Path(args.data_dir)
    if args.csvs:
        csv_paths = [Path(p) if Path(p).is_absolute() else (data_dir / p) for p in args.csvs]
    else:
        csv_paths = sorted(data_dir.glob(args.pattern), key=natural_key)

    if len(csv_paths) != 5:
        raise ValueError(f"Expected 5 csv files for 5 networks, but found {len(csv_paths)}: {[p.name for p in csv_paths]}")

    model_names = [
        'PhyGeoNet',
        '+ Data',
        '+ Voronoi',
        'VOSPINN',
        'VSOPINN(with CVT)',
    ]

    datasets = []
    for p in csv_paths:
        datasets.append(load_case_csv(
            csv_path=p,
            grid_res=args.grid_res,
            xlim=(args.xlim[0], args.xlim[1]),
            ylim=(args.ylim[0], args.ylim[1]),
            r_inner=args.r_inner,
            r_outer=args.r_outer
        ))

    out_dir = Path(args.out_dir)
    plot_fields_merged(
        datasets=datasets,
        model_names=model_names,
        out_dir=out_dir,
        tag=args.tag,
        r_inner=args.r_inner,
        r_outer=args.r_outer,
        xlim=(args.xlim[0], args.xlim[1]),
        ylim=(args.ylim[0], args.ylim[1]),
        dpi=args.dpi
    )

    if args.make_profile:
        plot_radial_profile(
            datasets=datasets,
            model_names=model_names,
            out_dir=out_dir,
            tag=args.tag,
            r_inner=args.r_inner,
            r_outer=args.r_outer,
            theta_deg=args.theta_deg,
            theta_window_deg=args.theta_window_deg,
            nbins=args.profile_bins,
            dpi=args.dpi,
            value=args.profile_value,
            xlim=(args.xlim[0], args.xlim[1]),
            ylim=(args.ylim[0], args.ylim[1]),
        )


if __name__ == '__main__':
    main()


