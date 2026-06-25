from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyMesh import hcubeMesh


@dataclass
class EdgeData:
    name: str
    df: pd.DataFrame


def load_edge_csv(path: str | Path, expected_name: str) -> EdgeData:
    path = Path(path)
    df = pd.read_csv(path)
    required = {'X_m', 'Y_m', 'Temperature_C'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{path.name} is missing columns: {sorted(missing)}')
    if df.empty:
        raise ValueError(f'{path.name} is empty.')
    name_in_file = str(df.iloc[0].get('edge_name', expected_name))
    return EdgeData(name=name_in_file, df=df.copy())


def compute_chainage(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    pts = np.column_stack([x, y])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def maybe_reverse_for_pymesh(edge: EdgeData) -> EdgeData:
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

    df = edge.df.copy().reset_index(drop=True)
    if edge.name in {'Top', 'Left'}:
        df = df.iloc[::-1].reset_index(drop=True)
    return EdgeData(name=edge.name, df=df)


def enforce_monotonic_chainage(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    s = compute_chainage(out['X_m'].to_numpy(), out['Y_m'].to_numpy())
    out['arc_length'] = s

    keep = np.concatenate([[True], np.diff(s) > 1e-15])
    out = out.loc[keep].reset_index(drop=True)
    out['arc_length'] = compute_chainage(out['X_m'].to_numpy(), out['Y_m'].to_numpy())
    return out


def resample_edge(df: pd.DataFrame, n_target: int) -> pd.DataFrame:
    if n_target < 5:
        raise ValueError('The number of resampling points must be at least 5.')
    work = enforce_monotonic_chainage(df)
    s = work['arc_length'].to_numpy(dtype=float)
    x = work['X_m'].to_numpy(dtype=float)
    y = work['Y_m'].to_numpy(dtype=float)
    t = work['Temperature_C'].to_numpy(dtype=float)

    s_new = np.linspace(0.0, s[-1], n_target)
    x_new = np.interp(s_new, s, x)
    y_new = np.interp(s_new, s, y)
    t_new = np.interp(s_new, s, t)

    out = pd.DataFrame({
        'edge_point_id': np.arange(n_target, dtype=int),
        'arc_length': s_new,
        'arc_length_norm': s_new / max(s[-1], 1e-15),
        'X_m': x_new,
        'Y_m': y_new,
        'Temperature_C': t_new,
    })
    return out


def close_corners(bottom: pd.DataFrame, right: pd.DataFrame, top: pd.DataFrame, left: pd.DataFrame):

    bl = bottom.loc[0, ['X_m', 'Y_m', 'Temperature_C']].to_numpy(dtype=float)
    br = bottom.loc[len(bottom) - 1, ['X_m', 'Y_m', 'Temperature_C']].to_numpy(dtype=float)
    tl = top.loc[0, ['X_m', 'Y_m', 'Temperature_C']].to_numpy(dtype=float)
    tr = top.loc[len(top) - 1, ['X_m', 'Y_m', 'Temperature_C']].to_numpy(dtype=float)

    left.loc[0, ['X_m', 'Y_m', 'Temperature_C']] = bl
    left.loc[len(left) - 1, ['X_m', 'Y_m', 'Temperature_C']] = tl
    right.loc[0, ['X_m', 'Y_m', 'Temperature_C']] = br
    right.loc[len(right) - 1, ['X_m', 'Y_m', 'Temperature_C']] = tr

    bottom.loc[0, ['X_m', 'Y_m', 'Temperature_C']] = bl
    bottom.loc[len(bottom) - 1, ['X_m', 'Y_m', 'Temperature_C']] = br
    top.loc[0, ['X_m', 'Y_m', 'Temperature_C']] = tl
    top.loc[len(top) - 1, ['X_m', 'Y_m', 'Temperature_C']] = tr


def compute_bbox_aspect(edges: Dict[str, pd.DataFrame]) -> float:
    all_x = np.concatenate([df['X_m'].to_numpy() for df in edges.values()])
    all_y = np.concatenate([df['Y_m'].to_numpy() for df in edges.values()])
    dx = float(all_x.max() - all_x.min())
    dy = float(all_y.max() - all_y.min())
    return dx / max(dy, 1e-15)


def choose_default_nx(ny: int, aspect: float) -> int:

    nx = int(round((ny - 1) * aspect)) + 1
    nx = max(nx, 65)
    if nx % 2 == 0:
        nx += 1
    return nx


def mesh_quality_summary(mesh: hcubeMesh) -> Dict[str, float]:
    J = mesh.J_ho
    return {
        'J_min': float(np.min(J)),
        'J_max': float(np.max(J)),
        'J_mean': float(np.mean(J)),
        'J_abs_min': float(np.min(np.abs(J))),
        'J_negative_count': int(np.sum(J <= 0.0)),
        'x_min': float(np.min(mesh.x)),
        'x_max': float(np.max(mesh.x)),
        'y_min': float(np.min(mesh.y)),
        'y_max': float(np.max(mesh.y)),
    }


def save_mesh_npz(mesh: hcubeMesh, edges_resampled: Dict[str, pd.DataFrame], save_path: str | Path):
    save_path = Path(save_path)
    np.savez_compressed(
        save_path,
        x=mesh.x,
        y=mesh.y,
        xi=mesh.xi,
        eta=mesh.eta,
        J=mesh.J,
        Jinv=mesh.Jinv,
        dxdxi=mesh.dxdxi,
        dydxi=mesh.dydxi,
        dxdeta=mesh.dxdeta,
        dydeta=mesh.dydeta,
        J_ho=mesh.J_ho,
        Jinv_ho=mesh.Jinv_ho,
        dxdxi_ho=mesh.dxdxi_ho,
        dydxi_ho=mesh.dydxi_ho,
        dxdeta_ho=mesh.dxdeta_ho,
        dydeta_ho=mesh.dydeta_ho,
        bottom_x=edges_resampled['Bottom']['X_m'].to_numpy(),
        bottom_y=edges_resampled['Bottom']['Y_m'].to_numpy(),
        bottom_T=edges_resampled['Bottom']['Temperature_C'].to_numpy(),
        right_x=edges_resampled['Right']['X_m'].to_numpy(),
        right_y=edges_resampled['Right']['Y_m'].to_numpy(),
        right_T=edges_resampled['Right']['Temperature_C'].to_numpy(),
        top_x=edges_resampled['Top']['X_m'].to_numpy(),
        top_y=edges_resampled['Top']['Y_m'].to_numpy(),
        top_T=edges_resampled['Top']['Temperature_C'].to_numpy(),
        left_x=edges_resampled['Left']['X_m'].to_numpy(),
        left_y=edges_resampled['Left']['Y_m'].to_numpy(),
        left_T=edges_resampled['Left']['Temperature_C'].to_numpy(),
    )


def plot_edges(original_edges: Dict[str, pd.DataFrame], resampled_edges: Dict[str, pd.DataFrame], save_path: str | Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax0, ax1 = axes
    colors = {'Bottom': 'tab:green', 'Right': 'tab:red', 'Top': 'tab:blue', 'Left': 'tab:orange'}

    for name, df in original_edges.items():
        ax0.plot(df['X_m'], df['Y_m'], '.', ms=2.5, color=colors[name], label=name)
    ax0.set_title('Original split edges')
    ax0.set_aspect('equal')
    ax0.set_xlabel('X / z (m)')
    ax0.set_ylabel('Y / r (m)')
    ax0.legend()

    for name, df in resampled_edges.items():
        ax1.plot(df['X_m'], df['Y_m'], '-o', ms=2.5, lw=1.0, color=colors[name], label=f'{name} ({len(df)})')
    ax1.set_title('Resampled edges for pyMesh')
    ax1.set_aspect('equal')
    ax1.set_xlabel('X / z (m)')
    ax1.set_ylabel('Y / r (m)')
    ax1.legend()

    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_mesh_and_jacobian(mesh: hcubeMesh, save_path: str | Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax0, ax1 = axes

    for j in range(mesh.x.shape[1]):
        ax0.plot(mesh.x[:, j], mesh.y[:, j], 'k-', lw=0.4)
    for i in range(mesh.x.shape[0]):
        ax0.plot(mesh.x[i, :], mesh.y[i, :], 'k-', lw=0.4)
    ax0.set_aspect('equal')
    ax0.set_title('Structured physical mesh')
    ax0.set_xlabel('X / z (m)')
    ax0.set_ylabel('Y / r (m)')

    im = ax1.imshow(mesh.J_ho, origin='lower', aspect='auto')
    ax1.set_title('J_ho on reference grid')
    ax1.set_xlabel('i')
    ax1.set_ylabel('j')
    fig.colorbar(im, ax=ax1, shrink=0.9)

    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Resample four boundary edges and build a pyMesh-compatible structured mesh.')
    parser.add_argument('--bottom', type=str, default='prep_output/boundary_bottom.csv')
    parser.add_argument('--right', type=str, default='prep_output/boundary_right.csv')
    parser.add_argument('--top', type=str, default='prep_output/boundary_top.csv')
    parser.add_argument('--left', type=str, default='prep_output/boundary_left.csv')
    parser.add_argument('--ny', type=int, default=65, help='Target number of points on Left/Right boundaries.')
    parser.add_argument('--nx', type=int, default=None, help='Target number of points on Bottom/Top boundaries. Default: auto from aspect ratio.')
    parser.add_argument('--h', type=float, default=0.01, help='Reference-grid spacing used by pyMesh.')
    parser.add_argument('--tol-mesh', type=float, default=1e-10)
    parser.add_argument('--tol-joint', type=float, default=1e-8)
    parser.add_argument('--out-dir', type=str, default='mesh_output')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    original_edges = {
        'Bottom': maybe_reverse_for_pymesh(load_edge_csv(args.bottom, 'Bottom')).df,
        'Right': maybe_reverse_for_pymesh(load_edge_csv(args.right, 'Right')).df,
        'Top': maybe_reverse_for_pymesh(load_edge_csv(args.top, 'Top')).df,
        'Left': maybe_reverse_for_pymesh(load_edge_csv(args.left, 'Left')).df,
    }

    aspect = compute_bbox_aspect(original_edges)
    nx = args.nx if args.nx is not None else choose_default_nx(args.ny, aspect)

    resampled_edges = {
        'Bottom': resample_edge(original_edges['Bottom'], nx),
        'Right': resample_edge(original_edges['Right'], args.ny),
        'Top': resample_edge(original_edges['Top'], nx),
        'Left': resample_edge(original_edges['Left'], args.ny),
    }
    close_corners(resampled_edges['Bottom'], resampled_edges['Right'], resampled_edges['Top'], resampled_edges['Left'])


    for name, df in resampled_edges.items():
        df = df.copy()
        df.insert(0, 'edge_name', name)
        df.to_csv(out_dir / f'resampled_{name.lower()}.csv', index=False)


    mesh = hcubeMesh(
        resampled_edges['Left']['X_m'].to_numpy(),
        resampled_edges['Left']['Y_m'].to_numpy(),
        resampled_edges['Right']['X_m'].to_numpy(),
        resampled_edges['Right']['Y_m'].to_numpy(),
        resampled_edges['Bottom']['X_m'].to_numpy(),
        resampled_edges['Bottom']['Y_m'].to_numpy(),
        resampled_edges['Top']['X_m'].to_numpy(),
        resampled_edges['Top']['Y_m'].to_numpy(),
        h=args.h,
        plotFlag=False,
        saveFlag=True,
        saveDir=str(out_dir / 'pymesh_builtin_view.png'),
        tolMesh=args.tol_mesh,
        tolJoint=args.tol_joint,
    )

    quality = mesh_quality_summary(mesh)
    summary = {
        'ny': int(args.ny),
        'nx': int(nx),
        'h': float(args.h),
        'bbox_aspect_ratio': float(aspect),
        'input_counts': {name: int(len(df)) for name, df in original_edges.items()},
        'resampled_counts': {name: int(len(df)) for name, df in resampled_edges.items()},
        'quality': quality,
        'corners': {
            'bottom_left': resampled_edges['Bottom'].loc[0, ['X_m', 'Y_m', 'Temperature_C']].to_dict(),
            'bottom_right': resampled_edges['Bottom'].loc[len(resampled_edges['Bottom']) - 1, ['X_m', 'Y_m', 'Temperature_C']].to_dict(),
            'top_left': resampled_edges['Top'].loc[0, ['X_m', 'Y_m', 'Temperature_C']].to_dict(),
            'top_right': resampled_edges['Top'].loc[len(resampled_edges['Top']) - 1, ['X_m', 'Y_m', 'Temperature_C']].to_dict(),
        },
    }
    (out_dir / 'mesh_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    plot_edges(original_edges, resampled_edges, out_dir / 'edges_resampled_check.png')
    plot_mesh_and_jacobian(mesh, out_dir / 'mesh_and_jacobian.png')
    save_mesh_npz(mesh, resampled_edges, out_dir / 'structured_mesh.npz')

    print(f'Completed. Outputs saved to: {out_dir.resolve()}')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
