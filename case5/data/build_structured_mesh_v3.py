from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from pyMesh import hcubeMesh
from build_structured_mesh import (
    load_edge_csv,
    maybe_reverse_for_pymesh,
    resample_edge,
    close_corners,
    compute_bbox_aspect,
    choose_default_nx,
    mesh_quality_summary,
    save_mesh_npz,
)


USE_HARDCODED_PATHS = True
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = {
    "bottom": str(BASE_DIR / "prep_output" / "boundary_bottom.csv"),
    "right": str(BASE_DIR / "prep_output" / "boundary_right.csv"),
    "top": str(BASE_DIR / "prep_output" / "boundary_top.csv"),
    "left": str(BASE_DIR / "prep_output" / "boundary_left.csv"),
    "ny_list": "29,33,37,41",
    "nx_list": "",
    "window_list": "9,11,15",
    "line_blend_list": "0.10,0.15,0.20,0.30",
    "h": 0.01,
    "tol_mesh": 1e-10,
    "tol_joint": 1e-8,
    "out_dir": str(BASE_DIR / "mesh_output_v3"),
}


@dataclass
class CandidateResult:
    ny: int
    nx: int
    window: int
    line_blend: float
    tol_mesh: float
    resampled_edges: Dict[str, pd.DataFrame]
    mesh: hcubeMesh
    quality: Dict[str, float]

    @property
    def is_valid(self) -> bool:
        return self.quality['J_negative_count'] == 0 and self.quality['J_ho_negative_count'] == 0


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(',') if x.strip()]


def smooth_edge_with_savgol(df: pd.DataFrame, window: int, polyorder: int = 3) -> pd.DataFrame:
    if window % 2 == 0:
        raise ValueError('The Savitzky-Golay window must be odd.')
    if len(df) <= window:
        raise ValueError(f'The number of boundary points {len(df)} must be greater than the smoothing window {window}。')

    x = df['X_m'].to_numpy(dtype=float)
    y = df['Y_m'].to_numpy(dtype=float)

    x_s = savgol_filter(x, window_length=window, polyorder=polyorder, mode='interp')
    y_s = savgol_filter(y, window_length=window, polyorder=polyorder, mode='interp')

    x_s[0], y_s[0] = x[0], y[0]
    x_s[-1], y_s[-1] = x[-1], y[-1]

    out = df.copy()
    out['X_m'] = x_s
    out['Y_m'] = y_s
    return out


def blend_edge_towards_chord(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError('line_blend must be in [0, 1].')
    if alpha == 0.0:
        return df.copy()

    x = df['X_m'].to_numpy(dtype=float)
    y = df['Y_m'].to_numpy(dtype=float)
    n = len(df)
    x_line = np.linspace(x[0], x[-1], n)
    y_line = np.linspace(y[0], y[-1], n)

    out = df.copy()
    out['X_m'] = (1.0 - alpha) * x + alpha * x_line
    out['Y_m'] = (1.0 - alpha) * y + alpha * y_line
    return out


def build_regularized_edges(
    original_edges: Dict[str, pd.DataFrame],
    ny: int,
    nx: int,
    window: int,
    line_blend: float,
) -> Dict[str, pd.DataFrame]:
    edges = {
        'Bottom': blend_edge_towards_chord(smooth_edge_with_savgol(resample_edge(original_edges['Bottom'], nx), window), line_blend),
        'Right': blend_edge_towards_chord(smooth_edge_with_savgol(resample_edge(original_edges['Right'], ny), window), line_blend),
        'Top': blend_edge_towards_chord(smooth_edge_with_savgol(resample_edge(original_edges['Top'], nx), window), line_blend),
        'Left': blend_edge_towards_chord(smooth_edge_with_savgol(resample_edge(original_edges['Left'], ny), window), line_blend),
    }
    close_corners(edges['Bottom'], edges['Right'], edges['Top'], edges['Left'])
    return edges


def evaluate_candidate(
    original_edges: Dict[str, pd.DataFrame],
    ny: int,
    nx: int,
    window: int,
    line_blend: float,
    h: float,
    tol_mesh: float,
    tol_joint: float,
) -> CandidateResult:
    edges = build_regularized_edges(original_edges, ny=ny, nx=nx, window=window, line_blend=line_blend)
    mesh = hcubeMesh(
        edges['Left']['X_m'].to_numpy(),
        edges['Left']['Y_m'].to_numpy(),
        edges['Right']['X_m'].to_numpy(),
        edges['Right']['Y_m'].to_numpy(),
        edges['Bottom']['X_m'].to_numpy(),
        edges['Bottom']['Y_m'].to_numpy(),
        edges['Top']['X_m'].to_numpy(),
        edges['Top']['Y_m'].to_numpy(),
        h=h,
        plotFlag=False,
        saveFlag=False,
        tolMesh=tol_mesh,
        tolJoint=tol_joint,
    )

    quality = mesh_quality_summary(mesh)
    quality['J_ho_negative_count'] = int(np.sum(mesh.J_ho <= 0.0))
    quality['J_ho_min'] = float(np.min(mesh.J_ho))
    quality['J_min_core'] = float(np.min(mesh.J))
    quality['J_abs_min_core'] = float(np.min(np.abs(mesh.J)))
    return CandidateResult(
        ny=ny,
        nx=nx,
        window=window,
        line_blend=line_blend,
        tol_mesh=tol_mesh,
        resampled_edges=edges,
        mesh=mesh,
        quality=quality,
    )


def candidate_sort_key(result: CandidateResult) -> Tuple:


    return (
        0 if result.is_valid else 1,
        -(result.ny * result.nx),
        result.line_blend,
        result.window,
        -min(result.quality['J_min'], result.quality['J_ho_min']),
    )


def plot_edge_regularization(original_edges: Dict[str, pd.DataFrame], regularized_edges: Dict[str, pd.DataFrame], save_path: str | Path):
    colors = {'Bottom': 'tab:green', 'Right': 'tab:red', 'Top': 'tab:blue', 'Left': 'tab:orange'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax0, ax1 = axes

    for name, df in original_edges.items():
        ax0.plot(df['X_m'], df['Y_m'], '.', ms=2.2, color=colors[name], label=name)
    ax0.set_title('Original split edges')
    ax0.set_aspect('equal')
    ax0.set_xlabel('X / z (m)')
    ax0.set_ylabel('Y / r (m)')
    ax0.legend()

    for name, df in regularized_edges.items():
        ax1.plot(df['X_m'], df['Y_m'], '-o', ms=2.0, lw=1.0, color=colors[name], label=f'{name} ({len(df)})')
    ax1.set_title('Regularized edges for pyMesh')
    ax1.set_aspect('equal')
    ax1.set_xlabel('X / z (m)')
    ax1.set_ylabel('Y / r (m)')
    ax1.legend()

    fig.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_mesh_and_jacobian(mesh: hcubeMesh, save_path: str | Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax0, ax1 = axes

    for j in range(mesh.x.shape[1]):
        ax0.plot(mesh.x[:, j], mesh.y[:, j], 'k-', lw=0.45)
    for i in range(mesh.x.shape[0]):
        ax0.plot(mesh.x[i, :], mesh.y[i, :], 'k-', lw=0.45)
    ax0.set_aspect('equal')
    ax0.set_title('Structured physical mesh (v2)')
    ax0.set_xlabel('X / z (m)')
    ax0.set_ylabel('Y / r (m)')

    im = ax1.imshow(mesh.J_ho, origin='lower', aspect='auto')
    ax1.set_title('J_ho on reference grid (v2)')
    ax1.set_xlabel('i')
    ax1.set_ylabel('j')
    fig.colorbar(im, ax=ax1, shrink=0.9)

    fig.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_pymesh_builtin_view_clean(mesh: hcubeMesh, regularized_edges: Dict[str, pd.DataFrame], save_path: str | Path):
\
\
\
\
\

    colors = {'Bottom': 'tab:green', 'Right': 'tab:red', 'Top': 'tab:blue', 'Left': 'tab:orange'}

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2), constrained_layout=True)
    ax0, ax1 = axes


    for j in range(mesh.x.shape[1]):
        ax0.plot(mesh.x[:, j], mesh.y[:, j], color='0.82', lw=0.40, zorder=1)
    for i in range(mesh.x.shape[0]):
        ax0.plot(mesh.x[i, :], mesh.y[i, :], color='0.82', lw=0.40, zorder=1)


    for name, df in regularized_edges.items():
        ax0.plot(
            df['X_m'], df['Y_m'],
            color=colors[name], lw=1.6, alpha=0.98,
            solid_capstyle='round', solid_joinstyle='round', zorder=3
        )

    ax0.set_title('Physics Domain Mesh')
    ax0.set_xlabel(r'$x$')
    ax0.set_ylabel(r'$y$')
    ax0.set_aspect('equal', adjustable='box')


    xi = np.asarray(mesh.xi)
    eta = np.asarray(mesh.eta)
    for j in range(xi.shape[1]):
        ax1.plot(xi[:, j], eta[:, j], color='0.86', lw=0.40, zorder=1)
    for i in range(xi.shape[0]):
        ax1.plot(xi[i, :], eta[i, :], color='0.86', lw=0.40, zorder=1)

    ax1.plot(xi[0, :], eta[0, :], color=colors['Bottom'], lw=1.8, solid_capstyle='round', zorder=3)
    ax1.plot(xi[:, -1], eta[:, -1], color=colors['Right'], lw=1.8, solid_capstyle='round', zorder=3)
    ax1.plot(xi[-1, :], eta[-1, :], color=colors['Top'], lw=1.8, solid_capstyle='round', zorder=3)
    ax1.plot(xi[:, 0], eta[:, 0], color=colors['Left'], lw=1.8, solid_capstyle='round', zorder=3)

    ax1.set_title('Reference Domain Mesh')
    ax1.set_xlabel(r'$\xi$')
    ax1.set_ylabel(r'$\eta$')
    ax1.set_aspect('auto')

    for ax in (ax0, ax1):
        ax.tick_params(pad=3)

    fig.savefig(save_path, dpi=260, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Regularize split boundary edges and search for a pyMesh-compatible structured mesh with positive Jacobians.')
    parser.add_argument('--bottom', type=str, default='prep_output/boundary_bottom.csv')
    parser.add_argument('--right', type=str, default='prep_output/boundary_right.csv')
    parser.add_argument('--top', type=str, default='prep_output/boundary_top.csv')
    parser.add_argument('--left', type=str, default='prep_output/boundary_left.csv')
    parser.add_argument('--ny-list', type=str, default='25,29,33', help='Candidate Left/Right point counts, comma-separated.')
    parser.add_argument('--window-list', type=str, default='9,11,15', help='Savitzky-Golay window list, comma-separated odd integers.')
    parser.add_argument('--line-blend-list', type=str, default='0.10,0.15,0.20,0.30,0.40', help='Blend-to-chord strengths, comma-separated floats.')
    parser.add_argument('--nx-list', type=str, default='', help='Optional explicit Bottom/Top point counts. Leave empty to infer from aspect ratio for each ny.')
    parser.add_argument('--h', type=float, default=0.01)
    parser.add_argument('--tol-mesh', type=float, default=1e-4)
    parser.add_argument('--tol-joint', type=float, default=1e-8)
    parser.add_argument('--out-dir', type=str, default='mesh_output_v2')

    if USE_HARDCODED_PATHS:
        class Args:
            pass
        args = Args()
        for k, v in DEFAULT_CONFIG.items():
            setattr(args, k.replace('-', '_'), v)
        print('Running with built-in script paths:')
        print(json.dumps(DEFAULT_CONFIG, indent=2, ensure_ascii=False))
    else:
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

    ny_list = parse_int_list(args.ny_list)
    window_list = parse_int_list(args.window_list)
    line_blend_list = parse_float_list(args.line_blend_list)
    nx_list = parse_int_list(args.nx_list) if args.nx_list.strip() else []

    tried: List[dict] = []
    best: CandidateResult | None = None

    for idx, ny in enumerate(ny_list):
        nx = nx_list[idx] if idx < len(nx_list) else choose_default_nx(ny, aspect)
        for window in window_list:
            if window >= min(ny, nx):
                continue
            if window % 2 == 0:
                continue
            for line_blend in line_blend_list:
                try:
                    result = evaluate_candidate(
                        original_edges=original_edges,
                        ny=ny,
                        nx=nx,
                        window=window,
                        line_blend=line_blend,
                        h=args.h,
                        tol_mesh=args.tol_mesh,
                        tol_joint=args.tol_joint,
                    )
                    record = {
                        'ny': ny,
                        'nx': nx,
                        'window': window,
                        'line_blend': line_blend,
                        'valid': result.is_valid,
                        'quality': result.quality,
                    }
                    tried.append(record)
                    print(json.dumps(record, ensure_ascii=False))
                    if best is None or candidate_sort_key(result) < candidate_sort_key(best):
                        best = result
                except Exception as exc:
                    record = {
                        'ny': ny,
                        'nx': nx,
                        'window': window,
                        'line_blend': line_blend,
                        'valid': False,
                        'error': str(exc),
                    }
                    tried.append(record)
                    print(json.dumps(record, ensure_ascii=False))

    if best is None:
        raise RuntimeError('No candidate structured mesh was generated successfully.')

    for name, df in best.resampled_edges.items():
        out_df = df.copy()
        out_df.insert(0, 'edge_name', name)
        out_df.to_csv(out_dir / f'regularized_{name.lower()}.csv', index=False)


    _ = hcubeMesh(
        best.resampled_edges['Left']['X_m'].to_numpy(),
        best.resampled_edges['Left']['Y_m'].to_numpy(),
        best.resampled_edges['Right']['X_m'].to_numpy(),
        best.resampled_edges['Right']['Y_m'].to_numpy(),
        best.resampled_edges['Bottom']['X_m'].to_numpy(),
        best.resampled_edges['Bottom']['Y_m'].to_numpy(),
        best.resampled_edges['Top']['X_m'].to_numpy(),
        best.resampled_edges['Top']['Y_m'].to_numpy(),
        h=args.h,
        plotFlag=False,
        saveFlag=False,
        saveDir=str(out_dir / 'pymesh_builtin_view_v2.png'),
        tolMesh=best.tol_mesh,
        tolJoint=args.tol_joint,
    )
    plot_pymesh_builtin_view_clean(best.mesh, best.resampled_edges, out_dir / 'pymesh_builtin_view_v2.png')

    summary = {
        'aspect_ratio': float(aspect),
        'chosen': {
            'ny': best.ny,
            'nx': best.nx,
            'window': best.window,
            'line_blend': best.line_blend,
            'tol_mesh': best.tol_mesh,
            'valid': best.is_valid,
            'quality': best.quality,
        },
        'input_counts': {name: int(len(df)) for name, df in original_edges.items()},
        'regularized_counts': {name: int(len(df)) for name, df in best.resampled_edges.items()},
        'search_trials': tried,
    }

    (out_dir / 'mesh_summary_v2.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    plot_edge_regularization(original_edges, best.resampled_edges, out_dir / 'edges_regularized_check_v2.png')
    plot_mesh_and_jacobian(best.mesh, out_dir / 'mesh_and_jacobian_v2.png')
    save_mesh_npz(best.mesh, best.resampled_edges, out_dir / 'structured_mesh_v2.npz')

    print('\n=== FINAL CHOICE ===')
    print(json.dumps(summary['chosen'], indent=2, ensure_ascii=False))
    print(f'Outputs saved to: {out_dir.resolve()}')


if __name__ == '__main__':
    main()
