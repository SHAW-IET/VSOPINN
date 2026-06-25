from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


BASE_DIR = Path(__file__).resolve().parent
FULL_FIELD_CSV = BASE_DIR / 'prep_output' / 'full_field_truth.csv'
STRUCTURED_MESH_NPZ = BASE_DIR / 'mesh_output_v2' / 'structured_mesh_v2.npz'
OUT_DIR = BASE_DIR / 'interp_output'
SAVE_CSV = True


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_field(x: np.ndarray, y: np.ndarray, field: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=180)
    m = ax.pcolormesh(x, y, field, shading='auto')
    fig.colorbar(m, ax=ax, label='Temperature (°C)')
    ax.set_xlabel('z / X (m)')
    ax.set_ylabel('r / Y (m)')
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_mask(x: np.ndarray, y: np.ndarray, mask: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=180)
    m = ax.pcolormesh(x, y, mask.astype(float), shading='auto')
    fig.colorbar(m, ax=ax, label='1 = filled from nearest')
    ax.set_xlabel('z / X (m)')
    ax.set_ylabel('r / Y (m)')
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    ensure_dir(OUT_DIR)

    if not FULL_FIELD_CSV.exists():
        raise FileNotFoundError(f'Full-field truth file not found: {FULL_FIELD_CSV}')
    if not STRUCTURED_MESH_NPZ.exists():
        raise FileNotFoundError(f'Structured mesh file not found: {STRUCTURED_MESH_NPZ}')

    df = pd.read_csv(FULL_FIELD_CSV)
    mesh = np.load(STRUCTURED_MESH_NPZ)

    x = mesh['x']
    y = mesh['y']

    pts = df[['X_m', 'Y_m']].to_numpy(dtype=float)
    temp = df['Temperature_C'].to_numpy(dtype=float)

    t_linear = griddata(points=pts, values=temp, xi=(x, y), method='linear')
    t_nearest = griddata(points=pts, values=temp, xi=(x, y), method='nearest')

    nearest_fill_mask = np.isnan(t_linear)
    t_ref = np.where(nearest_fill_mask, t_nearest, t_linear)


    t_ref[0, :] = mesh['bottom_T']
    t_ref[-1, :] = mesh['top_T']
    t_ref[:, 0] = mesh['left_T']
    t_ref[:, -1] = mesh['right_T']


    t_ref[0, 0] = mesh['bottom_T'][0]
    t_ref[0, -1] = mesh['bottom_T'][-1]
    t_ref[-1, 0] = mesh['top_T'][0]
    t_ref[-1, -1] = mesh['top_T'][-1]

    summary = {
        'full_field_csv': str(FULL_FIELD_CSV),
        'structured_mesh_npz': str(STRUCTURED_MESH_NPZ),
        'grid_shape': [int(v) for v in t_ref.shape],
        'temperature_min_C': float(np.nanmin(t_ref)),
        'temperature_max_C': float(np.nanmax(t_ref)),
        'linear_nan_count_before_fill': int(np.isnan(t_linear).sum()),
        'filled_by_nearest_count': int(nearest_fill_mask.sum()),
        'final_nan_count': int(np.isnan(t_ref).sum()),
    }

    np.save(OUT_DIR / 'T_ref.npy', t_ref)
    np.savez(
        OUT_DIR / 'temperature_field_on_mesh.npz',
        T_ref=t_ref,
        x=x,
        y=y,
        nearest_fill_mask=nearest_fill_mask.astype(np.uint8),
    )

    if SAVE_CSV:
        flat = pd.DataFrame({
            'grid_i': np.repeat(np.arange(t_ref.shape[0]), t_ref.shape[1]),
            'grid_j': np.tile(np.arange(t_ref.shape[1]), t_ref.shape[0]),
            'X_m': x.ravel(order='C'),
            'Y_m': y.ravel(order='C'),
            'Temperature_C': t_ref.ravel(order='C'),
            'filled_by_nearest': nearest_fill_mask.astype(np.uint8).ravel(order='C'),
        })
        flat.to_csv(OUT_DIR / 'temperature_field_on_mesh.csv', index=False)

    (OUT_DIR / 'interp_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    plot_field(x, y, t_ref, 'Structured temperature field T_ref', OUT_DIR / 'T_ref_on_mesh.png')
    plot_field(x, y, t_linear, 'Linear interpolation before hole filling', OUT_DIR / 'T_linear_before_fill.png')
    plot_mask(x, y, nearest_fill_mask, 'Locations filled by nearest interpolation', OUT_DIR / 'nearest_fill_mask.png')

    print('Interpolation completed. Output directory:')
    print(OUT_DIR.resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
