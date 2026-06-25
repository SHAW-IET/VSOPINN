import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


file_list = [
    '1 baseline.csv',
    '2 data.csv',
    '3 voronoi.csv',
    '4 sensor_opt.csv',
    '5 cvt.csv',
]

row_labels = [
    'Ground Truth',
    'PhyGeoNet',
    '+ Data',
    '+ Voronoi',
    '+ Sensor Opt.',
    '+ CVT (Ours)',
]
row_markers = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

value_cmap = 'coolwarm'
error_cmap = 'hot'

SHOW_ERROR_AS_PERCENT = False


def factor_pairs(n: int):
    pairs = []
    for a in range(2, int(np.sqrt(n)) + 1):
        if n % a == 0:
            pairs.append((a, n // a))
    return pairs


def infer_grid_shape(x_flat: np.ndarray, y_flat: np.ndarray):

    n = len(x_flat)
    pairs = factor_pairs(n)
    if not pairs:
        raise ValueError(f'Cannot infer a two-dimensional grid shape from point count {n} .')

    best_shape = None
    best_score = np.inf
    all_pairs = pairs + [(b, a) for a, b in pairs if a != b]
    for ny, nx in all_pairs:
        try:
            X = x_flat.reshape(ny, nx)
            Y = y_flat.reshape(ny, nx)
        except ValueError:
            continue

        row_dx = np.mean(np.abs(np.diff(X, axis=1)))
        row_dy = np.mean(np.abs(np.diff(Y, axis=1)))
        col_dx = np.mean(np.abs(np.diff(X, axis=0)))
        col_dy = np.mean(np.abs(np.diff(Y, axis=0)))
        score = row_dx + row_dy + col_dx + col_dy

        if score < best_score:
            best_score = score
            best_shape = (ny, nx)

    if best_shape is None:
        raise ValueError('Failed to infer the two-dimensional grid shape.')
    return best_shape


def load_data(filename, grid_shape=None):
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    required = ['x', 'y', 'T_true', 'T_pred', 'abs_error_T']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'{filename} is missing columns: {missing}')

    x_flat = df['x'].to_numpy(dtype=float)
    y_flat = df['y'].to_numpy(dtype=float)

    if grid_shape is None:
        grid_shape = infer_grid_shape(x_flat, y_flat)

    ny, nx = grid_shape
    return {
        'X': x_flat.reshape(ny, nx),
        'Y': y_flat.reshape(ny, nx),
        'T_true': df['T_true'].to_numpy(dtype=float).reshape(ny, nx),
        'T_pred': df['T_pred'].to_numpy(dtype=float).reshape(ny, nx),
        'T_err': df['abs_error_T'].to_numpy(dtype=float).reshape(ny, nx),
        'range': (x_flat.min(), x_flat.max(), y_flat.min(), y_flat.max()),
        'shape': (ny, nx),
    }


first_df = pd.read_csv(file_list[0])
first_df.columns = first_df.columns.str.strip()
grid_shape = infer_grid_shape(
    first_df['x'].to_numpy(dtype=float),
    first_df['y'].to_numpy(dtype=float)
)
print(f'Inferred grid shape: ny={grid_shape[0]}, nx={grid_shape[1]}')

datasets = []
print('Loading data...')
for f in file_list:
    try:
        datasets.append(load_data(f, grid_shape=grid_shape))
    except Exception as e:
        print(f'Error loading {f}: {e}')

if not datasets:
    raise ValueError('No data loaded!')


T_val_min = min(np.nanmin(d['T_true']) for d in datasets)
T_val_min = min(T_val_min, min(np.nanmin(d['T_pred']) for d in datasets))
T_val_max = max(np.nanmax(d['T_true']) for d in datasets)
T_val_max = max(T_val_max, max(np.nanmax(d['T_pred']) for d in datasets))


T_true_ref = datasets[0]['T_true']
T_rms = np.sqrt(np.nanmean(T_true_ref ** 2))
if T_rms <= 0:
    raise ValueError('Ground truth RMS temperature is non-positive; cannot normalize error.')

for d in datasets:
    d['T_nerr'] = d['T_err'] / T_rms
    if SHOW_ERROR_AS_PERCENT:
        d['T_nerr_plot'] = d['T_nerr'] * 100.0
    else:
        d['T_nerr_plot'] = d['T_nerr']

T_nerr_max = max(np.nanmax(d['T_nerr_plot']) for d in datasets) * 1.05


for i, d in enumerate(datasets, start=1):
    rel_l2 = np.sqrt(np.nanmean((d['T_pred'] - d['T_true']) ** 2)) / T_rms
    print(f'Model {i} Relative L2 = {rel_l2:.6f}')


x_min, x_max, y_min, y_max = datasets[0]['range']
data_ratio = (y_max - y_min) / (x_max - x_min)
subplot_width = 4.4
subplot_height = subplot_width * data_ratio
fig_width = subplot_width * 2.55
fig_height = max(subplot_height * 7.0, 8.8)

fig = plt.figure(figsize=(fig_width, fig_height))
gs = gridspec.GridSpec(
    6, 2, figure=fig,
    wspace=0.10, hspace=0.04,
    left=0.11, right=0.87, top=0.965, bottom=0.055,
)


def set_ticks(ax, row_idx, col_idx, total_rows=6):
    if row_idx == total_rows - 1:
        ax.set_xlabel('$x$')
    else:
        ax.set_xticklabels([])

    if col_idx == 0:
        ax.set_ylabel('$y$')
    else:
        ax.set_yticklabels([])


levels_val = np.linspace(T_val_min, T_val_max, 120)
levels_err = np.linspace(0.0, T_nerr_max, 120)
err_title = 'Normalized Absolute Error' if not SHOW_ERROR_AS_PERCENT else 'Normalized Absolute Error (%)'

for row in range(6):
    d = datasets[max(0, row - 1)]

    for col in range(2):
        ax = fig.add_subplot(gs[row, col])

        if row == 0:
            if col == 0:
                vals = d['T_true']
                levels = levels_val
                cmap = value_cmap
                vmin, vmax = T_val_min, T_val_max
            else:
                ax.axis('off')
                continue
        else:
            if col == 0:
                vals = d['T_pred']
                levels = levels_val
                cmap = value_cmap
                vmin, vmax = T_val_min, T_val_max
            else:
                vals = d['T_nerr_plot']
                levels = levels_err
                cmap = error_cmap
                vmin, vmax = 0.0, T_nerr_max

        ax.contourf(d['X'], d['Y'], vals, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_aspect('equal')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        set_ticks(ax, row, col)

        if row == 0:
            if col == 0:
                ax.set_title(r'$T$', fontsize=17, fontweight='bold', pad=6)
        elif row == 1:
            if col == 1:
                ax.set_title(err_title, fontsize=17, fontweight='bold', pad=6)

        if col == 0:
            ax.text(-0.30, 0.95, row_markers[row], transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top', ha='right')
            ax.text(-0.30, 0.50, row_labels[row], transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='center', ha='right')


total_h = 0.95 - 0.05
cbar_h = total_h / 2 * 0.72
spacing = total_h / 2
cbar_specs = [
    (T_val_min, T_val_max, value_cmap),
    (0.0, T_nerr_max, error_cmap),
]

for i, (vmin, vmax, cmap) in enumerate(cbar_specs):
    y_pos = 0.95 - (i + 1) * spacing + (spacing - cbar_h) / 2
    cbar_ax = fig.add_axes([0.895, y_pos, 0.018, cbar_h])

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.ax.tick_params(labelsize=12)
    cb.locator = MaxNLocator(nbins=5)
    cb.update_ticks()

    if abs(vmax) < 0.01 or abs(vmax) > 1000:
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()

print('Generating case5_temperature_compact_relative_error.(png/pdf)...')
print(f'RMS(T_true) used for normalization = {T_rms:.6f}')
plt.savefig('case5_temperature_compact_relative_error.png', dpi=900, bbox_inches='tight')
plt.savefig('case5_temperature_compact_relative_error.pdf', bbox_inches='tight')
plt.show()
