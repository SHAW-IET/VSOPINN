import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import griddata


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'


plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 17
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


file_list = [
    '1 no_data_points.csv',
    '2 with_data_points.csv',
    '3 Voronoi.csv',
    '4 optimized_sensors.csv',
    '5 optimized_sensors_attention_decoder.csv',
    '6 optimized_sensors_CVT.csv'
]

model_names = [
    'PhyGeoNet',
    '+ Data',
    '+ Voronoi',
    '+ Sensor Opt.',
    '+ Attn. Dec.',
    '+ CVT (Ours)'
]


def load_data(filename):
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    x = df['x'].values
    y = df['y'].values

    grid_res = 100
    xi = np.linspace(0, 1, grid_res)
    yi = np.linspace(0, 1, grid_res)
    X, Y = np.meshgrid(xi, yi)

    def interp(vals):
        return griddata((x, y), vals, (X, Y), method='linear')

    data = {'X': X, 'Y': Y}
    for field in ['u_true', 'v_true', 'mag_true', 'u_pred', 'v_pred', 'mag_pred']:
        data[field] = interp(df[field].values)

    data['err_u'] = np.abs(data['u_true'] - data['u_pred'])
    data['err_v'] = np.abs(data['v_true'] - data['v_pred'])

    if 'abs_error_mag' in df.columns:
        data['err_mag'] = interp(df['abs_error_mag'].values)
    else:
        data['err_mag'] = np.abs(data['mag_true'] - data['mag_pred'])

    return data


datasets = []
print("Loading data...")
for f in file_list:
    try:
        datasets.append(load_data(f))
    except Exception as e:
        print(f"Error loading {f}: {e}")

if not datasets:
    raise ValueError("No data loaded.")


def get_range(key_list):
    vals = []
    for d in datasets:
        for k in key_list:
            if k in d:
                vals.append(d[k])
    return np.nanmin(vals), np.nanmax(vals)

min_u, max_u = get_range(['u_true', 'u_pred'])
min_v, max_v = get_range(['v_true', 'v_pred'])
min_m, max_m = get_range(['mag_true', 'mag_pred'])

max_err_u = get_range(['err_u'])[1] * 1.05
max_err_v = get_range(['err_v'])[1] * 1.05
max_err_m = get_range(['err_mag'])[1] * 1.05


fig = plt.figure(figsize=(18, 18))


gs = gridspec.GridSpec(
    9, 6, figure=fig,
    height_ratios=[1.05, 1, 1, 1, 1, 1, 1, 0.22, 1.35],
    wspace=0.01, hspace=0.18,
    left=0.060, right=0.885, top=0.965, bottom=0.060
)

TICKS = [0.0, 0.5, 1.0]
FMT = FormatStrFormatter('%.1f')

def set_ticks(ax, row_idx, col_idx, is_last_contour_row=False):

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(TICKS)
    ax.set_yticks(TICKS)
    ax.xaxis.set_major_formatter(FMT)
    ax.yaxis.set_major_formatter(FMT)
    ax.tick_params(pad=6)


    if is_last_contour_row:
        ax.set_xlabel('$x$')

        if col_idx == 0:
            labels = [f"{t:.1f}" for t in TICKS]
            labels[0] = ""
            ax.set_xticklabels(labels)
    else:
        ax.set_xticklabels([])


    if col_idx == 0:
        ax.set_ylabel('$y$')
    else:
        ax.set_yticklabels([])


d0 = datasets[0]
truth_slots = {
    0: ('u_true', min_u, max_u, 'jet'),
    2: ('v_true', min_v, max_v, 'jet'),
    4: ('mag_true', min_m, max_m, 'jet')
}

for col in range(6):
    ax = fig.add_subplot(gs[0, col])
    if col in truth_slots:
        key, vmin, vmax, cmap = truth_slots[col]
        ax.contourf(d0['X'], d0['Y'], d0[key], levels=100, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_aspect('equal')
        set_ticks(ax, 0, col, is_last_contour_row=False)


    else:
        ax.axis('off')


plots_config = [
    ('u_pred', min_u, max_u, 'jet'),
    ('err_u', 0, max_err_u, 'hot'),
    ('v_pred', min_v, max_v, 'jet'),
    ('err_v', 0, max_err_v, 'hot'),
    ('mag_pred', min_m, max_m, 'jet'),
    ('err_mag', 0, max_err_m, 'hot')
]

for i, d in enumerate(datasets):
    row_idx = i + 1
    is_last_contour = (row_idx == 6)

    for col, (key, vmin, vmax, cmap) in enumerate(plots_config):
        ax = fig.add_subplot(gs[row_idx, col])
        ax.contourf(d['X'], d['Y'], d[key], levels=100, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_aspect('equal')
        set_ticks(ax, row_idx, col, is_last_contour_row=is_last_contour)


ax_spacer = fig.add_subplot(gs[7, :])
ax_spacer.axis('off')


ax_line = fig.add_subplot(gs[8, :])

mid_idx = datasets[0]['X'].shape[1] // 2
y_line = datasets[0]['Y'][:, mid_idx]
u_true_line = datasets[0]['u_true'][:, mid_idx]

ax_line.plot(y_line, u_true_line, 'k-', linewidth=2.8, label='Ground Truth')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
styles = ['--', '--', '-.', '-.', ':', '-']

all_y = [u_true_line.copy()]
for i in range(6):
    u_pred_line = datasets[i]['u_pred'][:, mid_idx]
    all_y.append(u_pred_line.copy())
    ax_line.plot(y_line, u_pred_line, color=colors[i], linestyle=styles[i],
                 linewidth=2.0, label=model_names[i])


ymin = np.nanmin(np.concatenate([np.ravel(v) for v in all_y]))
ymax = np.nanmax(np.concatenate([np.ravel(v) for v in all_y]))
yr = ymax - ymin if np.isfinite(ymax - ymin) and (ymax - ymin) > 0 else 1.0
ax_line.set_ylim(ymin - 0.08 * yr, ymax + 0.75 * yr)

ax_line.set_xlabel('$y$')
ax_line.set_ylabel(r'$u$-velocity ($x=0.5$)')


ax_line.legend(
    ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.98),
    frameon=False, framealpha=0.92, facecolor='white', edgecolor='none',
    fontsize=24
)

ax_line.set_xlim(0, 1)
ax_line.grid(True, linestyle=':', alpha=0.6)


cbar_configs = [
    (min_u, max_u, 'jet'), (0, max_err_u, 'OrRd'),
    (min_v, max_v, 'jet'), (0, max_err_v, 'OrRd'),
    (min_m, max_m, 'jet'), (0, max_err_m, 'OrRd')
]


y_top = 0.92
y_bot = 0.26
slot = (y_top - y_bot) / 6.0
bar_h = slot * 0.78

for i in range(6):
    vmin, vmax, cmap = cbar_configs[i]

    y_pos = y_top - (i + 1) * slot + (slot - bar_h) / 2

    cax = fig.add_axes([0.905, y_pos, 0.014, bar_h])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)

    cb.ax.tick_params(labelsize=13)
    if abs(vmax) < 0.01 or abs(vmax) > 1000:
        cb.formatter.set_powerlimits((0, 0))

print("Saving Fig_Compact_Style_useredit.(jpg/pdf)...")
plt.savefig('Fig_Compact_Style_useredit.jpg', dpi=1000, bbox_inches='tight')
plt.savefig('Fig_Compact_Style_useredit.pdf', bbox_inches='tight')
plt.show()
