import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
import matplotlib.cm as cm


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
    '1 no_data_points.csv',
    '2 with_data_points.csv',
    '3 hard_Voronoi.csv',
    '4 optimized_sensors.csv',
    '5 optimized_sensors_CVT.csv'
]


row_labels = [
    'Ground Truth',
    'PhyGeoNet',
    '+ Data',
    '+ Voronoi',
    '+ Sensor Opt.',
    '+ CVT (Ours)'
]


row_markers = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']


def load_data(filename):
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    x = df['x'].values;
    y = df['y'].values


    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()


    aspect_ratio = (y_max - y_min) / (x_max - x_min)
    grid_res_x = 150
    grid_res_y = int(150 * aspect_ratio)

    xi = np.linspace(x_min, x_max, grid_res_x)
    yi = np.linspace(y_min, y_max, grid_res_y)
    X, Y = np.meshgrid(xi, yi)

    def interp(vals):

        return griddata((x, y), vals, (X, Y), method='linear')

    data = {'X': X, 'Y': Y, 'range': (x_min, x_max, y_min, y_max)}


    for var in ['u', 'v', 'mag']:
        data[f'{var}_true'] = interp(df[f'{var}_true'].values)
        data[f'{var}_pred'] = interp(df[f'{var}_pred'].values)


        if var == 'mag' and 'abs_error_mag' in df.columns:
            data[f'{var}_err'] = interp(df['abs_error_mag'].values)
        else:
            data[f'{var}_err'] = np.abs(data[f'{var}_true'] - data[f'{var}_pred'])

    return data


datasets = []
print("Loading data...")
for f in file_list:
    try:
        datasets.append(load_data(f))
    except Exception as e:
        print(f"Error loading {f}: {e}")

if not datasets: raise ValueError("No data loaded!")


def get_global_limits(var_name):

    vmin = min(np.nanmin(d[f'{var_name}_true']) for d in datasets)
    vmin = min(vmin, min(np.nanmin(d[f'{var_name}_pred']) for d in datasets))
    vmax = max(np.nanmax(d[f'{var_name}_true']) for d in datasets)
    vmax = max(vmax, max(np.nanmax(d[f'{var_name}_pred']) for d in datasets))


    emax = max(np.nanmax(d[f'{var_name}_err']) for d in datasets)

    return (vmin, vmax), (0, emax * 1.05)


limits = {}
vars_list = ['u', 'v', 'mag']
var_titles = [r'$u$', r'$v$', r'$||\mathbf{u}||$']
cmaps = ['coolwarm', 'coolwarm', 'coolwarm']

for var in vars_list:
    limits[f'{var}_val'], limits[f'{var}_err'] = get_global_limits(var)


data_ratio = (datasets[0]['range'][3] - datasets[0]['range'][2]) / (datasets[0]['range'][1] - datasets[0]['range'][0])

subplot_width = 2.6
subplot_height = subplot_width * data_ratio
fig_width = subplot_width * 7
fig_height = subplot_height * 6.5

fig = plt.figure(figsize=(fig_width, fig_height))


gs = gridspec.GridSpec(6, 6, figure=fig,
                       wspace=0.04, hspace=0.04,
                       left=0.085, right=0.885, top=0.965, bottom=0.06)


def set_ticks(ax, row_idx, col_idx, total_rows=6):

    if row_idx == total_rows - 1:
        ax.set_xlabel('$x$')
    else:
        ax.set_xticklabels([])


    if col_idx == 0:
        ax.set_ylabel('$y$')
    else:
        ax.set_yticklabels([])


for row in range(6):
    d = datasets[max(0, row - 1)]

    for col in range(6):

        var_idx = col // 2
        is_err = (col % 2 != 0)
        var_name = vars_list[var_idx]

        ax = fig.add_subplot(gs[row, col])


        if row == 0:
            if is_err:

                ax.axis('off')
                continue
            else:

                data_key = f'{var_name}_true'
                vmin, vmax = limits[f'{var_name}_val']
                cmap = cmaps[var_idx]


        else:
            if is_err:
                data_key = f'{var_name}_err'
                vmin, vmax = limits[f'{var_name}_err']
                cmap = 'hot'
            else:
                data_key = f'{var_name}_pred'
                vmin, vmax = limits[f'{var_name}_val']
                cmap = cmaps[var_idx]


        im = ax.contourf(d['X'], d['Y'], d[data_key], levels=100, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_aspect('equal')


        set_ticks(ax, row, col)


        if row == 0 and not is_err:
            ax.set_title(var_titles[var_idx], fontsize=17, fontweight='bold', pad=6)


        if col == 0:

            ax.text(-0.28, 0.95, row_markers[row], transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top', ha='right')

            ax.text(-0.28, 0.5, row_labels[row], transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='center', ha='right', rotation=0)


total_h = 0.95 - 0.05
cbar_h = total_h / 6 * 0.7
spacing = total_h / 6

for i in range(6):


    y_pos = 0.95 - (i + 1) * spacing + (spacing - cbar_h) / 2

    cbar_ax = fig.add_axes([0.905, y_pos, 0.014, cbar_h])


    var_idx = i // 2
    is_err = (i % 2 != 0)
    var_name = vars_list[var_idx]

    if is_err:
        vmin, vmax = limits[f'{var_name}_err']
        cmap = 'hot'
    else:
        vmin, vmax = limits[f'{var_name}_val']
        cmap = cmaps[var_idx]

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cb = fig.colorbar(sm, cax=cbar_ax)

    cb.ax.tick_params(labelsize=12)

    if abs(vmax) < 0.01 or abs(vmax) > 1000:
        cb.formatter.set_powerlimits((0, 0))

print("Generating Fig_Case2_Compact_Matrix.jpg...")
plt.savefig('Fig_Case2_Compact_Matrix.jpg', dpi=600, bbox_inches='tight')

print("Done!")
plt.show()
