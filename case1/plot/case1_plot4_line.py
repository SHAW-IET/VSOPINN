import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    'VSOPINN',
    'VSOPINN (with Attention)',
    'VSOPINN (with CVT)'
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
        if field in df.columns:
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


fig, ax_line = plt.subplots(figsize=(10, 7))

mid_idx = datasets[0]['X'].shape[1] // 2
y_line = datasets[0]['Y'][:, mid_idx]
u_true_line = datasets[0]['u_true'][:, mid_idx]


ax_line.plot(y_line, u_true_line, 'k-', linewidth=2.8, label='Ground Truth')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
styles = ['--', '--', '-.', '-.', ':', '-']

all_y = [u_true_line.copy()]

for i in range(len(datasets)):
    u_pred_line = datasets[i]['u_pred'][:, mid_idx]
    all_y.append(u_pred_line.copy())
    ax_line.plot(y_line, u_pred_line, color=colors[i], linestyle=styles[i],
                 linewidth=2.0, label=model_names[i])


ymin = np.nanmin(np.concatenate([np.ravel(v) for v in all_y]))
ymax = np.nanmax(np.concatenate([np.ravel(v) for v in all_y]))
yr = ymax - ymin if np.isfinite(ymax - ymin) and (ymax - ymin) > 0 else 1.0
ax_line.set_ylim(ymin - 0.08 * yr, ymax + 0.08 * yr)

ax_line.set_xlabel('$y$')
ax_line.set_ylabel(r'$u$-velocity ($x=0.5$)')
ax_line.set_xlim(0, 1)
ax_line.grid(True, linestyle=':', alpha=0.6)


handles, labels = ax_line.get_legend_handles_labels()

fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05),
           fontsize=18, frameon=False)

plt.tight_layout()


plt.subplots_adjust(bottom=0.25)

print("Saving Fig_LinePlot_Standalone.(jpg/pdf)...")
plt.savefig('Fig_LinePlot_Standalone_tuli.jpg', dpi=1000, bbox_inches='tight')

plt.show()
