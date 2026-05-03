"""Generate analysis figures for intra-cell DEM experiment report."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 200,
})

root = Path('models/output/intracell_dem')
baseline = Path('models/output')
out_dir = Path('docs/analysis/intracell_dem')

# ============================================================
# Collect all data
# ============================================================
bundles = ['BASIC', 'BASIC_D10', 'BASIC_PCA6', 'BASIC_D10_STATS']
models_data = {
    'V2 ConvLSTM': {},
    'V4 GNN-TAT': {},
    'V10 Late Fusion': {},
}


def load_pred(path):
    p = np.load(path / 'predictions.npy')
    t = np.load(path / 'targets.npy')
    if p.ndim == 5:
        p = p.squeeze(-1)
    if t.ndim == 5:
        t = t.squeeze(-1)
    return p, t


# BASIC (Paper 4)
models_data['V2 ConvLSTM']['BASIC'] = load_pred(
    baseline / 'V2_Enhanced_Models/map_exports/H12/BASIC/ConvLSTM')
models_data['V4 GNN-TAT']['BASIC'] = load_pred(
    baseline / 'V4_GNN_TAT_Models/map_exports/H12/BASIC/GNN_TAT_GAT')

v10_basic = baseline / 'V10_Late_Fusion/BASIC'
if v10_basic.exists() and (v10_basic / 'predictions.npy').exists():
    models_data['V10 Late Fusion']['BASIC'] = load_pred(v10_basic)

# DEM bundles
for bundle in ['BASIC_D10', 'BASIC_PCA6', 'BASIC_D10_STATS']:
    for model_key, subdir in [('V2 ConvLSTM', 'ConvLSTM'), ('V4 GNN-TAT', 'GNN_TAT_GAT')]:
        bdir = root / subdir / bundle
        runs = sorted(bdir.glob('[0-9]*_*')) if bdir.exists() else []
        if runs:
            models_data[model_key][bundle] = load_pred(runs[-1])
    # V10
    bdir = root / 'Late_Fusion' / bundle
    runs = sorted(bdir.glob('[0-9]*_*')) if bdir.exists() else []
    if runs:
        models_data['V10 Late Fusion'][bundle] = load_pred(runs[-1])


def get_r2(model_name, bundle):
    if bundle in models_data[model_name]:
        p, t = models_data[model_name][bundle]
        return r2_score(t.ravel(), p.ravel())
    return np.nan


def get_rmse(model_name, bundle):
    if bundle in models_data[model_name]:
        p, t = models_data[model_name][bundle]
        return np.sqrt(mean_squared_error(t.ravel(), p.ravel()))
    return np.nan


def get_per_horizon_r2(model_name, bundle):
    if bundle in models_data[model_name]:
        p, t = models_data[model_name][bundle]
        return [r2_score(t[:, h].ravel(), p[:, h].ravel()) for h in range(p.shape[1])]
    return [np.nan] * 12


# ============================================================
# FIGURE 1: R2 and RMSE comparison bars
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
model_names = ['V2 ConvLSTM', 'V4 GNN-TAT', 'V10 Late Fusion']
colors = ['#0072B2', '#E69F00', '#009E73']
bundle_labels = ['BASIC\n(12 feat)', 'BASIC_D10\n(22 feat)',
                 'BASIC_PCA6\n(18 feat)', 'D10_STATS\n(27 feat)']
x = np.arange(len(bundles))
width = 0.25

# R2
ax = axes[0]
for i, (mn, c) in enumerate(zip(model_names, colors)):
    vals = [get_r2(mn, b) for b in bundles]
    bars = ax.bar(x + i * width, vals, width, label=mn, color=c, alpha=0.85, edgecolor='white')
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.008,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
ax.set_ylabel(u'R\u00b2')
ax.set_title(u'R\u00b2 by Feature Bundle and Model')
ax.set_xticks(x + width)
ax.set_xticklabels(bundle_labels, fontsize=9)
ax.legend(fontsize=9, loc='upper right')
ax.set_ylim(0, 0.78)
ax.axhline(y=get_r2('V10 Late Fusion', 'BASIC'), color='#009E73',
           linestyle='--', alpha=0.5, linewidth=1)
ax.text(3.4, get_r2('V10 Late Fusion', 'BASIC') + 0.008,
        'BASIC V10', fontsize=8, color='#009E73', ha='right')
ax.grid(axis='y', alpha=0.3)

# RMSE
ax = axes[1]
for i, (mn, c) in enumerate(zip(model_names, colors)):
    vals = [get_rmse(mn, b) for b in bundles]
    bars = ax.bar(x + i * width, vals, width, label=mn, color=c, alpha=0.85, edgecolor='white')
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
ax.set_ylabel('RMSE (mm)')
ax.set_title('RMSE by Feature Bundle and Model')
ax.set_xticks(x + width)
ax.set_xticklabels(bundle_labels, fontsize=9)
ax.legend(fontsize=9, loc='upper left')
ax.axhline(y=get_rmse('V10 Late Fusion', 'BASIC'), color='#009E73',
           linestyle='--', alpha=0.5, linewidth=1)
ax.text(0.0, get_rmse('V10 Late Fusion', 'BASIC') - 2,
        'BASIC V10', fontsize=8, color='#009E73')
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Intra-Cell DEM Feature Experiment: Performance Comparison',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(out_dir / 'fig_01_r2_rmse_comparison.png')
plt.close()
print('Fig 1 saved')

# ============================================================
# FIGURE 2: Degradation curve (features vs R2)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
bundle_order = ['BASIC', 'BASIC_PCA6', 'BASIC_D10', 'BASIC_D10_STATS']
feat_count = [12, 18, 22, 27]

for mn, c, marker in zip(model_names, colors, ['o', 's', 'D']):
    vals = [get_r2(mn, b) for b in bundle_order]
    ax.plot(feat_count, vals, f'-{marker}', color=c, label=mn,
            markersize=10, linewidth=2.5, markeredgecolor='white', markeredgewidth=1.5)

ax.annotate('BASIC\n(baseline)', xy=(12, 0.666), xytext=(14.5, 0.72),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'), color='gray')
ax.annotate('D10_STATS\n(worst)', xy=(27, 0.351), xytext=(24, 0.28),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'), color='gray')

ax.set_xlabel('Number of Features')
ax.set_ylabel(u'R\u00b2')
ax.set_title(u'Feature Count vs R\u00b2: More DEM Features = Worse Performance',
             fontweight='bold')
ax.legend(fontsize=10)
ax.set_xticks(feat_count)
ax.set_xticklabels(['12\nBASIC', '18\nPCA6', '22\nD10', '27\nD10+STATS'], fontsize=9)
ax.set_ylim(0.25, 0.78)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / 'fig_02_degradation_curve.png')
plt.close()
print('Fig 2 saved')

# ============================================================
# FIGURE 3: Per-horizon R2 heatmap
# ============================================================
fig, ax = plt.subplots(figsize=(13, 4.5))

row_configs = [
    ('V2 D10_STATS', 'V2 ConvLSTM', 'BASIC_D10_STATS'),
    ('V4 D10_STATS', 'V4 GNN-TAT', 'BASIC_D10_STATS'),
    ('V10 D10_STATS', 'V10 Late Fusion', 'BASIC_D10_STATS'),
    ('V10 BASIC (ref)', 'V10 Late Fusion', 'BASIC'),
]

data_matrix = []
row_labels = []
for label, mn, bundle in row_configs:
    vals = get_per_horizon_r2(mn, bundle)
    if not all(np.isnan(v) for v in vals):
        data_matrix.append(vals)
        row_labels.append(label)

data = np.array(data_matrix)
im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=0.75)

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        color = 'white' if data[i, j] < 0.1 else 'black'
        ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                fontsize=9, color=color, fontweight='bold')

ax.set_xticks(range(12))
ax.set_xticklabels([f'H{h}' for h in range(1, 13)])
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=10)
ax.set_title(u'Per-Horizon R\u00b2: BASIC_D10_STATS vs BASIC V10 Reference',
             fontweight='bold')
plt.colorbar(im, ax=ax, label=u'R\u00b2', shrink=0.8)
plt.tight_layout()
plt.savefig(out_dir / 'fig_03_horizon_heatmap.png')
plt.close()
print('Fig 3 saved')

# ============================================================
# FIGURE 4: Degradation percentage
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
dem_bundles = ['BASIC_D10', 'BASIC_PCA6', 'BASIC_D10_STATS']
dem_labels = ['D10\n(+10 deciles)', 'PCA6\n(+6 PCA)', 'D10_STATS\n(+15 DEM)']
x = np.arange(len(dem_bundles))
width = 0.25

for i, (mn, c) in enumerate(zip(model_names, colors)):
    base_r2 = get_r2(mn, 'BASIC')
    pcts = []
    for bundle in dem_bundles:
        r2 = get_r2(mn, bundle)
        pcts.append(100 * (r2 - base_r2) / base_r2 if not np.isnan(r2) else 0)
    bars = ax.bar(x + i * width, pcts, width, label=mn, color=c, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, pcts):
        y_pos = bar.get_height() - 2 if val < -5 else bar.get_height() + 0.5
        va = 'top' if val < -5 else 'bottom'
        fc = 'white' if val < -5 else 'black'
        ax.text(bar.get_x() + bar.get_width() / 2., y_pos,
                f'{val:.1f}%', ha='center', va=va, fontsize=8, fontweight='bold', color=fc)

ax.set_ylabel(u'R\u00b2 Change vs BASIC (%)')
ax.set_title('Performance Degradation by DEM Bundle\n(all negative = DEM features hurt)',
             fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(dem_labels, fontsize=10)
ax.legend(fontsize=9)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(-50, 5)
plt.tight_layout()
plt.savefig(out_dir / 'fig_04_degradation_pct.png')
plt.close()
print('Fig 4 saved')

print('\nAll 4 figures saved to models/intracell_dem/')
