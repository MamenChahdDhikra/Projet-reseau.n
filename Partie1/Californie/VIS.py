 
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from prepa import prepare_all_data, CropDataset
 
print("Chargement des données...")
torch.serialization.add_safe_globals([CropDataset])

if os.path.exists('datasets.pt'):
    datasets = torch.load('datasets.pt', weights_only=False)
    print("✅ datasets.pt chargé !")
else:
    datasets = prepare_all_data()

X_all    = torch.cat([datasets['train'].X,
                      datasets['val'].X,
                      datasets['test'].X], dim=0).numpy()
Y_all    = torch.cat([datasets['train'].Y,
                      datasets['val'].Y,
                      datasets['test'].Y], dim=0).numpy()
mask_all = torch.cat([datasets['train'].mask,
                      datasets['val'].mask,
                      datasets['test'].mask], dim=0).numpy()

Y_train  = datasets['train'].Y.numpy()
Y_val    = datasets['val'].Y.numpy()
Y_test   = datasets['test'].Y.numpy()

print(f"X    : {X_all.shape}")
print(f"Y    : {Y_all.shape}")
print(f"mask : {mask_all.shape}")
print(f"X max: {X_all.max():.4f}  X min: {X_all.min():.4f}")
 
CLASSES = {0:'Grapes', 1:'Rice', 2:'Alfalfa',
           3:'Almonds', 4:'Pistachios', 5:'Others'}

COLORS  = {0:'#7B1FA2', 1:'#0288D1', 2:'#388E3C',
           3:'#F57C00', 4:'#FBC02D', 5:'#9E9E9E'}

BANDS   = ['B2_Blue','B3_Green','B4_Red','B5_RE1',
           'B6_RE2','B7_RE3','B8_NIR','B8A_RE4',
           'B11_SWIR1','B12_SWIR2']

DOY     = np.arange(1, 37) * 10
IDX_NIR = 6
IDX_RED = 2

PAPER = {
    'Grapes':     {'total':2054, 'train':240, 'val':60, 'test':1754},
    'Rice':       {'total':2037, 'train':240, 'val':60, 'test':1737},
    'Alfalfa':    {'total':974,  'train':240, 'val':60, 'test':674},
    'Almonds':    {'total':783,  'train':240, 'val':60, 'test':483},
    'Pistachios': {'total':640,  'train':240, 'val':60, 'test':340},
    'Others':     {'total':3512, 'train':240, 'val':60, 'test':3212},
}

 
def compute_ndvi(X, mask):
    nir               = X[:, :, IDX_NIR].copy()
    red               = X[:, :, IDX_RED].copy()
    denom             = nir + red
    denom[denom == 0] = 1e-6
    ndvi              = (nir - red) / denom
    ndvi[mask == 1]   = np.nan
    return ndvi
 
def plot_table2_verification():
    obtained = {}
    for i, cls in CLASSES.items():
        n_train = int(np.sum(Y_train == i))
        n_val   = int(np.sum(Y_val   == i))
        n_test  = int(np.sum(Y_test  == i))
        obtained[cls] = {
            'total': n_train + n_val + n_test,
            'train': n_train,
            'val':   n_val,
            'test':  n_test
        }

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    fig.suptitle(
        'Distribution des données — Californie',
        fontsize=14, fontweight='bold', y=1.02
    )

    col_labels = ['Classe',
                  'Total\n(Attendu)', 'Total\n(Obtenu)',
                  'Train\n(Attendu)', 'Train\n(Obtenu)',
                  'Val\n(Attendu)',   'Val\n(Obtenu)',
                  'Test\n(Attendu)',  'Test\n(Obtenu)',
                  'Statut']

    rows = []
    for cls in CLASSES.values():
        p  = PAPER[cls]
        o  = obtained[cls]
        ok = (o['total'] == p['total'] and
              o['train'] == p['train'] and
              o['val']   == p['val']   and
              o['test']  == p['test'])
        rows.append([
            cls,
            str(p['total']), str(o['total']),
            str(p['train']), str(o['train']),
            str(p['val']),   str(o['val']),
            str(p['test']),  str(o['test']),
            '✅ Conforme' if ok else '⚠️ Diff'
        ])

    tp_tot = sum(PAPER[c]['total'] for c in PAPER)
    to_tot = sum(obtained[c]['total'] for c in obtained)
    tp_tr  = sum(PAPER[c]['train'] for c in PAPER)
    to_tr  = sum(obtained[c]['train'] for c in obtained)
    tp_v   = sum(PAPER[c]['val']   for c in PAPER)
    to_v   = sum(obtained[c]['val']   for c in obtained)
    tp_te  = sum(PAPER[c]['test']  for c in PAPER)
    to_te  = sum(obtained[c]['test']  for c in obtained)

    rows.append([
        'TOTAL',
        str(tp_tot), str(to_tot),
        str(tp_tr),  str(to_tr),
        str(tp_v),   str(to_v),
        str(tp_te),  str(to_te),
        ' Conforme' if to_tot == tp_tot else ' Diff'
    ])

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#37474F')
        table[0, j].set_text_props(color='white', fontweight='bold')

    colors_list = list(COLORS.values()) + ['#ECEFF1']
    for i in range(len(rows)):
        table[i+1, 0].set_facecolor(colors_list[i])
        table[i+1, 0].set_text_props(
            color='white' if i < 6 else '#37474F',
            fontweight='bold'
        )
        for j in [1, 3, 5, 7]:
            table[i+1, j].set_facecolor('#F5F5F5')
        for j in [2, 4, 6, 8]:
            table[i+1, j].set_facecolor('#E8F5E9')
        if '✅' in rows[i][-1]:
            table[i+1, 9].set_facecolor('#C8E6C9')
            table[i+1, 9].set_text_props(color='#2E7D32', fontweight='bold')
        else:
            table[i+1, 9].set_facecolor('#FFCCBC')
            table[i+1, 9].set_text_props(color='#BF360C', fontweight='bold')

    for j in range(len(col_labels)):
        table[7, j].set_text_props(fontweight='bold')
        table[7, j].set_facecolor('#ECEFF1')

    plt.tight_layout()
    plt.savefig('1_table2_verification.png', dpi=150, bbox_inches='tight')
    print("✅ Sauvegardé : 0_table2_verification.png")
    plt.show()
 
def plot_ndvi_timeseries():
    ndvi = compute_ndvi(X_all, mask_all)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('2. Profils NDVI temporels par culture — Californie',
                 fontsize=14, fontweight='bold')

    for i, cls_name in CLASSES.items():
        idx       = np.where(Y_all == i)[0]
        ndvi_cls  = ndvi[idx, :]
        mean_ndvi = np.nanmean(ndvi_cls, axis=0)

        valid_ratio = np.mean(~np.isnan(ndvi_cls), axis=0)
        mean_ndvi[valid_ratio < 0.2] = np.nan

        x_valid = np.where(~np.isnan(mean_ndvi))[0]
        y_valid = mean_ndvi[~np.isnan(mean_ndvi)]

        if len(x_valid) > 1:
            ax.plot(DOY[x_valid], y_valid,
                    color=COLORS[i], linewidth=2,
                    marker='o', markersize=4,
                    label=cls_name)

            std_ndvi = np.nanstd(ndvi_cls, axis=0)
            ax.fill_between(
                DOY[x_valid],
                y_valid - std_ndvi[x_valid],
                y_valid + std_ndvi[x_valid],
                color=COLORS[i], alpha=0.10
            )

    ax.set_xlabel('Day of Year (DOY)', fontsize=11)
    ax.set_ylabel('Mean NDVI Value', fontsize=11)
    ax.set_title('NDVI time-series profiles — California 2021')
    ax.set_xlim(0, 370)
    ax.set_ylim(-0.1, 1)
    ax.set_xticks(DOY[::3])
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig('2_ndvi_timeseries.png', dpi=150, bbox_inches='tight')
    print("✅ Sauvegardé : 2_ndvi_timeseries.png")
    plt.show()
 
def plot_temporal_patterns():
    bands_to_plot = {
        'B2_Blue'  : (0, '#2196F3'),
        'B4_Red'   : (2, '#F44336'),
        'B8_NIR'   : (6, '#4CAF50'),
        'B11_SWIR1': (8, '#FF9800'),
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('3. Patterns temporels par culture — Californie',
                 fontsize=14, fontweight='bold')
    axes = axes.flatten()

    for idx_cls, cls_name in CLASSES.items():
        ax    = axes[idx_cls]
        idx   = np.where(Y_all == idx_cls)[0]
        X_cls = X_all[idx, :, :]
        m_cls = mask_all[idx, :]

        max_val = 0
        for band_name, (band_idx, color) in bands_to_plot.items():
            vals             = X_cls[:, :, band_idx].copy()
            vals[m_cls == 1] = np.nan
            mean_vals        = np.nanmean(vals, axis=0)

            
            t          = np.arange(len(mean_vals))
            valid_mask = ~np.isnan(mean_vals)
            if valid_mask.sum() > 1:
                 mean_vals_interp = np.interp(
                    t,
                    t[valid_mask],
                    mean_vals[valid_mask]
                )
            else:
                mean_vals_interp = mean_vals

            v = np.nanmax(mean_vals_interp)
            if v > max_val:
                max_val = v

            ax.plot(DOY, mean_vals_interp,
                    color=color, linewidth=1.8,
                    marker='o', markersize=3,
                    label=band_name)

        ax.set_title(cls_name, fontweight='bold',
                     color=COLORS[idx_cls])
        ax.set_xlabel('DOY', fontsize=8)
        ax.set_ylabel('Réflectance', fontsize=8)
        ax.set_xlim(0, 370)
        ax.set_ylim(0, min(max_val * 1.2, 1.0))
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig('3_temporal_patterns.png', dpi=150, bbox_inches='tight')
    print("✅ Sauvegardé : 3_temporal_patterns.png")
    plt.show()
 
def plot_missing_values():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('4. Inspection des données manquantes — Californie',
                 fontsize=14, fontweight='bold')

    ax1          = axes[0]
    missing_rate = mask_all.mean(axis=0) * 100

    ax1.bar(DOY, missing_rate,
            color=['#EF5350' if r > 30 else '#42A5F5'
                   for r in missing_rate],
            alpha=0.85)
    ax1.axhline(y=missing_rate.mean(),
                color='red', linestyle='--', linewidth=1.5,
                label=f'Moyenne: {missing_rate.mean():.1f}%')
    ax1.set_xlabel('Day of Year (DOY)')
    ax1.set_ylabel('% données manquantes')
    ax1.set_title('Taux de manquants par pas de temps')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticks(DOY[::6])

    ax2              = axes[1]
    missing_by_class = []
    for i in range(6):
        idx  = np.where(Y_all == i)[0]
        rate = mask_all[idx, :].mean() * 100
        missing_by_class.append(rate)

    bars2 = ax2.bar(list(CLASSES.values()),
                    missing_by_class,
                    color=list(COLORS.values()),
                    alpha=0.85)
    for bar, rate in zip(bars2, missing_by_class):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f'{rate:.1f}%',
                 ha='center', va='bottom', fontsize=9)

    ax2.set_ylabel('% données manquantes')
    ax2.set_title('Taux de manquants par classe')
    ax2.set_xticklabels(list(CLASSES.values()),
                        rotation=20, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, max(missing_by_class) * 1.3)

    plt.tight_layout()
    plt.savefig('4_missing_values.png', dpi=150, bbox_inches='tight')
    print("✅ Sauvegardé : 4_missing_values.png")
    plt.show()
 
def plot_noise_inspection():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('5. Inspection du bruit — Californie',
                 fontsize=14, fontweight='bold')

    ax1 = axes[0]
    for band_idx, band_name in enumerate(BANDS):
        vals = X_all[:, :, band_idx].flatten()
        vals = vals[vals > 0]
        ax1.hist(vals, bins=50, alpha=0.4,
                 label=band_name, density=True)
    ax1.set_xlabel('Réflectance (0–1)')
    ax1.set_ylabel('Densité')
    ax1.set_title('Distribution des réflectances\n(toutes bandes)')
    ax1.set_xlim(0, 1)
    ax1.legend(fontsize=6, ncol=2)
    ax1.grid(alpha=0.3)

    ax2      = axes[1]
    outliers = []
    for band_idx in range(len(BANDS)):
        vals  = X_all[:, :, band_idx].flatten()
        vals  = vals[vals > 0]
        n_out = np.sum((vals < 0) | (vals > 1))
        outliers.append(n_out)

    bars = ax2.bar(BANDS, outliers,
                   color='#EF5350', alpha=0.85)
    for bar, n in zip(bars, outliers):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.1,
                 str(n), ha='center', va='bottom', fontsize=9)

    ax2.set_xticklabels(BANDS, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Nombre de valeurs aberrantes (hors [0,1])')
    ax2.set_title('Valeurs hors [0, 1] par bande')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('5_noise_inspection.png', dpi=150, bbox_inches='tight')
    print("✅ Sauvegardé : 5_noise_inspection.png")
    plt.show() 
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  DATA EXPLORATION — MCTNet — Californie")
    print("="*55)

    print("\n[1/5] Vérification Table 2 du papier...")
    plot_table2_verification()



    print("\n[2/5] Séries temporelles NDVI...")
    plot_ndvi_timeseries()

    print("\n[3/5] Patterns temporels...")
    plot_temporal_patterns()

    print("\n[4/5] Données manquantes...")
    plot_missing_values()

    print("\n[5/5] Inspection du bruit...")
    plot_noise_inspection()
 