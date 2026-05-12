import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

soil_z1  = pd.read_csv('df_zone1_with_soil.csv')
soil_z2  = pd.read_csv('df_zone2_with_soil.csv')
topo_z1  = pd.read_csv('df_zone1_with_topologie.csv')
topo_z2  = pd.read_csv('df_zone2_with_topologie.csv')
clim_z1  = pd.read_csv('df_zone1_with_clim.csv')
clim_z2  = pd.read_csv('df_zone2_with_clim.csv')

df_soil  = pd.concat([soil_z1, soil_z2], ignore_index=True)
df_topo  = pd.concat([topo_z1, topo_z2], ignore_index=True)
df_clim  = pd.concat([clim_z1, clim_z2], ignore_index=True)

# Moyenne climat sur 36 périodes
temp_cols = [f'T{t:02d}_temp_norm' for t in range(1, 37)]
prec_cols = [f'T{t:02d}_prec_norm' for t in range(1, 37)]
dew_cols  = [f'T{t:02d}_dew_norm'  for t in range(1, 37)]

df_clim['temp_mean'] = df_clim[temp_cols].mean(axis=1)
df_clim['prec_mean'] = df_clim[prec_cols].mean(axis=1)
df_clim['dew_mean']  = df_clim[dew_cols].mean(axis=1)

# Fusion des 3 sources sur pixel_id
df = df_soil[['pixel_id','class_name',
              'soil_ph_norm','soil_clay_norm','soil_org_norm']].merge(
     df_topo[['pixel_id','elevation','landforms']], on='pixel_id', how='left').merge(
     df_clim[['pixel_id','temp_mean','prec_mean','dew_mean']], on='pixel_id', how='left')

CLASSES    = ['Grapes','Rice','Alfalfa','Almonds','Pistachios','Others']
COLORS_CLS = ['#7B1FA2','#0288D1','#388E3C','#F57C00','#FBC02D','#9E9E9E']

VARIABLES = ['soil_ph_norm', 'soil_clay_norm', 'soil_org_norm',
             'elevation',    'landforms',
             'temp_mean',    'prec_mean',       'dew_mean']

COL_LABELS = ['Classe',
              'pH (norm)', 'Clay (norm)', 'Org C (norm)',
              'Elevation', 'Landforms',
              'Temp (moy)', 'Prec (moy)', 'Dew (moy)']

rows = []
for cls in CLASSES:
    sub = df[df['class_name'] == cls]
    row = [cls]
    for v in VARIABLES:
        mn = sub[v].min()
        mx = sub[v].max()
        row.append(f'{mn:.2f}–{mx:.2f}')
    rows.append(row)

fig, ax = plt.subplots(figsize=(20, 4))
ax.axis('off')
fig.suptitle(
    'Plage de valeurs des covariables par culture — Californie',
    fontsize=13, fontweight='bold', y=1.02)

col_widths = [0.10] + [0.113] * len(VARIABLES)
table = ax.table(
    cellText=rows,
    colLabels=COL_LABELS,
    loc='center',
    cellLoc='center',
    colWidths=col_widths
)
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 2.5)

for j in range(len(COL_LABELS)):
    table[0, j].set_facecolor('#1565C0')
    table[0, j].set_text_props(color='white', fontweight='bold')

for i, color in enumerate(COLORS_CLS):
    table[i+1, 0].set_facecolor(color)
    table[i+1, 0].set_text_props(color='white', fontweight='bold')
    for j in range(1, len(COL_LABELS)):
        table[i+1, j].set_facecolor('#e8eaf6' if i % 2 == 0 else 'white')

plt.tight_layout()
plt.savefig('CA_covariables_ranges.png', dpi=150, bbox_inches='tight')
print("✅ CA_covariables_ranges.png")
plt.show()