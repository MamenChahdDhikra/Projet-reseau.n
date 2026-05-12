import pandas as pd
import numpy as np
 
print("Chargement des données...")

df_z1   = pd.read_csv('AR_df_zone1.csv')
df_z2   = pd.read_csv('AR_df_zone2.csv')
topo_z1 = pd.read_csv('AR_Zone1_TOP2.csv')
topo_z2 = pd.read_csv('AR_Zone2_TOP2.csv')

print(f"Zone1 : {df_z1.shape}")
print(f"Zone2 : {df_z2.shape}")
print(f"Topo Zone1 : {topo_z1.shape}")
print(f"Topo Zone2 : {topo_z2.shape}")
 
print(f"\nColonnes topo_z1 : {topo_z1.columns.tolist()}")
 
print("\nNormalisation...")

all_topo = pd.concat([topo_z1, topo_z2])

landforms_min = all_topo['landforms'].min()
landforms_max = all_topo['landforms'].max()
elevation_min = all_topo['elevation'].min()
elevation_max = all_topo['elevation'].max()

print(f"  landforms  : [{landforms_min:.2f}, {landforms_max:.2f}]")
print(f"  elevation  : [{elevation_min:.2f}, {elevation_max:.2f}]")
 
topo_z1['landforms_norm'] = (topo_z1['landforms'] - landforms_min) \
                             / (landforms_max - landforms_min)
topo_z1['elevation_norm'] = (topo_z1['elevation'] - elevation_min) \
                             / (elevation_max - elevation_min)
 
topo_z2['landforms_norm'] = (topo_z2['landforms'] - landforms_min) \
                             / (landforms_max - landforms_min)
topo_z2['elevation_norm'] = (topo_z2['elevation'] - elevation_min) \
                             / (elevation_max - elevation_min)

print("\nFusion...")

cols_topo = ['pixel_id', 'landforms_norm', 'elevation_norm']

df_z1_final = df_z1.merge(
    topo_z1[cols_topo],
    on='pixel_id',
    how='left'
)

df_z2_final = df_z2.merge(
    topo_z2[cols_topo],
    on='pixel_id',
    how='left'
)
 
print("\nNaN avant remplissage :")
for col in ['landforms_norm', 'elevation_norm']:
    print(f"  Zone1 {col} NaN : {df_z1_final[col].isna().sum()}")
    print(f"  Zone2 {col} NaN : {df_z2_final[col].isna().sum()}")
 
for col in ['landforms_norm', 'elevation_norm']:
    df_z1_final[col].fillna(df_z1_final[col].mean(), inplace=True)
    df_z2_final[col].fillna(df_z2_final[col].mean(), inplace=True)
 
print(f"\nZone1 final : {df_z1_final.shape}")
print(f"Zone2 final : {df_z2_final.shape}")

print(f"\nValeurs normalisées Zone1 :")
for col in ['landforms_norm', 'elevation_norm']:
    print(f"  {col} : "
          f"{df_z1_final[col].min():.3f} → "
          f"{df_z1_final[col].max():.3f} ✅")

print(f"\nValeurs normalisées Zone2 :")
for col in ['landforms_norm', 'elevation_norm']:
    print(f"  {col} : "
          f"{df_z2_final[col].min():.3f} → "
          f"{df_z2_final[col].max():.3f} ✅")
 
print("\nSauvegarde...")
df_z1_final.to_csv('AR_df_zone1_with_topo.csv', index=False)
df_z2_final.to_csv('AR_df_zone2_with_topo.csv', index=False)

print(f"✅ AR_df_zone1_with_topo.csv sauvegardé — {df_z1_final.shape}")
print(f"✅ AR_df_zone2_with_topo.csv sauvegardé — {df_z2_final.shape}")