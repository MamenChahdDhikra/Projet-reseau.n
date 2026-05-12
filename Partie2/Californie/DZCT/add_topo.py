import pandas as pd
import numpy as np
 
print("Chargement des données...")

df_z1  = pd.read_csv('df_zone1.csv')
df_z2  = pd.read_csv('df_zone2.csv')
topo_z1 = pd.read_csv('CA_Zone1_TOPO2.csv')
topo_z2 = pd.read_csv('CA_Zone2_TOPO2.csv')

print(f"Zone1 : {df_z1.shape}")   
print(f"Zone2 : {df_z2.shape}")   
  

all_sol = pd.concat([topo_z1, topo_z2])

landforms_min = all_sol['landforms'].min()
landforms_max = all_sol['landforms'].max()
elevation_min = all_sol['elevation'].min()
elevation_max = all_sol['elevation'].max()
 

print(f"  landforms_min      : [{landforms_min:.2f},  {landforms_min:.2f}]")
print(f"  elevation_min   : [{elevation_min:.2f}, {elevation_max:.2f}]")

 
topo_z1['landforms']   = (topo_z1['landforms'] - landforms_min) \
                          / (landforms_max- landforms_min)
topo_z1['elevation'] = (topo_z1['elevation'] - elevation_min) \
                          / (elevation_max - elevation_min)
 

topo_z2['landforms']   = (topo_z2['landforms'] - landforms_min) \
                          / (landforms_max- landforms_min)
topo_z2['elevation'] = (topo_z2['elevation'] - elevation_min) \
                          / (elevation_max - elevation_min)
 
 
print("\nFusion...")

cols_sol = ['pixel_id', 'landforms',
            'elevation']

df_z1_final = df_z1.merge(
    topo_z1[cols_sol],
    on='pixel_id',
    how='left'
)

df_z2_final = df_z2.merge(
    topo_z2[cols_sol],
    on='pixel_id',
    how='left'
)
 
print("\nNaN avant remplissage :")
for col in ['landforms', 'elevation']:
    print(f"  Zone1 {col} NaN : "
          f"{df_z1_final[col].isna().sum()}")
    print(f"  Zone2 {col} NaN : "
          f"{df_z2_final[col].isna().sum()}")

# Remplir les NaN par la moyenne
for col in ['landforms', 'elevation']:
    df_z1_final[col].fillna(
        df_z1_final[col].mean(), inplace=True)
    df_z2_final[col].fillna(
        df_z2_final[col].mean(), inplace=True)
 
print(f"\nZone1 final : {df_z1_final.shape}")   
print(f"Zone2 final : {df_z2_final.shape}")  

print(f"\nValeurs normalisées Zone1 :")
for col in ['landforms', 'elevation']:
    print(f"  {col} : "
          f"{df_z1_final[col].min():.3f} → "
          f"{df_z1_final[col].max():.3f} ✅")
 
print("\nSauvegarde...")
df_z1_final.to_csv('df_zone1_with_topologie.csv', index=False)
df_z2_final.to_csv('df_zone2_with_topologie.csv', index=False)

print("✅ df_zone1_with_topo.csv sauvegardé")  
print("✅ df_zone2_with_topo.csv sauvegardé")  