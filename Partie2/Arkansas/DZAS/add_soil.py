 
import pandas as pd
import numpy as np
import json
 
print("Chargement des données Arkansas...")

df_z1  = pd.read_csv('AR_df_zone1.csv')
df_z2  = pd.read_csv('AR_df_zone2.csv')
sol_z1 = pd.read_csv('AR_Zone1_SOL3.csv')
sol_z2 = pd.read_csv('AR_Zone2_SOL3.csv')

print(f"Zone1 Sentinel-2 : {df_z1.shape}")
print(f"Zone2 Sentinel-2 : {df_z2.shape}")
print(f"Zone1 Sol        : {sol_z1.shape}")
print(f"Zone2 Sol        : {sol_z2.shape}")
 
print("\nColonnes SOL Zone1 :", list(sol_z1.columns))
print("Exemple SOL Zone1 :")
print(sol_z1[['lon', 'lat', 'soil_ph',
              'soil_clay', 'soil_organic']].head(3))

print("\nExemple df Zone1 :")
print(df_z1[['pixel_id', 'lon', 'lat']].head(3))
 
print("\nPréparation jointure lon/lat...")

decimals = 4

df_z1['lon_r']  = df_z1['lon'].round(decimals)
df_z1['lat_r']  = df_z1['lat'].round(decimals)
df_z2['lon_r']  = df_z2['lon'].round(decimals)
df_z2['lat_r']  = df_z2['lat'].round(decimals)
sol_z1['lon_r'] = sol_z1['lon'].round(decimals)
sol_z1['lat_r'] = sol_z1['lat'].round(decimals)
sol_z2['lon_r'] = sol_z2['lon'].round(decimals)
sol_z2['lat_r'] = sol_z2['lat'].round(decimals)

 
communs_z1 = set(zip(df_z1['lon_r'], df_z1['lat_r'])) & \
             set(zip(sol_z1['lon_r'], sol_z1['lat_r']))
communs_z2 = set(zip(df_z2['lon_r'], df_z2['lat_r'])) & \
             set(zip(sol_z2['lon_r'], sol_z2['lat_r']))

print(f"  Zone1 pixels en commun : {len(communs_z1)}")
print(f"  Zone2 pixels en commun : {len(communs_z2)}")
 
all_sol  = pd.concat([sol_z1, sol_z2])
ph_min   = all_sol['soil_ph'].min()
ph_max   = all_sol['soil_ph'].max()
clay_min = all_sol['soil_clay'].min()
clay_max = all_sol['soil_clay'].max()
org_min  = all_sol['soil_organic'].min()
org_max  = all_sol['soil_organic'].max()

print(f"  soil_ph      : [{ph_min:.2f}, {ph_max:.2f}]")
print(f"  soil_clay    : [{clay_min:.2f}, {clay_max:.2f}]")
print(f"  soil_organic : [{org_min:.2f}, {org_max:.2f}]")
 
sol_z1['soil_ph_norm']   = (sol_z1['soil_ph'] - ph_min) \
                          / (ph_max - ph_min)
sol_z1['soil_clay_norm'] = (sol_z1['soil_clay'] - clay_min) \
                          / (clay_max - clay_min)
sol_z1['soil_org_norm']  = (sol_z1['soil_organic'] - org_min) \
                          / (org_max - org_min)

 
sol_z2['soil_ph_norm']   = (sol_z2['soil_ph'] - ph_min) \
                          / (ph_max - ph_min)
sol_z2['soil_clay_norm'] = (sol_z2['soil_clay'] - clay_min) \
                          / (clay_max - clay_min)
sol_z2['soil_org_norm']  = (sol_z2['soil_organic'] - org_min) \
                          / (org_max - org_min)
 
print("\nFusion sur lon/lat...")

cols_sol = ['lon_r', 'lat_r',
            'soil_ph_norm',
            'soil_clay_norm',
            'soil_org_norm']

df_z1_final = df_z1.merge(
    sol_z1[cols_sol],
    on=['lon_r', 'lat_r'],
    how='left'
)

df_z2_final = df_z2.merge(
    sol_z2[cols_sol],
    on=['lon_r', 'lat_r'],
    how='left'
)
 
print("\nNaN avant remplissage :")
for col in ['soil_ph_norm', 'soil_clay_norm', 'soil_org_norm']:
    print(f"  Zone1 {col} NaN : "
          f"{df_z1_final[col].isna().sum()}")
    print(f"  Zone2 {col} NaN : "
          f"{df_z2_final[col].isna().sum()}")


for col in ['soil_ph_norm', 'soil_clay_norm', 'soil_org_norm']:
    df_z1_final[col] = df_z1_final[col].fillna(
        df_z1_final[col].mean())
    df_z2_final[col] = df_z2_final[col].fillna(
        df_z2_final[col].mean())
 
df_z1_final = df_z1_final.drop(columns=['lon_r', 'lat_r'])
df_z2_final = df_z2_final.drop(columns=['lon_r', 'lat_r'])
 
print(f"\nZone1 final : {df_z1_final.shape}")  
print(f"Zone2 final : {df_z2_final.shape}")    

print(f"\nValeurs normalisées Zone1 :")
for col in ['soil_ph_norm', 'soil_clay_norm', 'soil_org_norm']:
    print(f"  {col} : "
          f"{df_z1_final[col].min():.3f} → "
          f"{df_z1_final[col].max():.3f} ")
 
print("\nSauvegarde...")
df_z1_final.to_csv('AR_df_zone1_with_soil.csv', index=False)
df_z2_final.to_csv('AR_df_zone2_with_soil.csv', index=False)

print(" AR_df_zone1_with_soil.csv sauvegardé") 
print(" AR_df_zone2_with_soil.csv sauvegardé") 