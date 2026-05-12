import pandas as pd
import numpy as np
 
print("Chargement des données...")
df_z1  = pd.read_csv('df_zone1.csv')
df_z2  = pd.read_csv('df_zone2.csv')
sol_z1 = pd.read_csv('CA_Zone1_SOL3_V.csv')
sol_z2 = pd.read_csv('CA_Zone2_SOL3_V.csv')

print(f"Zone1 : {df_z1.shape}")   
print(f"Zone2 : {df_z2.shape}")   
  
all_sol = pd.concat([sol_z1, sol_z2])

ph_min      = all_sol['soil_ph'].min()
ph_max      = all_sol['soil_ph'].max()
clay_min    = all_sol['soil_clay'].min()
clay_max    = all_sol['soil_clay'].max()
org_min     = all_sol['soil_organic'].min()
org_max     = all_sol['soil_organic'].max()

print(f"  soil_ph      : [{ph_min:.2f},  {ph_max:.2f}]")
print(f"  soil_clay    : [{clay_min:.2f}, {clay_max:.2f}]")
print(f"  soil_organic : [{org_min:.2f},  {org_max:.2f}]")
 
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
 
print("\nFusion...")

cols_sol = ['pixel_id', 'soil_ph_norm',
            'soil_clay_norm', 'soil_org_norm']

df_z1_final = df_z1.merge(
    sol_z1[cols_sol],
    on='pixel_id',
    how='left'
)

df_z2_final = df_z2.merge(
    sol_z2[cols_sol],
    on='pixel_id',
    how='left'
)
 
print("\nNaN avant remplissage :")
for col in ['soil_ph_norm', 'soil_clay_norm', 'soil_org_norm']:
    print(f"  Zone1 {col} NaN : "
          f"{df_z1_final[col].isna().sum()}")
    print(f"  Zone2 {col} NaN : "
          f"{df_z2_final[col].isna().sum()}")
 
for col in ['soil_ph_norm', 'soil_clay_norm', 'soil_org_norm']:
    df_z1_final[col].fillna(
        df_z1_final[col].mean(), inplace=True)
    df_z2_final[col].fillna(
        df_z2_final[col].mean(), inplace=True)
 
print(f"\nZone1 final : {df_z1_final.shape}")  
print(f"Zone2 final : {df_z2_final.shape}")  

print(f"\nValeurs normalisées Zone1 :")
for col in ['soil_ph_norm', 'soil_clay_norm', 'soil_org_norm']:
    print(f"  {col} : "
          f"{df_z1_final[col].min():.3f} → "
          f"{df_z1_final[col].max():.3f} ✅")
 
print("\nSauvegarde...")
df_z1_final.to_csv('df_zone1_with_soil.csv', index=False)
df_z2_final.to_csv('df_zone2_with_soil.csv', index=False)

print("✅ df_zone1_with_soil.csv sauvegardé")  
print("✅ df_zone2_with_soil.csv sauvegardé")  