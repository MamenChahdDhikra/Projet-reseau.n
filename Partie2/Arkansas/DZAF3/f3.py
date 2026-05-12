
import pandas as pd
import numpy as np
 
SOIL_COLS = ['soil_ph_norm', 'soil_clay_norm', 'soil_org_norm']
TOPO_COLS = ['elevation_norm', 'landforms_norm']   
def merge_zone(zone_name):
    print(f"\n{'='*50}")
    print(f"Fusion {zone_name}...")
 
    df_clim = pd.read_csv(f'AR_df_{zone_name}_with_clim.csv')
    df_soil = pd.read_csv(f'AR_df_{zone_name}_with_soil.csv')
    df_topo = pd.read_csv(f'AR_df_{zone_name}_with_topo.csv')

    print(f"  clim  : {df_clim.shape}")
    print(f"  sol   : {df_soil.shape}")
    print(f"  topo  : {df_topo.shape}")

    df = df_clim.copy()

    df = df.merge(
        df_soil[['pixel_id'] + SOIL_COLS],
        on='pixel_id', how='left'
    )
 
    df = df.merge(
        df_topo[['pixel_id'] + TOPO_COLS],
        on='pixel_id', how='left'
    )
 
    cov_cols = SOIL_COLS + ['elevation_norm', 'landforms_norm']
    missing  = df[cov_cols].isnull().sum()
    if missing.any():
        print(f"  ⚠️  Valeurs manquantes :")
        print(missing[missing > 0])
        for col in cov_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
                print(f"     {col} → rempli par moyenne")
    else:
        print(f"  ✅ Aucune valeur manquante")
 
    out = f'df_{zone_name}_full.csv'
    df.to_csv(out, index=False)

    print(f"\n  Shape final : {df.shape}")
    print(f"  ✅ Sauvegardé : {out}")

    return df
 
if __name__ == '__main__':
    print("="*50)
    print("  FUSION SOL + TOPO + CLIMAT")
    print("="*50)

    df_z1 = merge_zone('zone1')
    df_z2 = merge_zone('zone2')

    print(f"\n{'='*50}")
    print(f"✅ FUSION TERMINÉE !")
    print(f"  df_zone1_full.csv : {df_z1.shape}")
    print(f"  df_zone2_full.csv : {df_z2.shape}")
  