
import os
import glob
import pandas as pd
 
CLIM_DIR   = '.'
CLIM_COLS  = ['temp_norm', 'prec_norm', 'dew_norm']
def load_climat(zone_name):
    print(f"\nChargement climat {zone_name}...")

    files = sorted(glob.glob(
        os.path.join(CLIM_DIR, f'{zone_name}_CLIM_T*.csv')
    ))

    if len(files) == 0:
        raise FileNotFoundError(
            f"Aucun fichier climat trouvé pour {zone_name}\n"
            f"Attendu : {zone_name}_CLIM_T01_DOY010.csv ... "
            f"{zone_name}_CLIM_T36_DOY360.csv"
        )

    print(f"  {len(files)} fichiers trouvés")

    df_ref = pd.read_csv(files[0])[['pixel_id']]

    for i, f in enumerate(files):
        t  = i + 1
        df = pd.read_csv(f)[['pixel_id'] + CLIM_COLS]

        df = df.rename(columns={
            'temp_norm' : f'T{t:02d}_temp_norm',
            'prec_norm' : f'T{t:02d}_prec_norm',
            'dew_norm'  : f'T{t:02d}_dew_norm'
        })

        df_ref = df_ref.merge(df, on='pixel_id', how='left')

    print(f"  Shape climat : {df_ref.shape}")
    return df_ref
 
def merge_zone_clim(zone_csv, zone_name, out_path):
    print(f"\n{'='*50}")
    print(f"Fusion {zone_name}")

    df_s2 = pd.read_csv(zone_csv)
    print(f"  S2 shape     : {df_s2.shape}")

    df_clim = load_climat(zone_name)

    df_merged = df_s2.merge(df_clim, on='pixel_id', how='left')

    # Vérification NaN
    clim_cols_all = [c for c in df_merged.columns
                     if '_norm' in c]
    n_missing = df_merged[clim_cols_all].isnull().sum().sum()

    if n_missing > 0:
        print(f"    {n_missing} valeurs manquantes → remplissage par moyenne")
        for col in clim_cols_all:
            df_merged[col] = df_merged[col].fillna(
                df_merged[col].mean()
            )
    else:
        print(f"   Aucune valeur manquante")

    df_merged.to_csv(out_path, index=False)

    print(f"  Shape final  : {df_merged.shape}")
    print(f"  Colonnes ajoutées : {len(clim_cols_all)} "
          f"(3 variables × 36 périodes)")
    print(f"  Sauvegardé : {out_path}")
    return df_merged


if __name__ == '__main__':

    df_z1 = merge_zone_clim(
        zone_csv  = 'AR_df_zone1.csv',
        zone_name = 'AR_Zone1',
        out_path  = 'AR_df_zone1_with_clim.csv'
    )

    df_z2 = merge_zone_clim(
        zone_csv  = 'AR_df_zone2.csv',
        zone_name = 'AR_Zone2',
        out_path  = 'AR_df_zone2_with_clim.csv'
    )

   