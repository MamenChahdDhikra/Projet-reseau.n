import os
import glob
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
DATA_DIR     = '.'
BANDS_PREFIX = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']
N_BANDS      = 10
N_TIMES      = 36

N_CLASSES = 5
LABEL_MAP = {
    'Corn'    : 0,
    'Cotton'  : 1,
    'Rice'    : 2,
    'Soybeans': 3,
    'Others'  : 4,
}
 

def parse_geo(s):
    try:
        g = json.loads(str(s))
        return g['coordinates'][0], g['coordinates'][1]
    except:
        return None, None


def get_class_name(code):
    m = {1:'Corn', 2:'Cotton', 3:'Rice', 5:'Soybeans', 99:'Others'}
    try:
        return m.get(int(code), 'Others')
    except:
        return 'Others'

 

def load_zone(zone_name):
    print(f"\nChargement {zone_name}...")

    pattern = os.path.join(DATA_DIR, f'Arkansas_{zone_name}_t*.csv')
    files   = sorted(glob.glob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(f"Aucun fichier trouvé : {pattern}")
    print(f"  {len(files)} fichiers trouvés")
 
    df_ref = None
    for f in files:
        try:
            tmp = pd.read_csv(f)
            if tmp.shape[0] > 0:
                df_ref = tmp
                break
        except pd.errors.EmptyDataError:
            continue

    if df_ref is None:
        raise ValueError(f"Tous les fichiers de {zone_name} sont vides.")

    n_points = len(df_ref)

    coords        = df_ref['.geo'].apply(parse_geo)
    df_ref['lon'] = coords.apply(lambda x: x[0])
    df_ref['lat'] = coords.apply(lambda x: x[1])

    zone_id = zone_name.replace('Zone', '')
    df_ref['pixel_id'] = [f'AR_Z{zone_id}_{i:05d}' for i in range(n_points)]

    if 'class_name' not in df_ref.columns:
        lc = 'class' if 'class' in df_ref.columns else 'cropland'
        df_ref['class_name'] = df_ref[lc].apply(get_class_name)

    
    if 'zone' not in df_ref.columns:
        df_ref['zone'] = int(zone_id)

    df_main = df_ref[['pixel_id','class_name','zone','lon','lat']].copy()
 
    for i, f in enumerate(files):
        t       = i + 1
        idx_str = f'{i:02d}'

        try:
            df_t     = pd.read_csv(f)
            is_empty = df_t.shape[0] == 0
        except pd.errors.EmptyDataError:
            is_empty = True

        if is_empty:
            print(f"  ⚠️  t{idx_str} VIDE → zéros + missing=1")
            for bp in BANDS_PREFIX:
                df_main[f'T{t:02d}_{bp}'] = 0.0
            df_main[f'T{t:02d}_missing'] = 1.0
            continue

        df_t     = df_t.reset_index(drop=True)
        n_actual = len(df_t)

        if n_actual != n_points:
            print(f"  ⚠️  t{idx_str} : {n_actual} lignes → padding")

        def get_vals(col_name):
            if col_name in df_t.columns:
                raw = df_t[col_name].fillna(0).values.astype(np.float32)
            else:
                raw = np.zeros(n_actual, dtype=np.float32)
            if len(raw) < n_points:
                padded         = np.zeros(n_points, dtype=np.float32)
                padded[:len(raw)] = raw
                return padded
            return raw[:n_points]

        is_missing_arr = np.zeros(n_points, dtype=np.float32)
        for bp in BANDS_PREFIX:
            vals = get_vals(f'{bp}_t{idx_str}')
            df_main[f'T{t:02d}_{bp}'] = vals
            is_missing_arr = np.where(vals == 0,
                                      is_missing_arr + 1,
                                      is_missing_arr)

        df_main[f'T{t:02d}_missing'] = (
            is_missing_arr == N_BANDS).astype(np.float32)

        if (i + 1) % 12 == 0:
            print(f"  {i+1}/36 fichiers chargés...")

    print(f"  Shape final : {df_main.shape}")
    dist = df_main['class_name'].value_counts()
    print(f"  Distribution : " +
          ' | '.join([f'{c}={dist.get(c,0)}' for c in LABEL_MAP]))
    return df_main

def prepare_all_data():
    df_z1 = load_zone('Zone1')
    df_z2 = load_zone('Zone2')

    df_all = pd.concat([df_z1, df_z2], ignore_index=True)
    print(f"\nTotal fusionné : {len(df_all)} pixels")

  
    for name, df in [('AR_df_zone1.csv', df_z1),
                     ('AR_df_zone2.csv', df_z2),
                     ('AR_df_all.csv',   df_all)]:
        df.to_csv(os.path.join(DATA_DIR, name), index=False)
    print(" CSV intermédiaires sauvegardés")

if __name__ == '__main__':
    datasets = prepare_all_data()