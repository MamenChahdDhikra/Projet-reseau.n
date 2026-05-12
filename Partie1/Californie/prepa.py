import os
import glob
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
 
DATA_DIR  = '.'
BANDS     = ['B2_Blue','B3_Green','B4_Red','B5_RedEdge1',
             'B6_RedEdge2','B7_RedEdge3','B8_NIR',
             'B8A_RedEdge4','B11_SWIR1','B12_SWIR2']
N_BANDS   = 10
N_TIMES   = 36
N_CLASSES = 6
LABEL_MAP = {
    'Grapes':     0,
    'Rice':       1,
    'Alfalfa':    2,
    'Almonds':    3,
    'Pistachios': 4,
    'Others':     5
} 
def load_zone(zone_name):
    print(f"\nChargement {zone_name}...")

    files = sorted(glob.glob(                       
        os.path.join(DATA_DIR, f'{zone_name}_T*.csv')
    ))

    if len(files) == 0:
        raise FileNotFoundError(f"Aucun fichier trouvé pour {zone_name}")

    print(f"  {len(files)} fichiers trouvés")

    df_ref = pd.read_csv(files[0]) 

    def parse_geo(s):
        try:
            g = json.loads(s)
            return g['coordinates'][0], g['coordinates'][1]  
        except:
            return None, None

    coords          = df_ref['.geo'].apply(parse_geo)
    df_ref['lon']   = coords.apply(lambda x: x[0])   
    df_ref['lat']   = coords.apply(lambda x: x[1])

    df_main = df_ref[['pixel_id','class_name','split',
                       'lon','lat']].copy()         

    for i, f in enumerate(files):
        t   = i + 1
        df  = pd.read_csv(f)

        for band in BANDS:
            df_main[f'T{t:02d}_{band}'] = df[band].values

        df_main[f'T{t:02d}_missing'] = df['is_missing'].values   

        if (i + 1) % 12 == 0:
            print(f"  {i+1}/36 fichiers chargés...")

    print(f"  Shape final : {df_main.shape}")
    return df_main 
 
def build_tensors(df):
    N = len(df)
    X = np.zeros((N, N_TIMES, N_BANDS), dtype=np.float32)
    for t in range(N_TIMES):
        for c, band in enumerate(BANDS):        
            col        = f'T{t+1:02d}_{band}'
            X[:, t, c] = df[col].values


    mask = np.zeros((N, N_TIMES), dtype=np.float32)
    for t in range(N_TIMES):
        col          = f'T{t+1:02d}_missing'      
        mask[:, t]   = df[col].values             

    
    Y = df['class_name'].map(LABEL_MAP).values.astype(np.int64) 

    return (
        torch.FloatTensor(X),
        torch.LongTensor(Y),
        torch.FloatTensor(mask)
    )
 
class CropDataset(Dataset):
    def __init__(self, X, Y, mask):
        self.X    = X
        self.Y    = Y
        self.mask = mask

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
         return self.X[idx], self.mask[idx], self.Y[idx]
def split_data(df, X, Y, mask):
    idx_train = []
    idx_val   = []
    idx_test  = []
                                          
    classes = df['class_name'].unique()

    for cls in classes:
        idx_cls       = df[df['class_name'] == cls].index.tolist()
        idx_cls_local = [df.index.get_loc(i) for i in idx_cls]

        np.random.seed(42)
        np.random.shuffle(idx_cls_local)

        idx_train += idx_cls_local[:240]
        idx_val   += idx_cls_local[240:300]
        idx_test  += idx_cls_local[300:]

    print(f"\nSplit train/val/test  :")
    print(f"  train : {len(idx_train):5d}  " )
    print(f"  val   : {len(idx_val):5d}  " )
    print(f"  test  : {len(idx_test):5d}  " )

    datasets = {}
    for name, idx in [('train', idx_train),
                      ('val',   idx_val),
                      ('test',  idx_test)]:
        datasets[name] = CropDataset(
            X[idx], Y[idx], mask[idx]
        )

    return datasets
 
def prepare_all_data():
    df_z1  = load_zone('CA_Zone1')
    df_z2  = load_zone('CA_Zone2')

    df_all = pd.concat([df_z1, df_z2], ignore_index=True)
    print(f"\nTotal fusionné : {len(df_all)} pixels")

    df_z1.to_csv('df_zone1.csv', index=False)
    df_z2.to_csv('df_zone2.csv', index=False)
    df_all.to_csv('df_all.csv',  index=False)
    print("df_zone1.csv, df_zone2.csv, df_all.csv sauvegardés")
 
    X, Y, mask = build_tensors(df_all)
    print(f"  X    : {X.shape}")
    print(f"  Y    : {Y.shape}")
    print(f"  mask : {mask.shape}")

    classes = list(LABEL_MAP.keys())
    for i, cls in enumerate(classes):
        n = (Y == i).sum().item()
        print(f"  {cls:<12} : {n:5d}  ({n/len(Y)*100:.1f}%)")

    datasets = split_data(df_all, X, Y, mask)

    return datasets
if __name__ == '__main__':
    datasets = prepare_all_data()
    torch.save(datasets, 'datasets.pt')
    print("\n Données prêtes!")
    print(f"  train : {len(datasets['train'])} pixels")
    print(f"  val   : {len(datasets['val'])}   pixels")
    print(f"  test  : {len(datasets['test'])}  pixels")