 

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

 
BANDS = ['B2','B3','B4','B5',
         'B6','B7','B8',
         'B8A','B11','B12']
N_BANDS   = 10
N_TIMES   = 36
N_CLASSES = 5

 
LABEL_MAP = {
    'Soybeans':0, 'Rice':1, 'Corn':2,
    'Cotton':3,   'Others':4
}

def load_data():
    df_z1 = pd.read_csv('AR_df_zone1.csv')
    df_z2 = pd.read_csv('AR_df_zone2.csv')
    print(f"Zone1 : {df_z1.shape}")
    print(f"Zone2 : {df_z2.shape}")
    df_all = pd.concat([df_z1, df_z2], ignore_index=True)
    print(f"Total : {df_all.shape}")
    return df_all

def build_tensors(df):
    N = len(df)
    print(f"\nConstruction tenseurs (N={N}, T=36, C=10)...")
    X = np.zeros((N, N_TIMES, N_BANDS ), dtype=np.float32)
 
    for t in range(N_TIMES):
        for c, band in enumerate(BANDS):
            col        = f'T{t+1:02d}_{band}'
            X[:, t, c] = df[col].values/10000
    mask = np.zeros((N, N_TIMES), dtype=np.float32)
    for t in range(N_TIMES):
        col        = f'T{t+1:02d}_missing'
        mask[:, t] = df[col].values
 
    Y = df['class_name'].map(LABEL_MAP).values.astype(np.int64)
    nan_mask = np.isnan(Y.astype(float))
    if nan_mask.sum() > 0:
        print(f"  ⚠️ {nan_mask.sum()} classes inconnues !")
        print(f"  Classes trouvées : {df['class_name'].unique()}")

    print(f"  X    : {X.shape}")
    print(f"  Y    : {Y.shape}")
    print(f"  mask : {mask.shape}")

    return (
        torch.FloatTensor(X),
        torch.LongTensor(Y),
        torch.FloatTensor(mask)
    )
class CropDatasetAR(Dataset):
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

    np.random.seed(42)

    for cls in classes:
        idx_cls       = df[df['class_name'] == cls].index.tolist()
        idx_cls_local = [df.index.get_loc(i) for i in idx_cls]

        np.random.shuffle(idx_cls_local)

        idx_train += idx_cls_local[:240]
        idx_val   += idx_cls_local[240:300]
        idx_test  += idx_cls_local[300:]

    print(f"\nSplit :")
    print(f"  train : {len(idx_train)}")
    print(f"  val   : {len(idx_val)}")
    print(f"  test  : {len(idx_test)}")

    datasets = {}
    for name, idx in [('train', idx_train),
                      ('val',   idx_val),
                      ('test',  idx_test)]:
        datasets[name] = CropDatasetAR(
            X[idx], Y[idx], mask[idx]
        )

    return datasets
def prepare_all_data():

    df_all     = load_data()
    X, Y, mask = build_tensors(df_all)

    print("\nDistribution classes :")
    for i, cls in enumerate(LABEL_MAP.keys()):
        n = (Y == i).sum().item()
        print(f"  {cls:<12} : {n}")

    datasets = split_data(df_all, X, Y, mask)

    torch.save(datasets, 'AR_datasets.pt')
    print("\n✅ AR_datasets.pt sauvegardé !")

    return datasets
 
if __name__ == '__main__':
    print("="*55)
    print("  PRÉPARATION DONNÉES ARKANSAS ")
    print("="*55)

    datasets = prepare_all_data()
    print(f"\nDistribution Y train :")
    for i, cls in enumerate(LABEL_MAP.keys()):
        n = (datasets['train'].Y == i).sum().item()
        print(f"  {cls:<12} : {n}")

    print("\n Données Arkansas avec 3 variables climat prêtes !")