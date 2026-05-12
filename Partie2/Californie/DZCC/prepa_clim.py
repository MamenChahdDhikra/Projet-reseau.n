 

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
 
BANDS = ['B2_Blue','B3_Green','B4_Red','B5_RedEdge1',
         'B6_RedEdge2','B7_RedEdge3','B8_NIR',
         'B8A_RedEdge4','B11_SWIR1','B12_SWIR2']
N_BANDS   = 10
N_TIMES   = 36
N_CLASSES = 6
LABEL_MAP = {
    'Grapes':0, 'Rice':1, 'Alfalfa':2,
    'Almonds':3, 'Pistachios':4, 'Others':5
}
 
def load_data_with_clim():
    print("\nChargement des données avec covariables climat...")

    df_z1 = pd.read_csv('df_zone1_with_clim.csv')
    df_z2 = pd.read_csv('df_zone2_with_clim.csv')

    print(f"Zone1 : {df_z1.shape}")
    print(f"Zone2 : {df_z2.shape}")

    df_all = pd.concat([df_z1, df_z2], ignore_index=True)
    print(f"Total : {df_all.shape}")

    return df_all
 
def build_tensors_with_clim(df):
    N = len(df)
    print(f"\nConstruction tenseurs (N={N}, T=36, C=13)...")
 
    X = np.zeros((N, N_TIMES, N_BANDS + 3), dtype=np.float32)
 
    for t in range(N_TIMES):
        for c, band in enumerate(BANDS):
            col        = f'T{t+1:02d}_{band}'
            X[:, t, c] = df[col].values
 
    for t in range(N_TIMES):
        col         = f'T{t+1:02d}_temp_norm'   
        X[:, t, 10] = df[col].values  
 
    for t in range(N_TIMES):
        col         = f'T{t+1:02d}_prec_norm'  
        X[:, t, 11] = df[col].values   
 
    for t in range(N_TIMES):
        col         = f'T{t+1:02d}_dew_norm'
        X[:, t, 12] = df[col].values
 
    mask = np.zeros((N, N_TIMES), dtype=np.float32)
    for t in range(N_TIMES):
        col        = f'T{t+1:02d}_missing'
        mask[:, t] = df[col].values
 
    Y = df['class_name'].map(LABEL_MAP).values.astype(np.int64)

    print(f"  X    : {X.shape}")    
    print(f"  Y    : {Y.shape}")    
    print(f"  mask : {mask.shape}") 

    return (
        torch.FloatTensor(X),
        torch.LongTensor(Y),
        torch.FloatTensor(mask)
    )
 
class CropDatasetClim(Dataset):
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

    print(f"\nSplit train/val/test :")
    print(f"  train : {len(idx_train):5d}")
    print(f"  val   : {len(idx_val):5d}")
    print(f"  test  : {len(idx_test):5d}")

    datasets = {}
    for name, idx in [('train', idx_train),
                      ('val',   idx_val),
                      ('test',  idx_test)]:
        datasets[name] = CropDatasetClim(
            X[idx], Y[idx], mask[idx]
        )

    return datasets
 
def prepare_all_data_with_clim():

    df_all     = load_data_with_clim()
    X, Y, mask = build_tensors_with_clim(df_all)

    print("\nDistribution classes :")
    for i, cls in enumerate(LABEL_MAP.keys()):
        n = (Y == i).sum().item()
        print(f"  {cls:<12} : {n}")

    datasets = split_data(df_all, X, Y, mask)

    torch.save(datasets, 'datasets_with_clim.pt')
    print("\n✅ datasets_with_clim.pt sauvegardé !")

    return datasets
 
if __name__ == '__main__':
    print("="*55)
    print("  PRÉPARATION DONNÉES AVEC 3 COVARIABLES CLIMAT")
    print("="*55)

    datasets = prepare_all_data_with_clim()

    x, mask, y = datasets['train'][0]
    print(f"\nTest pixel 0 :")
    print(f"  x    : {x.shape}")    
    print(f"  mask : {mask.shape}") 
    print(f"  y    : {y.item()}")

    print(f"\nCovariables climat pixel 0 :")
    print(f"  temp_norm (T01) : {x[0][10]:.4f}")
    print(f"  prec_norm (T01) : {x[0][11]:.4f}")
    print(f"  dew_norm  (T01) : {x[0][12]:.4f}")
    print(f"  temp_norm (T18) : {x[17][10]:.4f}")
    print(f"  prec_norm (T18) : {x[17][11]:.4f}")
    print(f"  dew_norm  (T18) : {x[17][12]:.4f}")
    print(f"  → valeurs différentes ")
 
