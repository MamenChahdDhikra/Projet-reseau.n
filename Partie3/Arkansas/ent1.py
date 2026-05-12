import os
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             f1_score, confusion_matrix,
                             precision_score, recall_score)
from torch.utils.data import Dataset, DataLoader
from model_se import MCTNetSE
from p import CropDatasetAR
# ─────────────────────────────────────────────────────────────
#  CROPDATASET
# ─────────────────────────────────────────────────────────────
class CropDataset(Dataset):
    def __init__(self, X, Y, mask):
        self.X = X; self.Y = Y; self.mask = mask
    def __len__(self): return len(self.Y)
    def __getitem__(self, i): return self.X[i], self.mask[i], self.Y[i]

# ─────────────────────────────────────────────────────────────
#  SEEDS
# ─────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
CONFIG = {
    'batch_size'   : 32,
    'epochs'       : 200,
    'lr'           : 0.001,
    'n_classes'    : 5,
    'in_channels'  : 10,
    'proj_channels': 10,
    'n_head'       : 5,
    'n_stage'      : 3,
    'se_reduction' : 4,
    'device'       : 'cuda' if torch.cuda.is_available() else 'cpu',
    'patience'     : 20,
}

CLASSES = ['Corn', 'Cotton', 'Rice', 'Soybeans', 'Others']
COLORS  = ['#2ca02c', '#d62728', '#8c564b', '#ff7f0e', '#7f7f7f']

# Résultats papier Arkansas (référence)
PAPER = {'OA': 0.968, 'Kappa': 0.951, 'F1': 0.933}

# ─────────────────────────────────────────────────────────────
#  CHARGEMENT  ← chemins locaux
# ─────────────────────────────────────────────────────────────
PT_PATH   = 'AR_datasets.pt'
SAVE_PATH = 'best_model_SE_AR.pt'

print(f"Chargement AR_datasets.pt... (device: {CONFIG['device']})")

torch.serialization.add_safe_globals([CropDatasetAR])
datasets = torch.load(PT_PATH, weights_only=False)
print("✅ Chargé !")

def seed_worker(worker_id):
    np.random.seed(SEED + worker_id); random.seed(SEED + worker_id)

g = torch.Generator(); g.manual_seed(SEED)

loaders = {
    'train': DataLoader(datasets['train'], batch_size=CONFIG['batch_size'],
                        shuffle=True, worker_init_fn=seed_worker, generator=g),
    'val'  : DataLoader(datasets['val'],   batch_size=CONFIG['batch_size'], shuffle=False),
    'test' : DataLoader(datasets['test'],  batch_size=CONFIG['batch_size'], shuffle=False),
}
print(f"Train:{len(datasets['train'])} | Val:{len(datasets['val'])} | Test:{len(datasets['test'])}")

# ─────────────────────────────────────────────────────────────
#  MODÈLE SE  ← doit être défini avant (model_se.py)
# ─────────────────────────────────────────────────────────────
model = MCTNetSE(
    in_channels   = CONFIG['in_channels'],
    n_classes     = CONFIG['n_classes'],
    n_head        = CONFIG['n_head'],
    n_stage       = CONFIG['n_stage'],
    proj_channels = CONFIG['proj_channels'],
    se_reduction  = CONFIG['se_reduction']
).to(CONFIG['device'])

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nMCTNetSE Arkansas — {n_params:,} paramètres")

# ─────────────────────────────────────────────────────────────
#  LOSS ET OPTIMISEUR
# ─────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=CONFIG['lr'], weight_decay=1e-2)

# ─────────────────────────────────────────────────────────────
#  ÉVALUATION
# ─────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            out  = model(x, mask)
            loss = criterion(out, y)
            pred = out.argmax(dim=1)
            total_loss  += loss.item()
            all_preds   += pred.cpu().tolist()
            all_labels  += y.cpu().tolist()
    loss  = total_loss / len(loader)
    oa    = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    f1    = f1_score(all_labels, all_preds, average='macro')
    return loss, oa, kappa, f1, all_labels, all_preds

# ─────────────────────────────────────────────────────────────
#  BOUCLE D'ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────
print("\n" + "="*75)
print("  ENTRAÎNEMENT MCTNetSE — Arkansas (10 bandes S2)")
print(f"  epochs={CONFIG['epochs']} | batch={CONFIG['batch_size']} | "
      f"lr={CONFIG['lr']} | patience={CONFIG['patience']}")
print(f"  Référence papier : OA={PAPER['OA']} Kappa={PAPER['Kappa']} F1={PAPER['F1']}")
print("="*75)
print(f"{'Ep':>4} | {'Tr Loss':>8} {'Tr OA':>7} {'Tr F1':>7} | "
      f"{'Va Loss':>8} {'Va OA':>7} {'Va F1':>7} {'Kappa':>7} | {'Pat':>4}")
print("-"*75)

best_val_oa, best_epoch, patience_count = 0.0, 0, 0
history = {k: [] for k in ['train_loss','val_loss','train_oa','val_oa','train_f1','val_f1']}

for epoch in range(1, CONFIG['epochs'] + 1):
    model.train()
    tr_loss, tr_preds, tr_labels = 0.0, [], []

    for x, mask, y in loaders['train']:
        x, mask, y = x.to(CONFIG['device']), mask.to(CONFIG['device']), y.to(CONFIG['device'])
        optimizer.zero_grad()
        out  = model(x, mask)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        pred       = out.argmax(dim=1)
        tr_loss   += loss.item()
        tr_preds  += pred.cpu().tolist()
        tr_labels += y.cpu().tolist()

    tr_loss /= len(loaders['train'])
    tr_oa    = accuracy_score(tr_labels, tr_preds)
    tr_f1    = f1_score(tr_labels, tr_preds, average='macro')

    va_loss, va_oa, va_kappa, va_f1, _, _ = evaluate(model, loaders['val'], CONFIG['device'])

    for k, v in zip(['train_loss','val_loss','train_oa','val_oa','train_f1','val_f1'],
                    [tr_loss, va_loss, tr_oa, va_oa, tr_f1, va_f1]):
        history[k].append(v)

    if va_oa > best_val_oa:
        best_val_oa, best_epoch, patience_count = va_oa, epoch, 0
        torch.save(model.state_dict(), SAVE_PATH)
        marker = ' ✅'
    else:
        patience_count += 1
        marker = ''

    print(f"{epoch:4d} | {tr_loss:8.4f} {tr_oa:7.4f} {tr_f1:7.4f} | "
          f"{va_loss:8.4f} {va_oa:7.4f} {va_f1:7.4f} {va_kappa:7.4f} | "
          f"{patience_count:4d}{marker}")

    if patience_count >= CONFIG['patience']:
        print(f"\n⚠️  Early Stopping époque {epoch}")
        break

print(f"\n✅ Meilleur modèle : époque {best_epoch} (Val OA={best_val_oa:.4f})")

# ─────────────────────────────────────────────────────────────
#  ÉVALUATION FINALE — TEST SET
# ─────────────────────────────────────────────────────────────
print("\n" + "="*75)
print("  ÉVALUATION FINALE — TEST SET")
print("="*75)

model.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
_, test_oa, test_kappa, test_f1, y_true, y_pred = evaluate(
    model, loaders['test'], CONFIG['device'])

y_true = np.array(y_true); y_pred = np.array(y_pred)
f1_cls   = f1_score(y_true, y_pred, average=None)
prec_cls = precision_score(y_true, y_pred, average=None, zero_division=0)
rec_cls  = recall_score(y_true, y_pred, average=None, zero_division=0)

print(f"\n  MCTNetSE — Arkansas :")
print(f"  OA    : {test_oa:.4f}   (papier: {PAPER['OA']}  | delta: {test_oa-PAPER['OA']:+.4f})")
print(f"  Kappa : {test_kappa:.4f}   (papier: {PAPER['Kappa']}  | delta: {test_kappa-PAPER['Kappa']:+.4f})")
print(f"  F1    : {test_f1:.4f}   (papier: {PAPER['F1']}  | delta: {test_f1-PAPER['F1']:+.4f})")

cm = confusion_matrix(y_true, y_pred)
print(f"\n  F1 par classe :")
for cls, f1 in zip(CLASSES, f1_cls):
    print(f"    {cls:<12} : {f1:.4f}")

# ─────────────────────────────────────────────────────────────
#  COURBES D'APPRENTISSAGE
# ─────────────────────────────────────────────────────────────
actual_epochs = len(history['train_loss'])
ep_range      = range(1, actual_epochs + 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Courbes — MCTNetSE Arkansas (+ SE Spectral Attention)',
             fontsize=13, fontweight='bold')

for ax, (tr_key, va_key, title, ref_val, ref_lbl) in zip(axes, [
    ('train_loss', 'val_loss', 'Loss',               None,          None),
    ('train_oa',   'val_oa',   'Overall Accuracy',    PAPER['OA'],  f"Papier ({PAPER['OA']})"),
    ('train_f1',   'val_f1',   'F1 Score (macro)',    PAPER['F1'],  f"Papier ({PAPER['F1']})"),
]):
    ax.plot(ep_range, history[tr_key], label='Train', color='#2196F3', linewidth=1.5)
    ax.plot(ep_range, history[va_key], label='Val',   color='#F44336', linewidth=1.5)
    ax.axvline(x=best_epoch, color='purple', linestyle='--', linewidth=1.2,
               label=f'Best (ep.{best_epoch})')
    if ref_val:
        ax.axhline(y=ref_val, color='green', linestyle='--', linewidth=1.2, label=ref_lbl)
    ax.set_title(title); ax.set_xlabel('Époque')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('learning_curves_SE_AR.png', dpi=150, bbox_inches='tight')
print("\n✅ learning_curves_SE_AR.png sauvegardé")
plt.show()

# ─────────────────────────────────────────────────────────────
#  MATRICE DE CONFUSION
# ─────────────────────────────────────────────────────────────
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
n_cls   = len(CLASSES)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'Matrice de Confusion — MCTNetSE Arkansas\n'
             f'OA={test_oa:.4f} | Kappa={test_kappa:.4f} | F1={test_f1:.4f}',
             fontsize=13, fontweight='bold')

for ax, data, title, cmap in zip(axes,
    [cm_norm, cm], ['Normalisée', 'Valeurs absolues'], ['Blues','Greens']):
    im = ax.imshow(data, cmap=cmap, vmin=0,
                   vmax=1 if title == 'Normalisée' else data.max())
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
    ax.set_xticklabels(CLASSES, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(CLASSES, fontsize=9)
    ax.set_xlabel('Prédit'); ax.set_ylabel('Réel')
    ax.set_title(title, fontweight='bold')
    thresh = data.max() / 2
    for i in range(n_cls):
        for j in range(n_cls):
            val = f'{data[i,j]:.3f}' if title == 'Normalisée' else str(data[i,j])
            ax.text(j, i, val, ha='center', va='center', fontsize=8,
                    color='white' if data[i,j] > thresh else 'black',
                    fontweight='bold' if i == j else 'normal')

plt.tight_layout()
plt.savefig('confusion_SE_AR.png', dpi=150, bbox_inches='tight')
print("✅ confusion_SE_AR.png sauvegardé")
plt.show()

# ─────────────────────────────────────────────────────────────
#  RÉSUMÉ COMPARATIF
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('MCTNetSE vs Baseline  — Arkansas',
             fontsize=13, fontweight='bold')

for (name, val, paper_val), ax in zip(
    [('OA', test_oa, PAPER['OA']),
     ('Kappa', test_kappa, PAPER['Kappa']),
     ('F1 macro', test_f1, PAPER['F1'])], axes):

    bars = ax.bar(['Baseline\n', 'MCTNetSE\n(notre modèle)'],
                  [paper_val, val],
                  color=['#90CAF9', '#1565C0'], width=0.5,
                  edgecolor='white', linewidth=2)
    for bar, v in zip(bars, [paper_val, val]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005, f'{v:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    diff  = val - paper_val
    color = '#2E7D32' if diff >= 0 else '#C62828'
    ax.set_title(f'{name}\nDelta : {diff:+.4f} ({diff/paper_val*100:.1f}%)',
                 fontsize=10, fontweight='bold', color=color)
    ax.set_ylim(0, 1.1); ax.set_ylabel('Score')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_SE_AR.png', dpi=150, bbox_inches='tight')
print("✅ comparison_SE_AR.png sauvegardé")
plt.show()
# ─────────────────────────────────────────────────────────────
#  MÉTRIQUES PAR CLASSE
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Métriques par classe — MCTNetSE Arkansas',
             fontsize=13, fontweight='bold')

x = np.arange(len(CLASSES))
w = 0.6
for ax, vals, title, subtitle in [
    (axes[0], f1_cls,   'F1 Score',  f'F1 macro = {test_f1:.4f}'),
    (axes[1], prec_cls, 'Précision', f'Précision moyenne = {prec_cls.mean():.4f}'),
    (axes[2], rec_cls,  'Rappel',    f'Rappel moyen = {rec_cls.mean():.4f}'),
]:
    bars = ax.bar(x, vals, w, color=COLORS, alpha=0.85,
                  edgecolor='white', linewidth=1.2)
    ref = PAPER['F1'] if title == 'F1 Score' else vals.mean()
    lbl = f"Papier ({PAPER['F1']})" if title == 'F1 Score' \
          else f'Moyenne ({vals.mean():.3f})'
    ax.axhline(y=ref, color='red', linestyle='--', linewidth=1.5, label=lbl)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01, f'{val:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=25, ha='right', fontsize=9)
    ax.set_ylim(0, 1.15); ax.set_ylabel(title)
    ax.set_title(f'{title}\n{subtitle}', fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('metrics_per_class_SE_AR.png', dpi=150, bbox_inches='tight')
print("✅ Sauvegardé : metrics_per_class_SE_AR.png")
plt.show()
# ─────────────────────────────────────────────────────────────
#  CARTES GÉOGRAPHIQUES
# ─────────────────────────────────────────────────────────────
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

CLASS_COLORS_MAP = {c: col for c, col in zip(CLASSES, COLORS)}
COLOR_LIST       = [CLASS_COLORS_MAP[c] for c in CLASSES]

def plot_geo_maps_AR(y_true, y_pred):
    csv_files = {
        'Zone 1': 'AR_df_zone1.csv',
        'Zone 2': 'AR_df_zone2.csv',
    }
    missing = [f for f in csv_files.values() if not os.path.exists(f)]
    if missing:
        print(f"  Fichiers manquants : {missing}")
        return

    df_z1 = pd.read_csv(csv_files['Zone 1'])[['pixel_id','class_name','lon','lat']]
    df_z2 = pd.read_csv(csv_files['Zone 2'])[['pixel_id','class_name','lon','lat']]
    df_z1['zone'] = 'Zone 1'
    df_z2['zone'] = 'Zone 2'
    df_all = pd.concat([df_z1, df_z2], ignore_index=True)

    np.random.seed(SEED)
    idx_test = []
    for cls in df_all['class_name'].unique():
        idx_cls = df_all[df_all['class_name'] == cls].index.tolist()
        np.random.shuffle(idx_cls)
        idx_test += idx_cls[300:]

    df_test = df_all.iloc[idx_test].reset_index(drop=True)
    df_test['y_true']  = y_true
    df_test['y_pred']  = y_pred
    df_test['correct'] = (df_test['y_true'] == df_test['y_pred'])

    cmap_classes   = ListedColormap(COLOR_LIST)
    legend_patches = [mpatches.Patch(color=COLOR_LIST[i], label=CLASSES[i])
                      for i in range(len(CLASSES))]
    error_patches  = [
        mpatches.Patch(color='#4CAF50', label='Correct ✅'),
        mpatches.Patch(color='#F44336', label='Erreur ❌'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(26, 18))
    fig.suptitle(
        'Arkansas — Vérité terrain / Prédiction / Erreurs MCTNetSE',
        fontsize=14, fontweight='bold', y=0.98
    )

    for row, zone in enumerate(['Zone 1', 'Zone 2']):
        df_z = df_test[df_test['zone'] == zone]
        if df_z.empty:
            for col in range(3):
                axes[row, col].set_visible(False)
            continue

        lon      = df_z['lon'].values
        lat      = df_z['lat'].values
        true_lbl = df_z['y_true'].values
        pred_lbl = df_z['y_pred'].values
        correct  = df_z['correct'].values
        s        = max(2, min(10, 4000 // len(df_z)))

        # Vérité terrain
        ax = axes[row, 0]
        ax.scatter(lon, lat, c=true_lbl, cmap=cmap_classes,
                   vmin=0, vmax=len(CLASSES)-1,
                   s=s, alpha=0.85, linewidths=0)
        ax.set_title(f'{zone} — Vérité terrain',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.grid(alpha=0.2); ax.set_facecolor('white')

        # Prédiction
        ax = axes[row, 1]
        ax.scatter(lon, lat, c=pred_lbl, cmap=cmap_classes,
                   vmin=0, vmax=len(CLASSES)-1,
                   s=s, alpha=0.85, linewidths=0)
        ax.set_title(f'{zone} — Prédiction',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.grid(alpha=0.2); ax.set_facecolor('white')

        # Erreurs
        ax = axes[row, 2]
        colors_err = np.where(correct, '#4CAF50', '#F44336')
        ax.scatter(lon, lat, c=colors_err,
                   s=s, alpha=0.85, linewidths=0)
        n_ok  = correct.sum()
        n_tot = len(correct)
        ax.set_title(
            f'{zone} — Erreurs\n'
            f'Correct : {n_ok}/{n_tot}  ({n_ok/n_tot:.1%})',
            fontsize=11, fontweight='bold'
        )
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.grid(alpha=0.2); ax.set_facecolor('white')

    all_patches = legend_patches + error_patches
    fig.legend(handles=all_patches,
               loc='lower center',
               ncol=len(CLASSES) + 2,
               fontsize=10,
               markerscale=1.5,
               framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    plt.subplots_adjust(left=0.05, right=0.98,
                        top=0.91,  bottom=0.08,
                        hspace=0.45, wspace=0.25)
    plt.savefig('geo_maps_SE_AR.png', dpi=150, bbox_inches='tight')
    print("✅ Sauvegardé : geo_maps_SE_AR.png")
    plt.show()

plot_geo_maps_AR(y_true, y_pred)