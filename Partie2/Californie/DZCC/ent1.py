 
import os
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             f1_score, confusion_matrix,
                             precision_score, recall_score)
from torch.utils.data import DataLoader
from prepa_clim import prepare_all_data_with_clim, CropDatasetClim   
from model_ca_bands import MCTNet
 
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
 
CONFIG = {
    'batch_size'   : 32,
    'epochs'       : 200,
    'lr'           : 0.001,
    'n_classes'    : 6,
    'in_channels'  : 13,    
    'proj_channels': 10,
    'n_head'       : 5,
    'n_stage'      : 3,
    'device'       : 'cpu',
    'patience'     : 20,
}

CLASSES = ['Grapes', 'Rice', 'Alfalfa',
           'Almonds', 'Pistachios', 'Others']

CLASS_COLORS_MAP = {
    'Grapes'    : '#8B0000',
    'Rice'      : '#FFD700',
    'Alfalfa'   : '#228B22',
    'Almonds'   : '#DEB887',
    'Pistachios': '#6B8E23',
    'Others'    : '#A9A9A9',
}
COLOR_LIST = [CLASS_COLORS_MAP[c] for c in CLASSES]
COLORS     = ['#7B1FA2', '#0288D1', '#388E3C',
              '#F57C00', '#FBC02D', '#9E9E9E']
 
print("Chargement des données Californie avec climat...")
torch.serialization.add_safe_globals([CropDatasetClim])

if os.path.exists('datasets_with_clim.pt'):
    datasets = torch.load('datasets_with_clim.pt', weights_only=False)
    print("✅ datasets_with_clim.pt chargé !")
else:
    datasets = prepare_all_data_with_clim()

def seed_worker(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)

g = torch.Generator()
g.manual_seed(SEED)

loaders = {
    'train': DataLoader(datasets['train'],
                        batch_size=CONFIG['batch_size'],
                        shuffle=True,
                        worker_init_fn=seed_worker,
                        generator=g),
    'val'  : DataLoader(datasets['val'],
                        batch_size=CONFIG['batch_size'],
                        shuffle=False),
    'test' : DataLoader(datasets['test'],
                        batch_size=CONFIG['batch_size'],
                        shuffle=False),
}

print(f"Train : {len(datasets['train'])} | "
      f"Val : {len(datasets['val'])} | "
      f"Test : {len(datasets['test'])}")
 
model = MCTNet(
    in_channels  =CONFIG['in_channels'],
    n_classes    =CONFIG['n_classes'],
    n_head       =CONFIG['n_head'],
    n_stage      =CONFIG['n_stage'],
    proj_channels=CONFIG['proj_channels']
).to(CONFIG['device'])

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nMCTNet avec climat — {n_params:,} paramètres")
 
 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=CONFIG['lr'], weight_decay=1e-2)

 
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for x_batch, mask_batch, y_batch in loader:
            x_batch    = x_batch.to(device)
            mask_batch = mask_batch.to(device)
            y_batch    = y_batch.to(device)
            outputs    = model(x_batch, mask_batch)
            loss       = criterion(outputs, y_batch)
            preds      = outputs.argmax(dim=1)
            total_loss  += loss.item()
            all_preds   += preds.cpu().tolist()
            all_labels  += y_batch.cpu().tolist()
    loss  = total_loss / len(loader)
    oa    = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    f1    = f1_score(all_labels, all_preds, average='macro')
    return loss, oa, kappa, f1, all_labels, all_preds
 
print("\n" + "="*75)
print("  ENTRAÎNEMENT MCTNet + Climat — Californie")
print(f"  epochs={CONFIG['epochs']} | batch={CONFIG['batch_size']} | "
      f"lr={CONFIG['lr']} | patience={CONFIG['patience']}")
print(f"  in_channels=13 (10 bandes + 3 climat)")
print(f"  SEED={SEED} — résultats reproductibles ✅")
print("="*75)
print(f"{'Ep':>4} | {'Tr Loss':>8} {'Tr OA':>7} {'Tr F1':>7} | "
      f"{'Va Loss':>8} {'Va OA':>7} {'Va F1':>7} {'Kappa':>7} | {'Pat':>4}")
print("-"*75)

best_val_oa    = 0.0
best_epoch     = 0
patience_count = 0
history = {'train_loss': [], 'val_loss': [],
           'train_oa':   [], 'val_oa':   [],
           'train_f1':   [], 'val_f1':   []}

for epoch in range(1, CONFIG['epochs'] + 1):

    model.train()
    train_loss, train_preds, train_labels = 0.0, [], []
    for x_batch, mask_batch, y_batch in loaders['train']:
        x_batch    = x_batch.to(CONFIG['device'])
        mask_batch = mask_batch.to(CONFIG['device'])
        y_batch    = y_batch.to(CONFIG['device'])
        optimizer.zero_grad()
        outputs = model(x_batch, mask_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        preds         = outputs.argmax(dim=1)
        train_loss   += loss.item()
        train_preds  += preds.cpu().tolist()
        train_labels += y_batch.cpu().tolist()

    train_loss /= len(loaders['train'])
    train_oa    = accuracy_score(train_labels, train_preds)
    train_f1    = f1_score(train_labels, train_preds, average='macro')

    val_loss, val_oa, val_kappa, val_f1, _, _ = evaluate(
        model, loaders['val'], CONFIG['device'])

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_oa'].append(train_oa)
    history['val_oa'].append(val_oa)
    history['train_f1'].append(train_f1)
    history['val_f1'].append(val_f1)

    if val_oa > best_val_oa:
        best_val_oa    = val_oa
        best_epoch     = epoch
        patience_count = 0
        torch.save(model.state_dict(), 'best_model_clim.pt')
        marker = ' ✅'
    else:
        patience_count += 1
        marker = ''

    print(f"{epoch:4d} | {train_loss:8.4f} {train_oa:7.4f} {train_f1:7.4f} | "
          f"{val_loss:8.4f} {val_oa:7.4f} {val_f1:7.4f} {val_kappa:7.4f} | "
          f"{patience_count:4d}{marker}")

    if patience_count >= CONFIG['patience']:
        print(f"\n⚠️  Early Stopping époque {epoch} — "
              f"meilleur Val OA={best_val_oa:.4f} (époque {best_epoch})")
        break

print(f"\n✅ Meilleur modèle : époque {best_epoch} "
      f"(Val OA = {best_val_oa:.4f})")

 
print("\n" + "="*75)
print("  ÉVALUATION FINALE — TEST SET")
print("="*75)

model.load_state_dict(torch.load('best_model_clim.pt', weights_only=True))

test_loss, test_oa, test_kappa, test_f1, \
    y_true, y_pred = evaluate(model, loaders['test'], CONFIG['device'])

y_true = np.array(y_true)
y_pred = np.array(y_pred)

f1_cls   = f1_score(y_true, y_pred, average=None)
prec_cls = precision_score(y_true, y_pred, average=None, zero_division=0)
rec_cls  = recall_score(y_true, y_pred, average=None, zero_division=0)

print(f"\nRésultats avec covariables climat :")
print(f"  OA    : {test_oa:.4f}   (papier: 0.8524)")
print(f"  Kappa : {test_kappa:.4f}   (papier: 0.8194)")
print(f"  F1    : {test_f1:.4f}   (papier: 0.8301)")

cm = confusion_matrix(y_true, y_pred)
print(f"\nMatrice de confusion :")
print(f"{'':12}", end='')
for cls in CLASSES:
    print(f"{cls:14}", end='')
print()
for i, cls in enumerate(CLASSES):
    print(f"{cls:12}", end='')
    for j in range(len(CLASSES)):
        print(f"{cm[i,j]:14}", end='')
    print()

print(f"\nF1 par classe :")
for cls, f1 in zip(CLASSES, f1_cls):
    print(f"  {cls:<12} : {f1:.4f}")
 
actual_epochs = len(history['train_loss'])
epochs_range  = range(1, actual_epochs + 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Courbes — MCTNet + Climat — Californie',
             fontsize=13, fontweight='bold')

axes[0].plot(epochs_range, history['train_loss'],
             label='Train', color='#2196F3', linewidth=1.5)
axes[0].plot(epochs_range, history['val_loss'],
             label='Val',   color='#F44336', linewidth=1.5)
axes[0].axvline(x=best_epoch, color='green', linestyle='--',
                linewidth=1.5, label=f'Best ({best_epoch})')
axes[0].set_title('Loss'); axes[0].set_xlabel('Époque')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(epochs_range, history['train_oa'],
             label='Train', color='#2196F3', linewidth=1.5)
axes[1].plot(epochs_range, history['val_oa'],
             label='Val',   color='#F44336', linewidth=1.5)
axes[1].axvline(x=best_epoch, color='purple', linestyle='--',
                linewidth=1.5, label=f'Best ({best_epoch})')
axes[1].set_title('Overall Accuracy (OA)'); axes[1].set_xlabel('Époque')
axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(epochs_range, history['train_f1'],
             label='Train', color='#2196F3', linewidth=1.5)
axes[2].plot(epochs_range, history['val_f1'],
             label='Val',   color='#F44336', linewidth=1.5)
axes[2].set_title('F1 Score (macro)'); axes[2].set_xlabel('Époque')
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('learning_curves_clim.png', dpi=150, bbox_inches='tight')
print("\n✅ Sauvegardé : learning_curves_clim.png")
plt.show()
 
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Matrice de Confusion — MCTNet + Climat\n'
             f'OA={test_oa:.4f} | Kappa={test_kappa:.4f} | F1={test_f1:.4f}',
             fontsize=13, fontweight='bold')

ax = axes[0]
im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(CLASSES, rotation=30, ha='right', fontsize=9)
ax.set_yticklabels(CLASSES, fontsize=9)
ax.set_xlabel('Classe prédite', fontsize=11)
ax.set_ylabel('Classe réelle', fontsize=11)
ax.set_title('Matrice normalisée', fontweight='bold')
for i in range(6):
    for j in range(6):
        val   = cm_norm[i, j]
        color = 'white' if val > 0.5 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                color=color, fontsize=8,
                fontweight='bold' if i == j else 'normal')

ax2 = axes[1]
im2 = ax2.imshow(cm, interpolation='nearest', cmap='Greens')
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
ax2.set_xticks(range(6)); ax2.set_yticks(range(6))
ax2.set_xticklabels(CLASSES, rotation=30, ha='right', fontsize=9)
ax2.set_yticklabels(CLASSES, fontsize=9)
ax2.set_xlabel('Classe prédite', fontsize=11)
ax2.set_ylabel('Classe réelle', fontsize=11)
ax2.set_title('Matrice (valeurs absolues)', fontweight='bold')
for i in range(6):
    for j in range(6):
        val   = cm[i, j]
        total = cm[i, :].sum()
        color = 'white' if val > total * 0.5 else 'black'
        ax2.text(j, i, str(val), ha='center', va='center',
                 color=color, fontsize=8,
                 fontweight='bold' if i == j else 'normal')

plt.tight_layout()
plt.savefig('6_confusion_matrix_clim.png', dpi=150, bbox_inches='tight')
print("✅ Sauvegardé : 6_confusion_matrix_clim.png")
plt.show()
 
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Métriques par classe — MCTNet + Climat',
             fontsize=13, fontweight='bold')

x = np.arange(6)
w = 0.6
for ax, vals, title, subtitle in [
    (axes[0], f1_cls,   'F1 Score',  f'F1 macro = {test_f1:.4f}'),
    (axes[1], prec_cls, 'Précision', f'Précision moyenne = {prec_cls.mean():.4f}'),
    (axes[2], rec_cls,  'Rappel',    f'Rappel moyen = {rec_cls.mean():.4f}'),
]:
    bars = ax.bar(x, vals, w, color=COLORS, alpha=0.85,
                  edgecolor='white', linewidth=1.2)
    ref = 0.8301 if title == 'F1 Score' else vals.mean()
    lbl = 'Papier (0.8301)' if title == 'F1 Score' \
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
plt.savefig('7_metrics_per_class_clim.png', dpi=150, bbox_inches='tight')
print("Sauvegardé : 7_metrics_per_class_clim.png")
plt.show()
 
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Résumé Global — MCTNet climat vs avec bandes',
             fontsize=13, fontweight='bold')

for (name, val, paper_val), ax in zip(
    [('OA', test_oa, 0.9260),
     ('Kappa', test_kappa, 0.9018),
     ('F1 macro', test_f1, 0.9025)], axes):
    bars = ax.bar(['bandes', 'Obtenu'], [paper_val, val],
                  color=['#90CAF9', '#1565C0'], width=0.5,
                  edgecolor='white', linewidth=2)
    for bar, v in zip(bars, [paper_val, val]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005, f'{v:.4f}',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    diff = val - paper_val
    ax.set_title(f'{name}\nDifférence : {diff:+.4f} ({diff/paper_val*100:.1f}%)',
                 fontsize=10, fontweight='bold',
                 color='#2E7D32' if diff > 0 else '#C62828')
    ax.set_ylim(0, 1.1); ax.set_ylabel('Score')
    ax.grid(axis='y', alpha=0.3); ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig('9_global_summary_bands.png', dpi=150, bbox_inches='tight')
print(" Sauvegardé : 9_global_summary_bands.png")
plt.show()

def plot_geo_maps(y_true, y_pred):
    csv_files = {
        'Zone 1': 'df_zone1.csv',
        'Zone 2': 'df_zone2.csv',
    }
    missing = [f for f in csv_files.values() if not os.path.exists(f)]
    if missing:
        print(f"⚠️  Fichiers manquants : {missing}")
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
        'Californie — Vérité terrain / Prédiction / Erreurs climat',
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
 
        ax = axes[row, 0]
        ax.scatter(lon, lat, c=true_lbl, cmap=cmap_classes,
                   vmin=0, vmax=len(CLASSES)-1,
                   s=s, alpha=0.85, linewidths=0)
        ax.set_title(f'{zone} — Vérité terrain',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.grid(alpha=0.2); ax.set_facecolor('white')
 
        ax = axes[row, 1]
        ax.scatter(lon, lat, c=pred_lbl, cmap=cmap_classes,
                   vmin=0, vmax=len(CLASSES)-1,
                   s=s, alpha=0.85, linewidths=0)
        ax.set_title(f'{zone} — Prédiction',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.grid(alpha=0.2); ax.set_facecolor('white')
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
    plt.savefig('geo_maps_bands_CA.png', dpi=150, bbox_inches='tight')
    print("✅ Sauvegardé : geo_maps_bands_CA.png")
    plt.show()


plot_geo_maps(y_true, y_pred)

 