 
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             cohen_kappa_score, f1_score,
                             precision_score, recall_score)
from torch.utils.data import DataLoader
from p import prepare_all_data_with_topo, CropDatasetCov
from model_ca_bands import MCTNet

CLASSES = ['Grapes', 'Rice', 'Alfalfa',
           'Almonds', 'Pistachios', 'Others']
COLORS  = ['#7B1FA2', '#0288D1', '#388E3C',
           '#F57C00', '#FBC02D', '#9E9E9E']
DEVICE  = 'cpu'
 
print("Chargement des données...")
torch.serialization.add_safe_globals([CropDatasetCov])

if os.path.exists('datasets_with_topo.pt'):
    datasets = torch.load('datasets_with_topo.pt',
                          weights_only=False)
    print("✅ datasets_with_topo.pt chargé !")
else:
    datasets = prepare_all_data_with_topo()

loader_test = DataLoader(datasets['test'],
                         batch_size=64, shuffle=False)

print("Chargement du modèle...")
model = MCTNet(in_channels=12, n_classes=6,
               n_head=5, n_stage=3,
               proj_channels=10).to(DEVICE)
model.load_state_dict(torch.load('best_model_cov.pt',
                                 weights_only=True))
model.eval()
 
y_true, y_pred = [], []
with torch.no_grad():
    for x_batch, mask_batch, y_batch in loader_test:
        outputs = model(x_batch.to(DEVICE),
                        mask_batch.to(DEVICE))
        preds   = outputs.argmax(dim=1)
        y_true += y_batch.tolist()
        y_pred += preds.cpu().tolist()

y_true = np.array(y_true)
y_pred = np.array(y_pred)

oa       = accuracy_score(y_true, y_pred)
kappa    = cohen_kappa_score(y_true, y_pred)
f1       = f1_score(y_true, y_pred, average='macro')
f1_cls   = f1_score(y_true, y_pred, average=None)
prec_cls = precision_score(y_true, y_pred, average=None,
                            zero_division=0)
rec_cls  = recall_score(y_true, y_pred, average=None,
                         zero_division=0)

print(f"\n{'='*50}")
print(f"  OA    : {oa:.4f}   (papier: 0.8524)")
print(f"  Kappa : {kappa:.4f}   (papier: 0.8194)")
print(f"  F1    : {f1:.4f}   (papier: 0.8301)")
print(f"{'='*50}")
 
def plot_confusion_matrix():
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Matrice de Confusion — MCTNet + Topo\n'
                 f'OA={oa:.4f} | Kappa={kappa:.4f} | F1={f1:.4f}',
                 fontsize=13, fontweight='bold')

    for ax, data, title, cmap in [
        (axes[0], cm_norm, 'Matrice normalisée',       'Blues'),
        (axes[1], cm,      'Matrice valeurs absolues', 'Greens')
    ]:
        im = ax.imshow(data, interpolation='nearest',
                       cmap=cmap,
                       vmin=0, vmax=1 if cmap=='Blues' else None)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(CLASSES, rotation=30,
                           ha='right', fontsize=9)
        ax.set_yticklabels(CLASSES, fontsize=9)
        ax.set_xlabel('Classe prédite', fontsize=11)
        ax.set_ylabel('Classe réelle', fontsize=11)
        ax.set_title(title, fontweight='bold')
        for i in range(6):
            for j in range(6):
                val = data[i, j]
                ref = 0.5 if cmap == 'Blues' else cm[i,:].sum()*0.5
                color = 'white' if val > ref else 'black'
                txt = f'{val:.3f}' if cmap == 'Blues' else str(val)
                ax.text(j, i, txt, ha='center', va='center',
                        color=color, fontsize=8,
                        fontweight='bold' if i==j else 'normal')

    plt.tight_layout()
    plt.savefig('6_confusion_matrix_topo.png', dpi=150,
                bbox_inches='tight')
    print("✅ Sauvegardé : 6_confusion_matrix_topo.png")
    plt.show()
 
def plot_metrics_per_class():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Métriques par classe — MCTNet + Topo',
                 fontsize=13, fontweight='bold')

    x = np.arange(6)
    for ax, vals, title in [
        (axes[0], f1_cls,   'F1 Score'),
        (axes[1], prec_cls, 'Précision'),
        (axes[2], rec_cls,  'Rappel'),
    ]:
        bars = ax.bar(x, vals, 0.6, color=COLORS,
                      alpha=0.85, edgecolor='white', linewidth=1.2)
        ax.axhline(y=0.8301 if title=='F1 Score' else vals.mean(),
                   color='red', linestyle='--', linewidth=1.5,
                   label='Papier (0.8301)' if title=='F1 Score'
                   else f'Moyenne ({vals.mean():.3f})')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01, f'{val:.3f}',
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(CLASSES, rotation=25,
                           ha='right', fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(f'{title}\nmoyenne={vals.mean():.4f}',
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('7_metrics_per_class_topo.png', dpi=150,
                bbox_inches='tight')
    print("✅ Sauvegardé : 7_metrics_per_class_topo.png")
    plt.show()
 
def plot_comparison_table():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    fig.suptitle('Comparaison Résultats — MCTNet + Topo',
                 fontsize=13, fontweight='bold', y=1.02)

    rows = [[cls, f'{f1_cls[i]:.4f}', f'{prec_cls[i]:.4f}',
             f'{rec_cls[i]:.4f}', str(int(np.sum(y_true==i)))]
            for i, cls in enumerate(CLASSES)]
    rows += [
        ['GLOBAL (macro)', f'{f1:.4f}  (papier: 0.8301)',
         f'{prec_cls.mean():.4f}', f'{rec_cls.mean():.4f}',
         str(len(y_true))],
        ['OA',    f'{oa:.4f}  (papier: 0.8524)', '', '', ''],
        ['Kappa', f'{kappa:.4f}  (papier: 0.8194)', '', '', ''],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=['Classe','F1 (obtenu)','Précision','Rappel','Support'],
        loc='center', cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)
    for j in range(5):
        table[0,j].set_facecolor('#1565C0')
        table[0,j].set_text_props(color='white', fontweight='bold')
    for i in range(6):
        table[i+1,0].set_facecolor(COLORS[i])
        table[i+1,0].set_text_props(color='white', fontweight='bold')
        color = ('#C8E6C9' if f1_cls[i]>=0.90
                 else '#FFF9C4' if f1_cls[i]>=0.80
                 else '#FFCCBC')
        table[i+1,1].set_facecolor(color)
    for i in range(6,9):
        for j in range(5):
            table[i+1,j].set_facecolor('#E3F2FD')
            table[i+1,j].set_text_props(fontweight='bold')

    plt.tight_layout()
    plt.savefig('8_comparison_table_topo.png', dpi=150,
                bbox_inches='tight')
    print("✅ Sauvegardé : 8_comparison_table_topo.png")
    plt.show()
 
def plot_global_summary():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Résumé Global — MCTNet + Topo vs Papier',
                 fontsize=13, fontweight='bold')

    for (name, val, paper_val), ax in zip(
        [('OA', oa, 0.8524),
         ('Kappa', kappa, 0.8194),
         ('F1 macro', f1, 0.8301)], axes
    ):
        bars = ax.bar(['Papier','Obtenu'], [paper_val, val],
                      color=['#90CAF9','#1565C0'], width=0.5,
                      edgecolor='white', linewidth=2)
        for bar, v in zip(bars, [paper_val, val]):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.005, f'{v:.4f}',
                    ha='center', va='bottom',
                    fontsize=13, fontweight='bold')
        diff = val - paper_val
        ax.set_title(f'{name}\nDifférence : {diff:+.4f} '
                     f'({diff/paper_val*100:.1f}%)',
                     fontsize=10, fontweight='bold',
                     color='#2E7D32' if diff>0 else '#C62828')
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig('9_global_summary_topo.png', dpi=150,
                bbox_inches='tight')
    print("✅ Sauvegardé : 9_global_summary_topo.png")
    plt.show()
 
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  VISUALISATION RÉSULTATS — MCTNet + Topo")
    print("="*55)
    plot_confusion_matrix()
    plot_metrics_per_class()
    plot_comparison_table()
    plot_global_summary()
  
