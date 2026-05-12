import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation sur la dimension spectrale (canaux).

    ENTRÉE : (batch, C, T)  — après convolution, avant activation finale
    SORTIE : (batch, C, T)  — mêmes dims, canaux ré-pondérés

    Paramètres :
      channels : nombre de canaux (= nombre de bandes/features)
      reduction : facteur de réduction du goulot (r dans le papier SE)
                  petit r = plus d'expressivité, plus de paramètres
                  grand r = plus léger, moins expressif
                  r=4 est le compromis standard du papier SE original
    """
    def __init__(self, channels, reduction=4):
        super(SEModule, self).__init__()

        bottleneck = max(channels // reduction, 1)

        self.squeeze   = nn.AdaptiveAvgPool1d(1)   # (batch, C, T) → (batch, C, 1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x : (batch, C, T)
        batch, C, T = x.shape

        # Squeeze : résumer chaque canal sur toute la dim temporelle
        s = self.squeeze(x).squeeze(-1)      # (batch, C)

        # Excitation : apprendre l'importance de chaque canal
        e = self.excitation(s)               # (batch, C)
        e = e.unsqueeze(-1)                  # (batch, C, 1) → broadcast sur T

        # Recalibration : pondérer chaque canal
        return x * e                         # (batch, C, T)


# ─────────────────────────────────────────────────────────────
#  2. CNN SUB-MODULE + SE  (seul changement vs baseline)
# ─────────────────────────────────────────────────────────────
class CNNModuleSE(nn.Module):
    """
    CNN sub-module avec connexion résiduelle + SE spectral attention.

    ENTRÉE : (batch, T, C)
    SORTIE : (batch, T, C)

    Différence vs CNNModule baseline :
      Après la 2ème convolution, avant l'activation finale,
      on applique SE pour ré-pondérer les canaux spectraux.
    """
    def __init__(self, in_channels, dropout=0.3, se_reduction=4):
        super(CNNModuleSE, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(in_channels)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(in_channels)
        self.drop  = nn.Dropout(dropout)

        # ← NOUVEAUTÉ : SE après la 2ème conv
        self.se = SEModule(in_channels, reduction=se_reduction)

    def forward(self, x):
        # x : (batch, T, C)
        x_t = x.transpose(1, 2)                   # (batch, C, T)

        out = F.relu(self.bn1(self.conv1(x_t)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))

        # ← NOUVEAUTÉ : attention spectrale avant l'activation finale
        out = self.se(out)                         # (batch, C, T)

        out = F.relu(out)
        out = self.drop(out)
        out = out.transpose(1, 2)                  # (batch, T, C)

        return out + x                             # connexion résiduelle


# ─────────────────────────────────────────────────────────────
#  3. ECA MODULE  (identique au baseline)
# ─────────────────────────────────────────────────────────────
class ECA(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv     = nn.Conv1d(1, 1, kernel_size=kernel_size,
                                  padding=kernel_size // 2, bias=False)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2)
        return x * self.sigmoid(y)


# ─────────────────────────────────────────────────────────────
#  4. ALPE MODULE  (identique au baseline)
# ─────────────────────────────────────────────────────────────
class ALPE(nn.Module):
    def __init__(self, d_model, max_len=36):
        super(ALPE, self).__init__()
        self.d_model = d_model

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.eca  = ECA(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        batch, T, C = x.shape
        pe       = self.pe[:T, :].unsqueeze(0).expand(batch, -1, -1)
        mask_exp = mask.unsqueeze(-1)
        pe       = pe * (1 - mask_exp)
        pe_t     = pe.transpose(1, 2)
        pe_t     = self.conv(pe_t)
        pe_t     = self.eca(pe_t)
        pe       = pe_t.transpose(1, 2)
        return self.norm(x + pe)


# ─────────────────────────────────────────────────────────────
#  5. TRANSFORMER SUB-MODULE  (identique au baseline)
# ─────────────────────────────────────────────────────────────
class TransformerModule(nn.Module):
    def __init__(self, d_model, n_head, d_ff, max_len=36, use_alpe=False):
        super(TransformerModule, self).__init__()

        self.pos_encoding = ALPE(d_model=d_model, max_len=max_len) \
                            if use_alpe else None

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_head,
            dropout=0.35, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(d_ff, d_model), nn.Dropout(0.35)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        if self.pos_encoding is not None:
            x = self.pos_encoding(x, mask)

        attn_out, _ = self.attention(
            x, x, x, key_padding_mask=mask.bool()
        )
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


# ─────────────────────────────────────────────────────────────
#  6. CTFUSION + SE  (CNN remplacé par CNNModuleSE)
# ─────────────────────────────────────────────────────────────
class CTFusionSE(nn.Module):
    """
    CTFusion avec SE dans le CNN sub-module.
    ENTRÉE : (batch, T, C)
    SORTIE : (batch, T, C*2)
    """
    def __init__(self, d_model, n_head, d_ff, max_len=36,
                 use_alpe=False, se_reduction=4):
        super(CTFusionSE, self).__init__()

        # ← CNN avec SE à la place du CNN simple
        self.cnn         = CNNModuleSE(in_channels=d_model,
                                       se_reduction=se_reduction)
        self.transformer = TransformerModule(
            d_model=d_model, n_head=n_head, d_ff=d_ff,
            max_len=max_len, use_alpe=use_alpe
        )

    def forward(self, x, mask):
        cnn_out   = self.cnn(x)
        trans_out = self.transformer(x, mask)
        return torch.cat([cnn_out, trans_out], dim=-1)


# ─────────────────────────────────────────────────────────────
#  7. MCTNET + SE  (architecture complète améliorée)
# ─────────────────────────────────────────────────────────────
class MCTNetSE(nn.Module):
    """
    MCTNet + SE Spectral Attention (Wang et al., 2024 + idée papier 2)

    Seule différence avec MCTNet baseline :
      CTFusion → CTFusionSE  (CNN sub-module enrichi du SE)

    Architecture :
      Projection  : (batch, 36, in_channels) → (batch, 36, proj_channels)
      Stage 1     : CTFusionSE(ALPE=True)  + MaxPool → (batch, 18, proj*2)
      Stage 2     : CTFusionSE(ALPE=False) + MaxPool → (batch,  9, proj*4)
      Stage 3     : CTFusionSE(ALPE=False) + MaxPool → (batch,  4, proj*8)
      GlobalMaxPool → (batch, proj*8)
      MLP → (batch, n_classes)

    Paramètres :
      in_channels   : nombre de features d'entrée (10 S2, ou 13 avec ERA5)
      n_classes     : 5 Arkansas / 6 California
      proj_channels : dimension de projection (doit être divisible par n_head)
      se_reduction  : facteur de réduction SE (4 = standard, 2 = plus expressif)
    """
    def __init__(self, in_channels=10, n_classes=5, n_head=5,
                 n_stage=3, d_ff_mult=4, proj_channels=10, se_reduction=4):
        super(MCTNetSE, self).__init__()

        self.n_stage    = n_stage
        self.input_proj = nn.Linear(in_channels, proj_channels)
        self.stages     = nn.ModuleList()

        current_channels = proj_channels
        max_len          = 36

        for i in range(n_stage):
            self.stages.append(
                CTFusionSE(
                    d_model      = current_channels,
                    n_head       = n_head,
                    d_ff         = current_channels * d_ff_mult,
                    max_len      = max_len,
                    use_alpe     = (i == 0),
                    se_reduction = se_reduction
                )
            )
            current_channels = current_channels * 2
            max_len          = max_len // 2

        self.classifier = nn.Sequential(
            nn.Linear(current_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, n_classes)
        )

    def forward(self, x, mask):
        x = self.input_proj(x)

        for i in range(self.n_stage):
            x    = self.stages[i](x, mask)
            x    = x[:, ::2, :]
            mask = mask[:, ::2]

        x = x.max(dim=1)[0]
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────
#  TEST ET COMPARAISON DES PARAMÈTRES
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from model_se import MCTNetSE as MCTNetBaseline

    print("=" * 60)
    print("  TEST MCTNetSE — MCTNet + SE Spectral Attention")
    print("=" * 60)

    x    = torch.randn(32, 36, 10)
    mask = torch.zeros(32, 36)
    mask[:, 2] = 1; mask[:, 5] = 1  # simuler quelques données manquantes

    # Baseline
    baseline = MCTNetBaseline(in_channels=10, n_classes=6,
                               n_head=5, n_stage=3, proj_channels=10)
    # Amélioré
    enhanced = MCTNetSE(in_channels=10, n_classes=6,
                        n_head=5, n_stage=3, proj_channels=10,
                        se_reduction=4)

    out_base = baseline(x, mask)
    out_enh  = enhanced(x, mask)

    n_base = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    n_enh  = sum(p.numel() for p in enhanced.parameters() if p.requires_grad)

    print(f"\n  Baseline MCTNet :")
    print(f"    Sortie  : {out_base.shape} "
          f"{'✅' if out_base.shape == torch.Size([32,6]) else '❌'}")
    print(f"    Params  : {n_base:,}")

    print(f"\n  MCTNetSE (+ SE spectral attention) :")
    print(f"    Sortie  : {out_enh.shape} "
          f"{'✅' if out_enh.shape == torch.Size([32,6]) else '❌'}")
    print(f"    Params  : {n_enh:,}")

    delta = n_enh - n_base
    print(f"\n  Paramètres supplémentaires : +{delta:,} "
          f"({delta/n_base*100:.1f}% d'augmentation)")

    print(f"\n  SE Modules dans le modèle :")
    for name, module in enhanced.named_modules():
        if isinstance(module, SEModule):
            channels = module.excitation[0].in_features
            print(f"    {name} — channels={channels}")

    print("\n" + "=" * 60)
    print("  ✅ MCTNetSE prêt !")
    print("=" * 60)