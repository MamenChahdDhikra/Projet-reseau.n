"""
MCTNet — Wang et al. (2024)
Californie : in_channels=10 (10 bandes uniquement, sans sol)
proj_channels=10, n_head=5 → 10/5=2 dims par tête
n_classes=6 (Grapes, Rice, Alfalfa, Almonds, Pistachios, Others)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
#  1. CNN SUB-MODULE
# ─────────────────────────────────────────────────────────────
class CNNModule(nn.Module):
    def __init__(self, in_channels,dropout=0.3):
        super(CNNModule, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels,
                               kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(in_channels)
        self.conv2 = nn.Conv1d(in_channels, in_channels,
                               kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(in_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        out = F.relu(self.bn1(self.conv1(x_t)))
        out = self.drop(out)  
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.drop(out)  
        out = out.transpose(1, 2)
        return out + x


# ─────────────────────────────────────────────────────────────
#  2. ECA MODULE
# ─────────────────────────────────────────────────────────────
class ECA(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1) #nedi la moyenne yweli vecteur de 10
        self.conv     = nn.Conv1d(1, 1,
                                  kernel_size=kernel_size, 
                                  padding=kernel_size // 2,
                                  bias=False)
        self.sigmoid  = nn.Sigmoid() #ndiro bin 0 1

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2)
        y = self.sigmoid(y)
        return x * y


# ─────────────────────────────────────────────────────────────
#  3. ALPE MODULE
# ─────────────────────────────────────────────────────────────
class ALPE(nn.Module):
    def __init__(self, d_model, max_len=36):
        super(ALPE, self).__init__()
        self.d_model = d_model

        pe       = torch.zeros(max_len, d_model) #matrice (d_model (10),max_len(36)) m3amer b 0
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term) #pair sin
        pe[:, 1::2] = torch.cos(position * div_term) #impaire cos
        self.register_buffer('pe', pe)

        self.conv = nn.Conv1d(d_model, d_model,
                              kernel_size=3, padding=1)
        self.eca  = ECA(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        batch, T, C = x.shape
        pe          = self.pe[:T, :].unsqueeze(0)\
                          .expand(batch, -1, -1) #n3amer pe
        mask_exp    = mask.unsqueeze(-1)
        pe          = pe * (1 - mask_exp) #ndir mask
        pe_t        = pe.transpose(1, 2)
        pe_t        = self.conv(pe_t) #ndirlo conv
        pe_t        = self.eca(pe_t) #ndirlo eca
        pe          = pe_t.transpose(1, 2)
        return self.norm(x + pe) #x la9dim m3a pe


# ─────────────────────────────────────────────────────────────
#  4. TRANSFORMER SUB-MODULE
# ─────────────────────────────────────────────────────────────
class TransformerModule(nn.Module):
    def __init__(self, d_model , n_head, d_ff,
                 max_len=36, use_alpe=False):
        super(TransformerModule, self).__init__()

        if use_alpe:
            self.pos_encoding = ALPE(d_model=d_model,
                                     max_len=max_len)
        else:
            self.pos_encoding = None

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=0.35,
            batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(d_ff, d_model),
            nn.Dropout(0.35)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        if self.pos_encoding is not None:
            x = self.pos_encoding(x, mask)

        attn_mask   = mask.bool()
        attn_out, _ = self.attention(
            x, x, x,
            key_padding_mask=attn_mask
        )
        x      = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x      = self.norm2(x + ff_out)
        return x


# ─────────────────────────────────────────────────────────────
#  5. CTFUSION
# ─────────────────────────────────────────────────────────────
class CTFusion(nn.Module):
    def __init__(self, d_model, n_head, d_ff,
                 max_len=36, use_alpe=False):
        super(CTFusion, self).__init__()
        self.cnn         = CNNModule(in_channels=d_model)
        self.transformer = TransformerModule(
            d_model=d_model,#nbr col
            n_head=n_head, #5
            d_ff=d_ff, #d_model -> d_ff(40,80,80*4)
            max_len=max_len, #nbr lignes
            use_alpe=use_alpe #rana stage 1 wela nn
        )

    def forward(self, x, mask):
        cnn_out   = self.cnn(x)
        trans_out = self.transformer(x, mask)
        return torch.cat([cnn_out, trans_out], dim=-1) #conca cnn trans


# ─────────────────────────────────────────────────────────────
#  6. MCTNET COMPLET — CALIFORNIE SANS SOL
# ─────────────────────────────────────────────────────────────
class MCTNet(nn.Module):
    """
    MCTNet — Wang et al. (2024) — Californie sans sol

    in_channels=10  (10 bandes uniquement)
    proj_channels=10, n_head=5 → 10/5=2 ✅
    n_classes=6 (Grapes, Rice, Alfalfa, Almonds, Pistachios, Others)

    Architecture :
      Projection  : (batch, 36, 10) → (batch, 36, 10)
      Stage 1     : CTFusion → (batch, 18, 20)
      Stage 2     : CTFusion → (batch,  9, 40)
      Stage 3     : CTFusion → (batch,  4, 80)
      GlobalMaxPool → (batch, 80)
      MLP         → (batch, 6)
    """
    def __init__(self,
                 in_channels=10,
                 n_classes=6,
                 n_head=5,
                 n_stage=3,
                 d_ff_mult=4,
                 proj_channels=10):
        super(MCTNet, self).__init__()

        self.n_stage    = n_stage
        self.input_proj = nn.Linear(in_channels, proj_channels)

        self.stages      = nn.ModuleList()
        current_channels = proj_channels
        max_len          = 36

        for i in range(n_stage):
            self.stages.append(
                CTFusion(
                    d_model=current_channels, #nbr col ta3 dekhla
                    n_head=n_head, #5
                    d_ff=current_channels * d_ff_mult, #li nesta3mlo fel ff 10->40,20->80...
                    max_len=max_len, #nbr ligne
                    use_alpe=(i == 0) #pour voir esq ndiro pe wela nn
                )
            )
            current_channels = current_channels * 2 #apres chaque stage le nbr de col yedoubla
            max_len          = max_len // 2         #""""""                    de ligne yete9sem

        self.classifier = nn.Sequential(
            nn.Linear(current_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, n_classes)
        )

    def forward(self, x, mask):
        x = self.input_proj(x) #13 ->10

        for i in range(self.n_stage):
            x    = self.stages[i](x, mask)
            x    = x[:, ::2, :]
            mask = mask[:, ::2]

        x = x.max(dim=1)[0]
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────
#  TEST
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  TEST MCTNet — Californie (10 canaux, 6 classes)")
    print("=" * 55)

    x    = torch.randn(32, 36, 10)
    mask = torch.zeros(32, 36)

    model = MCTNet(in_channels=10, n_classes=6,
                   n_head=5, n_stage=3, proj_channels=10)
    out   = model(x, mask)

    n_params = sum(p.numel() for p in model.parameters()
                   if p.requires_grad)

    print(f"  Entrée  : {x.shape}")
    print(f"  Sortie  : {out.shape} "
          f"{'✅' if out.shape == torch.Size([32,6]) else '❌'}")
    print(f"  Params  : {n_params:,}")
    print(f"\n  Dimensions :")
    print(f"    Entrée          : (batch, 36, 10)")
    print(f"    Projection 10→10: (batch, 36, 10)")
    print(f"    Stage 1         : (batch, 18, 20)")
    print(f"    Stage 2         : (batch,  9, 40)")
    print(f"    Stage 3         : (batch,  4, 80)")
    print(f"    GlobalMaxPool   : (batch, 80)")
    print(f"    MLP             : (batch,  6)")
    print("=" * 55)
