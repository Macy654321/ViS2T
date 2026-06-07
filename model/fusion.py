import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossGateFusion(nn.Module):
    """
    Input:
        x1: [B, N1, D]   (e.g., visual tokens)
        x2: [B, N2, D]   (e.g., relation/text tokens)
    Output:
        y_fused: [B, N1+N2, D]  # output after gated fusion plus residual concatenation with the original input
    """
    def __init__(self, d_model=768, nhead=8, dropout=0.1):
        super().__init__()
        self.attn_1to2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)  # x1←x2
        self.attn_2to1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)  # x2←x1

        # Gating: generate a per-dimension gate for each token based on [original token || cross-attention token]
        self.gate1 = nn.Sequential(
            nn.LayerNorm(2*d_model),
            nn.Linear(2*d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.gate2 = nn.Sequential(
            nn.LayerNorm(2*d_model),
            nn.Linear(2*d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        # Normalization after the residual connection
        self.ln_out = nn.LayerNorm(d_model)

        # Optional: use a lightweight FFN for further refinement; remove it if unnecessary
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x1, x2, mask1=None, mask2=None):
        """
        mask*: Optional key_padding_mask with shape [B, N*], where True indicates padding (invisible)
        """
        # Bidirectional cross-attention
        # x1 <- x2
        c12, _ = self.attn_1to2(query=x1, key=x2, value=x2, key_padding_mask=mask2)  # [B,N1,D]
        # x2 <- x1
        c21, _ = self.attn_2to1(query=x2, key=x1, value=x1, key_padding_mask=mask1)  # [B,N2,D]

        g1 = self.gate1(torch.cat([x1, c12], dim=-1))   # [B,N1,D]
        g2 = self.gate2(torch.cat([x2, c21], dim=-1))   # [B,N2,D]
        m1 = g1 * c12                                   # [B,N1,D]
        m2 = g2 * c21                                   # [B,N2,D]

        fused = torch.cat([m1, m2], dim=1)              # [B,N1+N2,D]
        raw   = torch.cat([x1, x2], dim=1)              # [B,N1+N2,D]


        y = self.ln_out(raw + fused)
        y = y + self.ffn(self.ln_out(y))                # This line can be commented out for a lighter model

        return  y



class TRGCA(nn.Module):
    """
    Temporal-Relational Gated Cross-Attention
    x1: [B, N1, D]  # primary branch
    x2: [B, N2, D]  # [object + relation] summary from 8 frames, N2≈16
    frame_ids: [N2]  # 0..7 frame index（two tokens per frame: Obj/Rel）
    type_ids:  [N2]  # 0:Obj, 1:Rel
    """
    def __init__(self, d=768, nhead=8, k_select=8, n_frames=8, dropout=0.1):
        super().__init__()
        assert d % nhead == 0
        self.d = d
        self.h = nhead
        self.dk = d // nhead
        self.k_select = k_select

        # Learnable frame-position and type embeddings
        self.frame_emb = nn.Embedding(n_frames, d)
        self.type_emb  = nn.Embedding(2, d)  # 0=Obj, 1=Rel

        # Linear projections for attention
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d, d)
        self.proj_drop = nn.Dropout(dropout)

        # Gated fusion + feed-forward network
        self.gate = nn.Sequential(
            nn.Linear(2*d, d), nn.GELU(),
            nn.Linear(d, 1)  # scalar gate（can also be changedforper-channel gate：output d）
        )
        self.ffn = nn.Sequential(
            nn.Linear(d, 4*d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4*d, d), nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        frame_ids = torch.tensor([0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7], dtype=torch.long)
        type_ids  = torch.tensor([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1], dtype=torch.long)
        self.register_buffer("frame_ids", frame_ids)
        self.register_buffer("type_ids",  type_ids)

    def forward(self, x1, x2, mask2=None):
        B, N1, D = x1.shape
        _, N2, _ = x2.shape

        # 1) Add frame/type embeddings to token2; if not provided, default to all-zero frame/type IDs
        frame_ids = self.frame_ids
        type_ids = self.type_ids
        x2 = x2 + self.frame_emb(frame_ids)[None, :, :] + self.type_emb(type_ids)[None, :, :]

        # 2) Compute scores and perform Top-k selection; each token1 attends to only k token2 tokens
        Q = self.Wq(x1)  # [B,N1,D]
        K = self.Wk(x2)  # [B,N2,D]
        V = self.Wv(x2)  # [B,N2,D]

        # Split into multiple heads
        Qh = Q.view(B, N1, self.h, self.dk).transpose(1, 2)        # [B,h,N1,dk]
        Kh = K.view(B, N2, self.h, self.dk).transpose(1, 2)        # [B,h,N2,dk]
        Vh = V.view(B, N2, self.h, self.dk).transpose(1, 2)        # [B,h,N2,dk]

        # Scoring
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / (self.dk ** 0.5)  # [B,h,N1,N2]

        # Optional: mask2 (True=keep), set padding positions to -inf
        if mask2 is not None:
            # mask2: [B,N2] -> [B,1,1,N2]
            m2 = (~mask2).unsqueeze(1).unsqueeze(2)  # True=PAD
            scores = scores.masked_fill(m2, float('-inf'))

        # Top-k indices
        k = min(self.k_select, N2)
        topk_scores, topk_idx = torch.topk(scores, k, dim=-1)  # [B,h,N1,k]

        # Select the corresponding K/V and perform sparse attention
        # gather: [B,h,N1,k,dk]
        # Kh_sel = torch.gather(Kh, dim=2, index=topk_idx.unsqueeze(-1).expand(-1,-1,-1,-1,self.dk))
        Vh_exp = Vh.unsqueeze(2).expand(-1, -1, topk_idx.size(2), -1, -1)

        # Then gather along dim=3 (the N2 dimension) using topk_idx to obtain [B, h, N1, k, dk]
        Vh_sel = torch.gather(
            Vh_exp, dim=3,
            index=topk_idx.unsqueeze(-1).expand(-1, -1, -1, -1, Vh.size(-1))
        )

        attn = F.softmax(topk_scores, dim=-1)                       # [B,h,N1,k]
        attn = self.attn_drop(attn)
        # Weighted sum
        yh = torch.einsum('bhnt,bhntd->bhnd', attn, Vh_sel)         # [B,h,N1,dk]
        y  = yh.transpose(1, 2).contiguous().view(B, N1, D)         # [B,N1,D]
        y  = self.proj_drop(self.proj(y))

        # 3) Gated fusion (conservative by default: initialize the last-layer bias positive, or multiply by a negative constant first)
        g = torch.sigmoid(self.gate(torch.cat([x1, y], dim=-1)))    # [B,N1,1]
        fused = x1 * g + y * (1 - g)

        # 4) Residual connection + FFN
        out = self.norm1(fused + x1)   # Keep the early output closer to the original token1
        out = self.norm2(out + self.ffn(out))
        return out
class LiteFiLMXAttn(nn.Module):
    """
    x1: [B, N1, D]  # primary branch
    x2: [B, N2, D]  # [object + relation] summary from 8 frames, N2≈16
    """
    def __init__(self, d=768, nhead=4, n_frames=8, dropout=0.1, adapter_rank=32):
        super().__init__()
        assert d % nhead == 0
        self.d = d
        self.h = nhead
        self.dk = d // nhead

        # Lightweight temporal/type embeddings with very few parameters
        self.frame_emb = nn.Embedding(n_frames, d)
        self.type_emb  = nn.Embedding(2, d)  # 0=Obj, 1=Rel
        frame_ids = torch.tensor([0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7], dtype=torch.long)
        type_ids  = torch.tensor([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1], dtype=torch.long)
        self.register_buffer("frame_ids", frame_ids)
        self.register_buffer("type_ids",  type_ids)

        # 1) FiLM condition: generate gamma/beta from the token2 summary
        hidden = max(d // 8, 64)  # Small bottleneck
        self.cond_pool = nn.LayerNorm(d)
        self.cond_mlp  = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(),
            nn.Linear(hidden, 2*d)  # -> [gamma, beta]
        )

        # 2) Lightweight cross-attention without Top-k
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d, d)
        self.proj_drop = nn.Dropout(dropout)

        # 3) Small adapter in a LoRA-style down-up projection
        self.adapter = nn.Sequential(
            nn.Linear(d, adapter_rank, bias=False),
            nn.GELU(),
            nn.Linear(adapter_rank, d, bias=False),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4*d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4*d, d), nn.Dropout(dropout)
        )

    def forward(self, x1, x2, mask2=None):
        """
        mask2: [B, N2] (True=valid)，pass None if unavailable
        """
        B, N1, D = x1.shape
        _, N2, _ = x2.shape

        # ---- Add lightweight temporal/type embeddings to token2 ----
        x2 = x2 + self.frame_emb(self.frame_ids)[None, :, :] + self.type_emb(self.type_ids)[None, :, :]

        # ---- 1) Global condition -> FiLM ----
        # Global summary: mean + LN (can also be replaced with attention pooling)
        c = self.cond_pool(x2.mean(dim=1))          # [B, D]
        gamma, beta = torch.chunk(self.cond_mlp(c), 2, dim=-1)  # [B, D], [B, D]
        gamma = gamma.unsqueeze(1)                   # [B, 1, D]
        beta  = beta.unsqueeze(1)                    # [B, 1, D]
        x1_film = x1 * (1 + torch.tanh(gamma)) + beta  # Stable affine transformation (tanh prevents explosion)

        # ---- 2) Lightweight full cross-attention; N2=16 is small and stable----
        Q = self.Wq(x1_film)  # [B,N1,D]
        K = self.Wk(x2)       # [B, N2,D]
        V = self.Wv(x2)       # [B, N2,D]

        Qh = Q.view(B, N1, self.h, self.dk).transpose(1, 2)     # [B,h,N1,dk]
        Kh = K.view(B, N2, self.h, self.dk).transpose(1, 2)     # [B,h,N2,dk]
        Vh = V.view(B, N2, self.h, self.dk).transpose(1, 2)     # [B,h,N2,dk]

        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / (self.dk ** 0.5)  # [B,h,N1,N2]
        if mask2 is not None:
            # mask2: True=valid；convert to key_padding_mask semantics：True=PAD
            m2 = (~mask2).unsqueeze(1).unsqueeze(2)  # [B,1,1,N2]
            scores = scores.masked_fill(m2, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        yh = torch.matmul(attn, Vh)                                # [B,h,N1,dk]
        y  = yh.transpose(1, 2).contiguous().view(B, N1, D)        # [B,N1,D]
        y  = self.proj_drop(self.proj(y))

        # Residual 1: FiLM-modulated backbone + cross-attention + lightweight adapter
        out = self.norm1(x1 + y + self.adapter(x1_film))

        # 3) FFN
        out = self.norm2(out + self.ffn(out))
        return out

class ConcatLinearFusion(nn.Module):
    """
    Fuse two token sequences:
    - x1: [B, N1, D]  # primary branch token
    - x2: [B, N2, D]  # auxiliary tokens
    Fusion method: Concat -> Linear -> Residual
    """
    def __init__(self, d=768):
        super().__init__()
        self.proj = nn.Linear(2*d, d)   # dimension after input concatenation 2D -> output D
        self.norm = nn.LayerNorm(d)

    def forward(self, x1, x2):

        fused = torch.cat([x1, x2], dim=-1)  # [B, N1, 2D]

        out = self.proj(fused)                        # [B, N1, D]
        out = self.norm(x1 + out)

        return out

# *****
class FiLMfusion(nn.Module):
    """
    Keep only FiLM modulation:
    - x1: [B, N1, D]  # primary branch
    - x2: [B, N2, D]  # auxiliary branch (object + relation summary from 8 frames, N2≈16)
    """
    def __init__(self, d=768, n_frames=8, dropout=0.1):
        super().__init__()
        self.d = d

        # Temporal/type embeddings
        self.frame_emb = nn.Embedding(n_frames, d)
        self.type_emb  = nn.Embedding(2, d)  # 0=Obj, 1=Rel
        frame_ids = torch.tensor([0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7], dtype=torch.long)
        type_ids  = torch.tensor([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1], dtype=torch.long)
        self.register_buffer("frame_ids", frame_ids)
        self.register_buffer("type_ids",  type_ids)

        # FiLM condition generates gamma/beta
        hidden = max(d // 8, 64)
        self.cond_pool = nn.LayerNorm(d)
        self.cond_mlp  = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(),
            nn.Linear(hidden, 2*d)  # output [gamma, beta]
        )

        # Post-processing
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x1, x2):

        # for token2 addTemporal/type embeddings
        x2 = x2 + self.frame_emb(self.frame_ids)[None, :, :] + self.type_emb(self.type_ids)[None, :, :]

        # Pool to obtain the global condition vector
        c = self.cond_pool(x2.mean(dim=1))  # [B, D]

        # Generate gamma and beta
        gamma, beta = torch.chunk(self.cond_mlp(c), 2, dim=-1)  # [B, D], [B, D]
        gamma = gamma.unsqueeze(1)  # [B, 1, D]
        beta  = beta.unsqueeze(1)   # [B, 1, D]

        # FiLM modulation
        x1_film = x1 * (1 + torch.tanh(gamma)) + beta

        # Residual connection + LN
        out = self.norm(x1 + self.dropout(x1_film))
        return out

class CrossAttentionFusion(nn.Module):
    """
    Standalone cross-attention fusion:
    - x1: [B, N1, D]  # primary branchsequence
    - x2: [B, N2, D]  # auxiliary sequence (N2≈16)
    outputand x1 same shape [B, N1, D]
    """
    def __init__(self, d=768, nhead=8, dropout=0.1):
        super().__init__()
        assert d % nhead == 0
        self.d = d
        self.h = nhead
        self.dk = d // nhead

        # Q, K, V projections
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d, d)
        self.proj_drop = nn.Dropout(dropout)

        # Residual connection + FFN
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4*d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4*d, d), nn.Dropout(dropout)
        )

    def forward(self, x1, x2, mask2=None):
        B, N1, D = x1.shape
        _, N2, _ = x2.shape

        # 1) Q,K,V
        Q = self.Wq(x1)  # [B,N1,D]
        K = self.Wk(x2)  # [B,N2,D]
        V = self.Wv(x2)  # [B,N2,D]

        # Split into multiple heads
        Qh = Q.view(B, N1, self.h, self.dk).transpose(1, 2)  # [B,h,N1,dk]
        Kh = K.view(B, N2, self.h, self.dk).transpose(1, 2)  # [B,h,N2,dk]
        Vh = V.view(B, N2, self.h, self.dk).transpose(1, 2)  # [B,h,N2,dk]

        # 2) Attention scores
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / (self.dk ** 0.5)  # [B,h,N1,N2]

        # mask2: True=valid, False=pad
        if mask2 is not None:
            m2 = (~mask2).unsqueeze(1).unsqueeze(2)  # [B,1,1,N2]
            scores = scores.masked_fill(m2, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        # 3) Weighted sum
        yh = torch.matmul(attn, Vh)  # [B,h,N1,dk]
        y  = yh.transpose(1, 2).contiguous().view(B, N1, D)  # [B,N1,D]
        y  = self.proj_drop(self.proj(y))

        # 4) Residual connection + FFN
        out = self.norm1(x1 + y)
        out = self.norm2(out + self.ffn(out))
        return out

class ConcatEncoderFusion(nn.Module):
    """
    Two-branch token fusion: Concat -> (TypeEmb + PosEmb) -> TransformerEncoder
    Input:
        x1: [B, N1, D]   # primary branch/modalityA
        x2: [B, N2, D]   # auxiliary/modality B
        mask1: [B, N1]   # True=valid, False=pad，can be None
        mask2: [B, N2]   # True=valid, False=pad，can be None
    Output:
        fused_all:  [B, N1+N2, D]  # the entire fused sequence
        fused_x1:   [B, N1, D]     # slice back to the modality-A part
        fused_x2:   [B, N2, D]     # slice back to the modality-B part
    """
    def __init__(self,
                 d=768,
                 nhead=8,
                 depth=2,
                 dim_ff=3072,
                 dropout=0.1,
                 max_len=50):
        super().__init__()
        self.d = d

        # Learnable positional and modality-type embeddings
        self.pos_emb  = nn.Embedding(max_len, d)   # absolute position
        self.type_emb = nn.Embedding(2, d)         # 0: x1, 1: x2

        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,   # makeInputoutputboth are [B, L, D]
            activation="gelu",
            norm_first=True     # Pre-Norm is more stable
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.dropout = nn.Dropout(dropout)

        # Normalization
        self.norm_in  = nn.LayerNorm(d)
        self.norm_out = nn.LayerNorm(d)

    def forward(self, x1, x2, mask1=None, mask2=None):
        B, N1, D = x1.shape
        _, N2, _ = x2.shape
        assert D == self.d, f"input dim {D} != model dim {self.d}"

        x   = torch.cat([x1, x2], dim=1)  # [B, N1+N2, D]
        L   = N1 + N2

        # 2) Modality-type embeddings
        type_ids = torch.cat([
            torch.zeros(N1, dtype=torch.long, device=x.device),
            torch.ones(N2,  dtype=torch.long, device=x.device)
        ], dim=0)                               # [N1+N2]
        type_emb = self.type_emb(type_ids)[None, :, :]  # [1, L, D]

        # 3) positional embeddings（absolute position，can be replaced with relative-position implementation）
        pos_ids = torch.arange(L, device=x.device, dtype=torch.long)  # [L]
        pos_emb = self.pos_emb(pos_ids)[None, :, :]                   # [1, L, D]

        x = self.norm_in(x + type_emb + pos_emb)
        x = self.dropout(x)

        # 4) Construct key_padding_mask (True=pad)
        if mask1 is None:
            mask1 = torch.ones(B, N1, dtype=torch.bool, device=x.device)
        if mask2 is None:
            mask2 = torch.ones(B, N2, dtype=torch.bool, device=x.device)
        mask_cat = torch.cat([mask1, mask2], dim=1)  # [B, L], True=valid

        key_padding_mask = ~mask_cat                 # True=pad

        # 5) Encode and fuse
        fused_all = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [B,L,D]
        fused_all = self.norm_out(fused_all)


        return fused_all


import torch
import torch.nn as nn

class SeparateEncodersAddFusionEqLen(nn.Module):
    """
    For equal-length sequences only: separate encoding -> additive fusion
      x1: [B, N, D]
      x2: [B, N, D]
      mask1: [B, N]  True=valid, False=pad（can be None）
      mask2: [B, N]  True=valid, False=pad（can be None）

    Parameters:
      d: hidden dimension
      nhead: number of attention heads
      dim_ff: FFN width
      dropout: dropout rate
      depth: number of layers in each encoder
      share: whether the two encoders share weights (True saves parameters)
    """
    def __init__(self, d=768, nhead=8, dim_ff=3072, dropout=0.1, depth=1, share=False):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        if share:
            self.enc1 = self.enc2 = nn.TransformerEncoder(enc_layer, num_layers=depth)
        else:
            self.enc1 = nn.TransformerEncoder(enc_layer, num_layers=depth)
            self.enc2 = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d, nhead=nhead, dim_feedforward=dim_ff,
                    dropout=dropout, activation='gelu',
                    batch_first=True, norm_first=True
                ),
                num_layers=depth
            )

        self.norm = nn.LayerNorm(d)

        # Optional: a lightweight learnable fusion weight (gate), disabled by default
        self.use_gate = False
        self.gate = nn.Sequential(
            nn.Linear(2*d, d), nn.GELU(),
            nn.Linear(d, 1), nn.Sigmoid()
        )

    def enable_gate(self, flag=True):
        self.use_gate = flag
        return self

    def forward(self, x1, x2, mask1=None, mask2=None):
        B, N, D = x1.shape
        assert x2.shape[1] == N and x2.shape[2] == D, "Equal-length fusion requires x1 and x2 to have the same shape"

        if mask1 is None:
            mask1 = torch.ones(B, N, dtype=torch.bool, device=x1.device)
        if mask2 is None:
            mask2 = torch.ones(B, N, dtype=torch.bool, device=x2.device)

        pad1 = ~mask1  # True=pad
        pad2 = ~mask2

        x1_enc = self.enc1(x1, src_key_padding_mask=pad1)  # [B,N,D]
        x2_enc = self.enc2(x2, src_key_padding_mask=pad2)  # [B,N,D]

        if self.use_gate:
            # Learn a per-token fusion ratio: out = g*x1_enc + (1-g)*x2_enc
            g = self.gate(torch.cat([x1_enc, x2_enc], dim=-1))  # [B,N,1]
            out = g * x1_enc + (1.0 - g) * x2_enc
        else:
            out = x1_enc + x2_enc

        return self.norm(out)

class FiLMPlusNoAttn(nn.Module):
    """
    Enhanced FiLM module without cross-attention:
      - x2 uses lightweight pooling without attention: mean / mean+max / conv1d
      - condition vector c -> generate AdaLN gamma/beta
      - residual scaling + gating for stable injection
    Shapes:
      x1: [B, N1, D]   (primary branch)
      x2: [B, N2, D]   (auxiliary branch, N2≈16)
      mask2: [B, N2] (True=valid) can be None
    """
    def __init__(self, d=768, n_frames=8, dropout=0.1,
                 pool_type="conv",     # "mean" | "meanmax" | "conv"
                 conv_kernel=3, conv_stride=1,
                 hidden_ratio=8,
                 use_type_time_emb=True):
        super().__init__()
        self.d = d
        self.pool_type = pool_type
        self.use_type_time_emb = use_type_time_emb

        # —— Optional：Temporal/type embeddings（lightweight）——
        if use_type_time_emb:
            self.frame_emb = nn.Embedding(n_frames, d)
            self.type_emb  = nn.Embedding(2, d)  # 0=Obj, 1=Rel
            frame_ids = torch.tensor([0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7], dtype=torch.long)
            type_ids  = torch.tensor([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1], dtype=torch.long)
            self.register_buffer("frame_ids", frame_ids)
            self.register_buffer("type_ids",  type_ids)

        # —— x2 Lightweight pooling without attention——
        if pool_type == "conv":
            # Apply 1D convolution along the temporal dimension (N2), then global pooling
            # Input [B,N2,D] -> transpose [B,D,N2]
            self.temporal_conv = nn.Conv1d(d, d, kernel_size=conv_kernel,
                                           stride=conv_stride, padding=conv_kernel//2, groups=1)
            self.bn = nn.BatchNorm1d(d)

        # —— Condition vector -> AdaLN parameters —— #
        hidden = max(d // hidden_ratio, 64)
        # Apply simple normalization first to stabilize statistics
        self.cond_norm = nn.LayerNorm(d)
        # For meanmax, concatenate first and then linearly project to d
        in_dim = d if pool_type != "meanmax" else 2*d
        self.cond_proj = nn.Linear(in_dim, d) if in_dim != d else nn.Identity()

        self.cond_mlp = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(),
            nn.Linear(hidden, 2*d)  # -> gamma, beta
        )
        self.x1_norm = nn.LayerNorm(d)  # AdaLN standard LN before AdaLN

        # —— Residual scaling + gating —— #
        self.dropout = nn.Dropout(dropout)
        self.res_scale = nn.Parameter(torch.tensor(0.1))  # small initial value for stable injection
        self.gate = nn.Sequential(
            nn.Linear(2*d, d), nn.GELU(),
            nn.Linear(d, 1), nn.Sigmoid()
        )
        self.out_norm = nn.LayerNorm(d)

    def _pool_x2(self, x2, mask2=None):
        """
        Pool x2: [B,N2,D] into c: [B,D]，without using attention。
        """
        B, N2, D = x2.shape
        if mask2 is not None:
            # Set pad to zero，to facilitate averaging；and countvalidlength
            valid = mask2.unsqueeze(-1).to(x2.dtype)    # [B,N2,1]
            x2_masked = x2 * valid
            denom = valid.sum(dim=1).clamp_min(1.0)     # [B,1,1]
        else:
            x2_masked = x2
            denom = torch.full((B,1,1), float(N2), device=x2.device, dtype=x2.dtype)

        if self.pool_type == "mean":
            c = x2_masked.sum(dim=1) / denom.squeeze(1)        # [B,D]
            return c

        elif self.pool_type == "meanmax":
            mean_vec = (x2_masked.sum(dim=1) / denom.squeeze(1))         # [B,D]
            if mask2 is not None:
                # Use very small values at padding positions to avoid affecting max pooling
                x2_for_max = x2.masked_fill(~mask2.unsqueeze(-1), float("-inf"))
            else:
                x2_for_max = x2
            max_vec, _ = x2_for_max.max(dim=1)                              # [B,D]
            max_vec = torch.nan_to_num(max_vec, nan=0.0, neginf=0.0)        # prevent -inf
            c_cat = torch.cat([mean_vec, max_vec], dim=-1)                  # [B,2D]
            return self.cond_proj(c_cat)                                    # [B,D]

        elif self.pool_type == "conv":
            # [B,N2,D] -> [B,D,N2] -> Conv1d -> BN -> GELU -> global average pooling
            z = x2.transpose(1, 2)                                          # [B,D,N2]
            z = self.temporal_conv(z)                                       # [B,D,N2]
            z = self.bn(z)
            z = F.gelu(z)
            if mask2 is not None:
                # Set padding positions to zero, then compute weighted mean
                valid = mask2.to(z.dtype).unsqueeze(1)                      # [B,1,N2]
                z = z * valid
                denom = valid.sum(dim=2, keepdim=True).clamp_min(1.0)       # [B,1,1]
            else:
                denom = torch.tensor(N2, dtype=z.dtype, device=z.device).view(1,1,1)
            c = z.sum(dim=2, keepdim=False) / denom.squeeze(2)              # [B,D]
            return c

        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

    def forward(self, x1, x2, mask2=None):
        # Optional: add temporal/type embeddings to x2
        if self.use_type_time_emb:
            x2 = x2 + self.frame_emb(self.frame_ids)[None, :, :] + self.type_emb(self.type_ids)[None, :, :]

        # 1) Lightweight pooling without attention -> condition vector c
        c = self._pool_x2(x2, mask2)                # [B,D]
        c = self.cond_norm(c)

        # 2) AdaLN parameters
        gamma, beta = torch.chunk(self.cond_mlp(c), 2, dim=-1)   # [B,D], [B,D]
        gamma = torch.tanh(gamma).unsqueeze(1)                   # [B,1,D]  bounded for better stability
        beta  = beta.unsqueeze(1)                                 # [B,1,D]

        # 3) AdaLN modulate x1
        x1_ln   = self.x1_norm(x1)                # [B,N1,D]
        x1_adaln = x1_ln * (1.0 + gamma) + beta   # [B,N1,D]

        # 4) Residual scaling + gating
        g = self.gate(torch.cat([x1, x1_adaln], dim=-1))   # [B,N1,1]
        fused = x1 + self.res_scale * self.dropout(g * x1_adaln)

        return self.out_norm(fused)
