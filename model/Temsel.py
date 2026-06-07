import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Temporal domain: Depthwise-Separable 1D Conv Block ----
class TemporalConvBlock(nn.Module):
    def __init__(self, d_model=32, kernel_size=5, dropout=0.1, causal=False):
        super().__init__()
        padding = (kernel_size - 1) if causal else (kernel_size // 2)
        self.causal = causal
        self.dw = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, groups=d_model, bias=False)
        self.pw = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):  # x: [B,T,D]
        residual = x
        h = x.transpose(1, 2)  # [B,D,T]

        if self.causal:
            # Use left padding to implement causal convolution (avoid looking into the future)
            k = self.dw.kernel_size[0]
            pad = (k - 1, 0)  # left padding
            h = F.pad(h, pad)

        h = self.dw(h)
        h = self.pw(h)               # [B,D,T]
        h = h.transpose(1, 2)        # [B,T,D]
        h = self.act(h)
        h = self.drop(h)
        return self.ln(h + residual) # Residual + LN

# ---- Temporal self-attention (low-dimensional) ----
class TemporalEncoderLayer(nn.Module):
    def __init__(self, d_model=32, nhead=2, dropout=0.1, causal=False, ff_ratio=4):
        super().__init__()
        self.causal = causal

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_ratio * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):  # x: [B, T, D]
        B, T, D = x.shape

        h = self.ln1(x)

        if self.causal:
            mask = torch.full((T, T), float('-inf'), device=x.device)
            mask = torch.triu(mask, diagonal=1)
        else:
            mask = None

        h2, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + h2

        x = x + self.ff(self.ln2(x))

        return x


class TemporalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model=32,
        nhead=2,
        dropout=0.1,
        causal=False,
        ff_ratio=4,
        num_layers=1
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TemporalEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                causal=causal,
                ff_ratio=ff_ratio
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):  # x: [B, T, D]
        for layer in self.layers:
            x = layer(x)
        return x

class TinyTemporalSelector(nn.Module):
    """
    Input: x ∈ [B, T, 3, 150]
    Output:
        When use_softmax=True by default:
            cls_token ∈ [B, T, 768]
            rel_token ∈ [B, T, 768]

    ablation_mode:
        "full"                  : Full TG-Former
        "no_feature_encoder"    : Remove the Feature Encoder and keep only graph tokenization
        "no_graph_tokenizer"    : Do not use class/relation embeddings; directly use graph features
        "no_class_embedding"    : Remove class embeddings and keep only relation embeddings
        "no_relation_embedding" : Remove relation embeddings and keep only class embeddings
    """

    def __init__(
        self,
        cls_emb,
        rel_emb,
        d_model=32,
        nhead=2,
        dropout=0.1,
        causal=False,
        temperature=1.0,
        use_softmax=True,
        ablation_mode="full"
    ):
        super().__init__()

        self.temperature = temperature
        self.use_softmax = use_softmax
        self.d_model = d_model

        self.num_classes = cls_emb.shape[0]      # 150
        self.num_relations = rel_emb.shape[0]    # 50
        self.emb_dim = cls_emb.shape[1]          # usually 768

        assert cls_emb.shape[1] == rel_emb.shape[1], \
            "cls_emb and rel_emb must have the same embedding dimension"

        self.valid_modes = {
            "full",
            "no_feature_encoder",
            "no_graph_tokenizer",
            "no_class_embedding",
            "no_relation_embedding"
        }

        if ablation_mode not in self.valid_modes:
            raise ValueError(f"Invalid ablation_mode: {ablation_mode}")

        self.ablation_mode = ablation_mode

        in_dim = 3 * self.num_classes

        # 1) Feature Encoder: linear dimensionality reduction
        self.proj_in = nn.Sequential(
            nn.Linear(in_dim, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model)
        )
        self.ln_in = nn.LayerNorm(d_model)
        self.res_in = nn.Linear(in_dim, d_model)

        # 2) Local temporal modeling
        self.tcn = TemporalConvBlock(
            d_model=d_model,
            kernel_size=5,
            dropout=dropout,
            causal=causal
        )

        # 3) Global temporal modeling
        self.tsa = TemporalSelfAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            causal=causal,
            ff_ratio=4,
            num_layers=1
        )

        # 4) Graph Tokenizer: graph feature -> class / relation distribution
        self.proj_out = nn.Linear(d_model, self.num_classes)
        self.proj_rel = nn.Linear(d_model, self.num_relations)

        # For no_feature_encoder: bypass the Feature Encoder and obtain the relation distribution directly from the raw graph input
        # Class logits can be obtained directly by averaging the three channels of the raw x
        self.raw_rel_proj = nn.Linear(in_dim, self.num_relations)

        # For no_graph_tokenizer: do not use class/relation embeddings; directly map graph features to the semantic space
        self.graph_feat_proj_cls = nn.Linear(d_model, self.emb_dim)
        self.graph_feat_proj_rel = nn.Linear(d_model, self.emb_dim)

        # class / relation embedding
        self.cls_emb = nn.Parameter(cls_emb.detach().clone())
        self.rel_emb = nn.Parameter(rel_emb.detach().clone())

    def set_ablation_mode(self, mode: str):
        """
        The ablation mode can be dynamically switched during training or testing.
        """
        if mode not in self.valid_modes:
            raise ValueError(f"Invalid ablation_mode: {mode}")
        self.ablation_mode = mode

    def encode_feature(self, x_flat):
        """
        Feature Encoder + TCN + Temporal Self-Attention
        x_flat: [B, T, 450]
        return: [B, T, D]
        """
        h = self.proj_in(x_flat)
        h = self.ln_in(h + self.res_in(x_flat))

        h = self.tcn(h)
        h = self.tsa(h)

        return h

    def graph_tokenize(self, cls_logits, rel_logits):
        """
        Convert class/relation logits into semantic tokens.
        """
        if not self.use_softmax:
            return cls_logits, rel_logits

        cls_prob = torch.softmax(cls_logits / self.temperature, dim=-1)
        rel_prob = torch.softmax(rel_logits / self.temperature, dim=-1)

        cls_token = torch.matmul(cls_prob, self.cls_emb)  # [B, T, 768]
        rel_token = torch.matmul(rel_prob, self.rel_emb)  # [B, T, 768]

        return cls_token, rel_token

    def forward(self, x):
        """
        x: [B, T, 3, 150]
        """
        B, T, C, V = x.shape
        assert C == 3, f"Expected C=3, but got C={C}"
        assert V == self.num_classes, f"Expected V={self.num_classes}, but got V={V}"

        x = x.float()
        x_flat = x.reshape(B, T, C * V)  # [B, T, 450]

        mode = self.ablation_mode

        # --------------------------------------------------
        # 1. w/o Feature Encoder
        # Remove the Feature Encoder and bypass proj_in / TCN / TSA.
        # Generate class/relation tokens only from the raw graph input.
        # --------------------------------------------------
        if mode == "no_feature_encoder":
            # Raw input x: [B,T,3,150]
            # Directly average the three channels as the initial logits of the class distribution
            cls_logits = x.mean(dim=2)              # [B,T,150]
            rel_logits = self.raw_rel_proj(x_flat)  # [B,T,50]

            return self.graph_tokenize(cls_logits, rel_logits)

        # --------------------------------------------------
        # The following modes first pass through the Feature Encoder
        # --------------------------------------------------
        h = self.encode_feature(x_flat)  # [B,T,D]

        # --------------------------------------------------
        # 2. w/o Graph Tokenizer
        # Do not use class/relation embeddings and do not perform class/relation tokenization.
        # Directly map graph features to the embedding space.
        # --------------------------------------------------
        if mode == "no_graph_tokenizer":
            cls_token = self.graph_feat_proj_cls(h)  # [B,T,768]
            rel_token = self.graph_feat_proj_rel(h)  # [B,T,768]
            return cls_token, rel_token

        # --------------------------------------------------
        # Graph Tokenizer
        # --------------------------------------------------
        cls_logits = self.proj_out(h)  # [B,T,150]
        rel_logits = self.proj_rel(h)  # [B,T,50]

        cls_token, rel_token = self.graph_tokenize(cls_logits, rel_logits)

        # --------------------------------------------------
        # 3. w/o Class Embedding
        # Remove class embeddings and keep only relation embeddings。
        # Set cls_token to zero to keep the output interface unchanged.
        # --------------------------------------------------
        if mode == "no_class_embedding":
            cls_token = torch.zeros_like(cls_token)
            return cls_token, rel_token

        # --------------------------------------------------
        # 4. w/o Relation Embedding
        # Remove relation embeddings and keep only class embeddings。
        # Set rel_token to zero to keep the output interface unchanged.
        # --------------------------------------------------
        if mode == "no_relation_embedding":
            rel_token = torch.zeros_like(rel_token)
            return cls_token, rel_token

        # --------------------------------------------------
        # 5. Full TG-Former
        # --------------------------------------------------
        return cls_token, rel_token

if __name__ == '__main__':
    B, T = 4, 12
    x = F.one_hot(torch.randint(0, 150, (B, T, 3)), num_classes=150).float()  # [B,T,3,150]

    model = TinyTemporalSelector(
        d_model=32, nhead=2, dropout=0.1,
        causal=False, temperature=0.7, use_softmax=True
    )
    y,y2= model(x)   # [B,T,150]
    print(y.shape, y2.shape)

