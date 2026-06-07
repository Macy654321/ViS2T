import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import math
from typing import Optional
import torch.nn.functional as F
from .Temsel import *
from .fusion import *

CLASSES = ['airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
           'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
           'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
           'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
           'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
           'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
           'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
           'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
           'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
           'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
           'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
           'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
           'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
           'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

REL_CLASSES = ['above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
               'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
               'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
               'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
               'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
               'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

class VisualToTextProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(VisualToTextProjection, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # Map visual features to the GPT-2 input dimension through a linear layer

    def forward(self, visual_features):
        return self.fc(visual_features)

class MultimodalModel(nn.Module):
    def __init__(self, vit_model, gpt2_model, projection_layer, tokenizer):
        super(MultimodalModel, self).__init__()
        self.vit_model = vit_model           # Trainable visual encoder
        self.gpt2_model = gpt2_model         # GPT-2 can be frozen
        self.projection_layer = projection_layer  # Trainable linear projection
        self.tokenizer = tokenizer

    def forward(self, batch, mode='train', max_length=50):
        """
        batch: dict containing
            'visual': image tensor [B, N, C, H, W] or [B, C, H, W]
            'text': target text required in training mode
            'input_text': optional prompt text
        mode: 'train' or 'generate'
        """
        visual = batch['visual']  # Image input
        B = visual.shape[0]
        N = visual.shape[1]
        device = visual.device
        # 1. Extract visual features
        if visual.dim() == 5:  # Video sequence [B, N, C, H, W]

            features = []
            for b in range(B):
                frames = visual[b]  # [N, C, H, W]
                feat = self.vit_model(frames)  # [N, D]
                features.append(feat)
            visual_features = torch.stack(features, dim=0)  # [B, N, D]
        else:
            visual_features = self.vit_model(visual)  # [B, D]

        # 2. Project to the GPT-2 embedding dimension
        projected_features = self.projection_layer(visual_features)  # [B, gpt2_dim]

        # 3. Training mode: return logits for CrossEntropyLoss
        if mode == 'train':
            target_text = batch['text']
            attention_mask = batch['mask']
            embedding_text = self.gpt2_model.transformer.wte(target_text)
            inputs_embeds = torch.cat([projected_features, embedding_text], dim=1)
            dummy_token = torch.zeros((B, N), dtype=torch.int64, device=device)
            input_labels = torch.cat((dummy_token, target_text), dim=1)
            outputs = self.gpt2_model(inputs_embeds=inputs_embeds, labels=input_labels,attention_mask=attention_mask,ignore_index=0)
            return outputs  # outputs.loss, outputs.logits

        # 4. Generation mode: directly generate text
        elif mode == 'generate':
            # input_text = batch['input_text']
            # input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)
            # inputs_embeds = torch.cat([projected_features, self.gpt2_model.transformer.wte(input_ids)], dim=1)
            B, T, H = projected_features.size()  # [B, T, hidden_size]
            attn_mask = torch.ones(B, T, dtype=torch.long, device=projected_features.device)
            generated_ids = self.gpt2_model.generate(
                inputs_embeds=projected_features,
                max_length=max_length,
                attention_mask=attn_mask,
                num_beams=5,
                pad_token_id=self.tokenizer.pad_token_id,
                early_stopping=True
            )
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return generated_text




# ---------------------------
# 1) Temporal positional encoding (sinusoidal)
# ---------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)         # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)   # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)   # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)   # Odd dimensions
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)  # [1, T, C]


# -----------------------------------
# 2) 1D Transformer Encoder (stackable)
# -----------------------------------
class TemporalTransformer(nn.Module):
    def __init__(self,
                 d_model: int = 768,
                 nhead: int = 12,
                 num_layers: int = 2,
                 dim_feedforward: int = 3072,
                 dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C]
        key_padding_mask: [B, T]  True=ignored/is padding
        """
        x = self.encoder(x)  # [B, T, C]
        x = self.ln_out(x)
        return x


# ---------------------------
# 3) Basic feed-forward MLP
# ---------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# --------------------------------------------
# 4) A single Q-Former layer: self-attention + cross-attention + MLP
#    - Self-attention within query tokens
#    - Cross-attention: Q=query, K/V=video sequence
# --------------------------------------------
class QFormerLayer(nn.Module):
    def __init__(self, d_model: int = 768, nhead: int = 12, dropout: float = 0.1, mlp_ratio: float = 4.0):
        super().__init__()
        self.ln_q1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.ln_q2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.ln_q3 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, mlp_ratio, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                q_tokens: torch.Tensor,                 # [B, Nq, C]
                vid_tokens: torch.Tensor               # [B, T, C]
                ) -> torch.Tensor:
        # Self-attention within queries
        q = self.ln_q1(q_tokens)
        q_sa, _ = self.self_attn(q, q, q, need_weights=False)
        q_tokens = q_tokens + self.dropout(q_sa)

        # Cross-attention: Q=query, K/V=video
        q = self.ln_q2(q_tokens)
        k = vid_tokens
        v = vid_tokens
        q_ca, _ = self.cross_attn(q, k, v, need_weights=False)
        q_tokens = q_tokens + self.dropout(q_ca)

        # MLP
        q = self.ln_q3(q_tokens)
        q_ffn = self.mlp(q)
        q_tokens = q_tokens + q_ffn
        return q_tokens


class QFormer(nn.Module):
    def __init__(self,
                 d_model: int = 768,
                 nhead: int = 12,
                 num_layers: int = 4,
                 n_queries: int = 16,
                 dropout: float = 0.1,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.n_queries = n_queries
        self.query_embed = nn.Parameter(torch.randn(1, n_queries, d_model) / math.sqrt(d_model))
        self.layers = nn.ModuleList([
            QFormerLayer(d_model=d_model, nhead=nhead, dropout=dropout, mlp_ratio=mlp_ratio)
            for _ in range(num_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self,
                vid_tokens: torch.Tensor                     # [B, T, C]
                ) -> torch.Tensor:
        B = vid_tokens.size(0)
        q = self.query_embed.expand(B, -1, -1)  # [B, Nq, C]
        for layer in self.layers:
            q = layer(q, vid_tokens)
        q = self.ln_out(q)   # [B, Nq, C]
        return q


class TemporalAdapter(nn.Module):
    """
    Temporal-Adapter: a lightweight side module that models only along the temporal dimension
    x: [B, T, D]
    """
    def __init__(self, d_model=768, bottleneck=64, kernel_size=5, dropout=0.1, causal=False, use_glu=True):
        super().__init__()
        self.causal = causal
        self.use_glu = use_glu

        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, bottleneck)

        # Depthwise separable convolution: independent convolution for each channel along the temporal dimension
        padding = (kernel_size - 1) if causal else (kernel_size // 2)
        self.dwconv = nn.Conv1d(
            in_channels=bottleneck, out_channels=bottleneck,
            kernel_size=kernel_size, padding=padding, groups=bottleneck, bias=True
        )
        self.up = nn.Linear(bottleneck, d_model)

        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

        # Optional GLU gating for more stable gain control
        if use_glu:
            self.gate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Sigmoid()
            )
        else:
            self.gate = None

        # Initialization suggestion: set the up-projection close to zero to start with near-zero gain
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, x):   # x: [B,T,D]
        residual = x
        x = self.norm(x)

        h = self.down(x)           # [B,T,r]
        h = h.transpose(1, 2)      # [B,r,T]

        if self.causal:
            k = self.dwconv.kernel_size[0]
            h = F.pad(h, (k-1, 0))  # Left padding to avoid looking into the future

        h = self.dwconv(h)         # [B,r,T]
        h = h.transpose(1, 2)      # [B,T,r]
        h = self.act(h)
        h = self.drop(h)
        h = self.up(h)             # [B,T,D]
        h = self.drop(h)

        out = residual + h         # Residual
        if self.gate is not None:
            g = self.gate(out)
            out = residual + g * (out - residual)  # gated residual
        return out


class VideoToGPT2Tokens(nn.Module):
    def __init__(self,
                 in_dim: int = 768,         # Input video feature dimension
                 gpt2_dim: int = 768,       # GPT-2 hidden dimension (GPT-2 small/base = 768)
                 qformer_layers: int = 4,
                 qformer_heads: int = 12,
                 n_queries: int = 16,
                 dropout: float = 0.1):
        super().__init__()

        self.model_dim = in_dim

        self.qformer = QFormer(
            d_model=self.model_dim, nhead=qformer_heads,
            num_layers=qformer_layers, n_queries=n_queries,
            dropout=dropout, mlp_ratio=4.0
        )
        self.adapter = TemporalAdapter(d_model=self.model_dim, bottleneck=64, kernel_size=5, dropout=0.1, causal=False)
        self.choose = 1
        self.ln_out = nn.LayerNorm(gpt2_dim)

    @torch.no_grad()
    def _make_pad_mask(self, x: torch.Tensor, lengths: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Generate key_padding_mask from lengths: True=padding
        x: [B, T, C], lengths: [B]
        """
        if lengths is None:
            return None
        B, T = x.size(0), x.size(1)
        arange = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)  # [B, T]
        mask = arange >= lengths.unsqueeze(1)  # True=padding
        return mask

    def forward(self,
                vid_feats: torch.Tensor               # [B, T, in_dim]
                ) -> torch.Tensor:
        """
        Return: gpt2_prefix_tokens: [B, Nq, gpt2_dim]
        """
        x = vid_feats
        q_tokens = self.qformer(x)  # [B, Nq, C]
        if self.choose:
            q_tokens = self.adapter(q_tokens)
        gpt2_tokens = self.ln_out(q_tokens)
        return gpt2_tokens



class FeatureModel(nn.Module):
    def __init__(self, gpt2_model, projection_layer, tokenizer):
        super(FeatureModel, self).__init__()
        self.gpt2_model = gpt2_model         # GPT-2 can be frozen
        self.projection_layer = projection_layer  # Trainable linear projection
        self.tqformer = VideoToGPT2Tokens(
        in_dim=768, gpt2_dim=768,
        qformer_layers=4, qformer_heads=12,
        n_queries=24,
        dropout=0.1
    )
        self.tokenizer = tokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cls_emb = []
        for cls in CLASSES:
            target_text = tokenizer.encode(cls, add_special_tokens=False)
            target_text = torch.tensor(target_text, dtype=torch.long, device=device)
            embedding_text = gpt2_model.transformer.wte(target_text)
            embedding_text = embedding_text.mean(dim=0)
            self.cls_emb.append(embedding_text)
        self.cls_emb = torch.stack(self.cls_emb, dim=0).to(device)
        self.rel_emb = []
        for rel in REL_CLASSES:
            target_text = tokenizer.encode(rel, add_special_tokens=False)
            target_text = torch.tensor(target_text, dtype=torch.long, device=device)
            embedding_text = gpt2_model.transformer.wte(target_text)
            embedding_text = embedding_text.mean(dim=0)
            self.rel_emb.append(embedding_text)
        self.rel_emb = torch.stack(self.rel_emb, dim=0).to(device)
        self.temsel = TinyTemporalSelector(
            cls_emb=self.cls_emb,rel_emb=self.rel_emb,
            d_model=32, nhead=2, dropout=0.1,
            causal=False, temperature=0.7, use_softmax=True
        )

        self.fusion = FiLMPlusNoAttn()
        self.choose = 4 # 0:linear 1:qformer 2:tqformer 3:sg 4:fusion

    def forward(self, batch, mode='train', max_length=100):
        """
        batch: dict containing
            'visual': image features [B, N, C]
            'text': target text required in training mode
            'input_text': optional prompt text
        mode: 'train' or 'generate'
        """
        visual_features = batch['visual']  # Image input
        B = visual_features.shape[0]

        projected_features = self.projection_layer(visual_features)  # [B, gpt2_dim]
        device = visual_features.device
        if self.choose == 0: # linear
            prefixed_features = projected_features
        elif self.choose == 1: # q-former
            projected_features = self.tqformer(projected_features)
            prefixed_features = projected_features
        elif self.choose == 2: # tq-former
            projected_features = self.tqformer(projected_features)
            prefixed_features = projected_features
        elif self.choose == 3:  # sg
            projected_features = self.tqformer(projected_features)
            scene_graph = batch['graph']
            cls_token, rel_token = self.temsel(scene_graph)
            scene_feature = torch.cat([cls_token, rel_token], dim=1)
            # prefixed_features = torch.cat([scene_feature,projected_features], dim=1)
            prefixed_features = scene_feature+projected_features
        elif self.choose == 4: # fusion
            projected_features = self.tqformer(projected_features)
            scene_graph = batch['graph']
            cls_token, rel_token = self.temsel(scene_graph)
            interleaved = torch.stack([cls_token, rel_token], dim=2)
            scene_feature = interleaved.view(cls_token.size(0), -1, cls_token.size(-1))
            prefixed_features = self.fusion(projected_features,scene_feature)
        N = prefixed_features.shape[1]

        # 3. Training mode: return logits for CrossEntropyLoss
        if mode == 'train':
            target_text = batch['text']
            attention_mask = batch['mask']
            embedding_text = self.gpt2_model.transformer.wte(target_text)
            inputs_embeds = torch.cat([prefixed_features, embedding_text], dim=1)
            dummy_token = torch.full((B, N-1),50256, dtype=torch.int64, device=device)
            input_labels = torch.cat((dummy_token, target_text), dim=1)
            pad_token = torch.full((B, 1),50256, dtype=torch.int64, device=device)
            input_labels = torch.cat((input_labels, pad_token), dim=1)
            mask_token = torch.full((B, N-1), 1, dtype=torch.int64, device=device)
            attention_mask = torch.cat((mask_token,attention_mask), dim=1)
            outputs = self.gpt2_model(inputs_embeds=inputs_embeds, labels=input_labels, attention_mask=attention_mask)
            return outputs  # outputs.loss, outputs.logits

        # 4. Generation mode: directly generate text
        elif mode == 'generate':
            prefix_features = prefixed_features
            B, T, H = prefix_features.size()  # [B, T, hidden_size]
            attn_mask = torch.ones(B, T, dtype=torch.long, device=prefix_features.device)
            generated_ids = self.gpt2_model.generate(
                inputs_embeds=prefix_features,
                max_new_tokens=max_length,
                attention_mask=attn_mask,
                num_beams=5,
                # no_repeat_ngram_size=2,  # Prevent repetition (optional)
                # repetition_penalty=1.2,  # Prevent repetition (optional)
                length_penalty=1.0,
                eos_token_id=50256,
                pad_token_id=50256,
                do_sample=False,
                use_cache=True,
                early_stopping=True
            )
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return generated_text

def define_multimodal_model(vit_model_name, gpt2_model_path,vit_model_path,device):
    import clip

    gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_path).to(device)

    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path)
    # Set pad_token to eos_token
    tokenizer.pad_token = tokenizer.eos_token

    gpt2_model.config.pad_token_id = tokenizer.pad_token_id

    # Freeze all GPT-2 parameters
    for param in gpt2_model.parameters():
        param.requires_grad = False

    # Load the CLIP model (visual encoder)
    vit_model, _ = clip.load(vit_model_name, device=device,download_root=vit_model_path)
    visual = vit_model.visual
    visual = visual.to(torch.float)
    # Create the projection network
    projection_layer = VisualToTextProjection(input_dim=512, output_dim=gpt2_model.config.n_embd).to(device)

    return MultimodalModel(visual, gpt2_model, projection_layer, tokenizer) ,tokenizer

# Define the model function
def define_feature_model(gpt2_model_path, device, gpt2_mode=1):
    """
    gpt2_mode:
        1: Use pretrained GPT-2 and allow training
        2: Use the GPT-2 architecture with randomly initialized parameters
        3: Use pretrained GPT-2 with frozen parameters
    """

    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    if gpt2_mode == 1:
        # 1. Use pretrained GPT-2 weights and allow training
        gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_path)

    elif gpt2_mode == 2:
        # 2. Use the GPT-2 config with randomly initialized weights
        config = GPT2Config.from_pretrained(gpt2_model_path)
        gpt2_model = GPT2LMHeadModel(config)

    elif gpt2_mode == 3:
        # 3. Use pretrained GPT-2 weights with frozen parameters
        gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_path)

        for param in gpt2_model.parameters():
            param.requires_grad = False

    else:
        raise ValueError("gpt2_mode must be 1, 2, or 3.")

    gpt2_model = gpt2_model.to(device)

    gpt2_model.config.pad_token_id = tokenizer.pad_token_id

    # Create a mapping layer from visual features to the GPT-2 token embedding space
    projection_layer = VisualToTextProjection(
        input_dim=512,
        output_dim=gpt2_model.config.n_embd
    ).to(device)

    return FeatureModel(gpt2_model, projection_layer, tokenizer), tokenizer

if __name__ == "__main__":
    B, T, C = 2, 8, 768
    vid = torch.randn(B, T, C)

    model = VideoToGPT2Tokens(
        in_dim=768, gpt2_dim=768,  # If your GPT-2 is base/small
        qformer_layers=4, qformer_heads=12,
        n_queries=16,  # Generate 16 prefix tokens
        dropout=0.1
    )

    gpt2_prefix = model(vid)  # [B, 16, 768]
    print(gpt2_prefix.shape)

    # # Set device (GPU or CPU)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    # # Define model paths and the ViT model name
    # gpt2_model_path = r"E:\pycharmProject\clipcap\pretrained\gpt2"
    # vit_model_path = r"E:\pycharmProject\clipcap\pretrained\vit"
    # vit_model_name = 'ViT-B/32'  # Other ViT models can be selected as needed
    #
    # # Call the function to define the model
    # multimodal_model = define_multimodal_model(vit_model_name, gpt2_model_path,vit_model_path, device)
    # feature_model = define_feature_model(gpt2_model_path, device)
    # # Example image: replace this with actual image input
    # image = torch.randn(1,10,3, 224, 224)  # Assume the input is a randomly generated 3x224x224 image; replace it with a real image
    # image = image.to(device)
    # text = "a panda walks here"
    # # Input text prompt
    # input_text = "Describe the panda in the video"
    #
    # batch ={'visual':image, 'text':text, 'input_text':input_text}
    #
    # generated_description = multimodal_model(batch)
    # print(generated_description.loss)

