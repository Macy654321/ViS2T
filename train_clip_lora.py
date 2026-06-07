from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    import clip
except ImportError as exc:
    raise ImportError(
        "Missing dependency: clip. Install it in the target environment before running this script."
    ) from exc


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_ids_from_csv(csv_path: str | Path) -> list[str]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    values: list[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            value = row[0].strip()
            if not value:
                continue
            values.append(value)
    return values

class PandaFrameTextDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        frame_dir: str | Path,
        description_json: str | Path,
        preprocess,
        text_mode: str = "action",
        suffixes: tuple[int, ...] = (1, 2, 3),
        limit: int | None = None,
    ) -> None:
        self.frame_dir = Path(frame_dir)
        self.preprocess = preprocess
        self.text_mode = text_mode
        self.suffixes = suffixes

        with Path(description_json).open("r", encoding="utf-8") as handle:
            descriptions = json.load(handle)

        if text_mode not in {"action", "detail"}:
            raise ValueError(f"text_mode must be 'action' or 'detail', got: {text_mode}")

        samples: list[dict[str, str]] = []
        for video_id in read_ids_from_csv(csv_path):
            for suffix in suffixes:
                image_name = f"{video_id}_{suffix}.jpg"
                image_path = self.frame_dir / image_name
                if not image_path.exists():
                    continue

                desc_item = descriptions.get(image_name)
                if not isinstance(desc_item, dict):
                    continue

                text = desc_item.get(text_mode, "")
                if not isinstance(text, str) or not text.strip():
                    continue

                samples.append(
                    {
                        "image_name": image_name,
                        "image_path": str(image_path),
                        "text": text.strip(),
                        "video_id": video_id,
                    }
                )

        if not samples:
            raise ValueError(
                f"No valid samples found for csv={csv_path}, frame_dir={frame_dir}, text_mode={text_mode}."
            )

        if limit is not None:
            samples = samples[:limit]

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        image_tensor = self.preprocess(image)
        return {
            "image": image_tensor,
            "text": sample["text"],
            "image_name": sample["image_name"],
            "video_id": sample["video_id"],
        }

class LoRAQVMultiheadAttention(nn.Module):
    def __init__(
        self,
        base_attn: nn.MultiheadAttention,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if not isinstance(base_attn, nn.MultiheadAttention):
            raise TypeError("LoRAQVMultiheadAttention only supports nn.MultiheadAttention.")

        if base_attn.in_proj_weight is None:
            raise ValueError("Expected attention module with combined QKV in_proj_weight.")

        self.base = base_attn
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.embed_dim = base_attn.embed_dim

        # Translated comment.
        for param in self.base.parameters():
            param.requires_grad = False

        # LoRA for Q
        self.q_lora_A = nn.Parameter(
            torch.empty(rank, self.embed_dim, dtype=torch.float32)
        )
        self.q_lora_B = nn.Parameter(
            torch.zeros(self.embed_dim, rank, dtype=torch.float32)
        )

        # LoRA for V
        self.v_lora_A = nn.Parameter(
            torch.empty(rank, self.embed_dim, dtype=torch.float32)
        )
        self.v_lora_B = nn.Parameter(
            torch.zeros(self.embed_dim, rank, dtype=torch.float32)
        )

        nn.init.kaiming_uniform_(self.q_lora_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_lora_A, a=math.sqrt(5))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask=None,
        need_weights: bool = True,
        attn_mask=None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        base = self.base
        embed_dim = self.embed_dim

        # Translated comment.
        q_weight = base.in_proj_weight[:embed_dim]
        k_weight = base.in_proj_weight[embed_dim: 2 * embed_dim]
        v_weight = base.in_proj_weight[2 * embed_dim:]

        # Translated comment.
        delta_q = (self.q_lora_B @ self.q_lora_A) * self.scaling
        delta_v = (self.v_lora_B @ self.v_lora_A) * self.scaling

        delta_q = delta_q.to(dtype=q_weight.dtype, device=q_weight.device)
        delta_v = delta_v.to(dtype=v_weight.dtype, device=v_weight.device)

        # Translated comment.
        merged_in_proj_weight = torch.cat(
            [
                q_weight + delta_q,
                k_weight,
                v_weight + delta_v,
            ],
            dim=0,
        )

        return F.multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=embed_dim,
            num_heads=base.num_heads,
            in_proj_weight=merged_in_proj_weight,
            in_proj_bias=base.in_proj_bias,
            bias_k=base.bias_k,
            bias_v=base.bias_v,
            add_zero_attn=base.add_zero_attn,
            dropout_p=base.dropout,
            out_proj_weight=base.out_proj.weight,
            out_proj_bias=base.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=False,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

    def lora_state_dict(self, prefix: str) -> dict[str, torch.Tensor]:
        return {
            f"{prefix}.q_lora_A": self.q_lora_A.detach().cpu(),
            f"{prefix}.q_lora_B": self.q_lora_B.detach().cpu(),
            f"{prefix}.v_lora_A": self.v_lora_A.detach().cpu(),
            f"{prefix}.v_lora_B": self.v_lora_B.detach().cpu(),
        }

def get_module_by_name(root: nn.Module, module_name: str) -> nn.Module:
    module = root
    for part in module_name.split("."):
        module = getattr(module, part)
    return module


def set_module_by_name(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)

def freeze_model(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

def inject_lora_into_clip_visual(
    model,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> list[str]:
    replaced: list[str] = []

    for module_name, module in list(model.visual.named_modules()):
        if not module_name.endswith("attn"):
            continue
        if not isinstance(module, nn.MultiheadAttention):
            continue

        lora_module = LoRAQVMultiheadAttention(
            module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        set_module_by_name(model.visual, module_name, lora_module)
        replaced.append(module_name)

    if not replaced:
        raise ValueError("No visual attention modules were replaced by Q+V LoRA.")

    return replaced

def collect_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []

    for module in model.modules():
        if isinstance(module, LoRAQVMultiheadAttention):
            parameters.extend(
                [
                    module.q_lora_A,
                    module.q_lora_B,
                    module.v_lora_A,
                    module.v_lora_B,
                ]
            )

    return parameters

def build_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}

    for module_name, module in model.visual.named_modules():
        if isinstance(module, LoRAQVMultiheadAttention):
            state_dict.update(
                module.lora_state_dict(prefix=f"visual.{module_name}")
            )

    return state_dict

def clip_contrastive_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    targets = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
    image_loss = F.cross_entropy(logits_per_image, targets)
    text_loss = F.cross_entropy(logits_per_text, targets)
    return 0.5 * (image_loss + text_loss)


def collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    texts = [item["text"] for item in batch]
    image_names = [item["image_name"] for item in batch]
    video_ids = [item["video_id"] for item in batch]
    return {
        "image": images,
        "text": texts,
        "image_name": image_names,
        "video_id": video_ids,
    }


@torch.no_grad()
def evaluate(model, data_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    for batch in data_loader:
        images = batch["image"].to(device)
        text_tokens = clip.tokenize(batch["text"], truncate=True).to(device)
        logits_per_image, logits_per_text = model(images, text_tokens)
        loss = clip_contrastive_loss(logits_per_image, logits_per_text)
        total_loss += loss.item()
        total_steps += 1
    return total_loss / max(total_steps, 1)


def train_one_epoch(model, data_loader: DataLoader, optimizer, device: torch.device, log_interval: int) -> float:
    model.train()
    total_loss = 0.0
    total_steps = 0

    for step, batch in enumerate(data_loader, start=1):
        images = batch["image"].to(device)
        text_tokens = clip.tokenize(batch["text"], truncate=True).to(device)

        logits_per_image, logits_per_text = model(images, text_tokens)
        loss = clip_contrastive_loss(logits_per_image, logits_per_text)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_steps += 1

        if step % log_interval == 0:
            print(f"step={step} train_loss={loss.item():.4f}")

    return total_loss / max(total_steps, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for CLIP ViT-B/32 on panda frame-text pairs.")
    parser.add_argument("--train_csv", type=str, default="dataset/panda/train_data.csv")
    parser.add_argument("--test_csv", type=str, default="dataset/panda/test_data.csv")
    parser.add_argument("--frame_dir", type=str, default="dataset/panda/frames")
    parser.add_argument("--description_json", type=str, default="dataset/panda/frame_descriptions.json")
    parser.add_argument("--text_mode", type=str, choices=["action", "detail"], default="detail")
    parser.add_argument("--clip_model_name", type=str, default="ViT-B/32")
    parser.add_argument("--download_root", type=str, default="pretrained/vit")
    parser.add_argument("--save_path", type=str, default="logs/clip_lora_panda/lora_detail.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_test", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(args.clip_model_name, device=device, download_root=args.download_root)

    freeze_model(model)
    replaced_modules = inject_lora_into_clip_visual(
        model,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
    )
    model = model.to(device)
    trainable_parameters = collect_lora_parameters(model)
    if not trainable_parameters:
        raise ValueError("No trainable LoRA parameters were found.")

    train_dataset = PandaFrameTextDataset(
        csv_path=args.train_csv,
        frame_dir=args.frame_dir,
        description_json=args.description_json,
        preprocess=preprocess,
        text_mode=args.text_mode,
        limit=args.limit_train,
    )
    test_dataset = PandaFrameTextDataset(
        csv_path=args.test_csv,
        frame_dir=args.frame_dir,
        description_json=args.description_json,
        preprocess=preprocess,
        text_mode=args.text_mode,
        limit=args.limit_test,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)

    print(f"device={device}")
    print(f"text_mode={args.text_mode}")
    print(f"train_samples={len(train_dataset)} test_samples={len(test_dataset)}")
    print(f"lora_modules={len(replaced_modules)}")

    best_val_loss = float("inf")
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.log_interval)
        val_loss = evaluate(model, test_loader, device)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            lora_state = {
                "text_mode": args.text_mode,
                "rank": args.rank,
                "alpha": args.alpha,
                "dropout": args.dropout,
                "target_modules": list(replaced_modules),
                "state_dict": build_lora_state_dict(model),
            }
            torch.save(lora_state, save_path)
            print(f"saved_lora={save_path}")


if __name__ == "__main__":
    main()
