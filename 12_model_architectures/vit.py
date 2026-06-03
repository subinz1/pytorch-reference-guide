"""
Vision Transformer (ViT) — Complete Implementation
===================================================

Implements ViT from "An Image Is Worth 16x16 Words: Transformers for Image
Recognition at Scale" (Dosovitskiy et al., 2020).

Key ideas:
- Split image into fixed-size patches
- Linearly embed each patch
- Add position embeddings and a CLS token
- Process with a standard Transformer encoder
- Classify using the CLS token's output
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Convert an image into a sequence of patch embeddings.

    Uses a single Conv2d with kernel_size=patch_size and stride=patch_size,
    which is equivalent to splitting into patches + linear projection but
    much more efficient.
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, d_model=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x):
        # x: (B, C, H, W) -> (B, d_model, H/P, W/P) -> (B, num_patches, d_model)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention (no causal masking for ViT)."""

    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, d_k)
        q, k, v = qkv.unbind(0)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_dropout(self.proj(out))


class TransformerBlock(nn.Module):
    """ViT Transformer block: pre-norm attention + pre-norm MLP."""

    def __init__(self, d_model, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for image classification.

    Architecture:
        PatchEmbedding -> prepend CLS token -> add position embeddings
        -> N x TransformerBlock -> LayerNorm -> MLP head on CLS token
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        d_model=768,
        num_heads=12,
        num_layers=12,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        num_patches = self.patch_embed.num_patches

        # Learnable CLS token: aggregates information from all patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Learnable position embeddings for CLS + all patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))

        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Classification head: LayerNorm -> Linear
        self.head = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: images, (batch, channels, height, width)
        Returns:
            logits: (batch, num_classes)
        """
        B = x.shape[0]

        # Patch embedding: (B, num_patches, d_model)
        x = self.patch_embed(x)

        # Prepend CLS token: (B, 1 + num_patches, d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embeddings
        x = self.pos_drop(x + self.pos_embed)

        # Transformer encoder
        x = self.blocks(x)
        x = self.norm(x)

        # Classification from CLS token (position 0)
        cls_output = x[:, 0]
        return self.head(cls_output)


# ---------------------------------------------------------------------------
# Standard ViT configurations from the paper
# ---------------------------------------------------------------------------

def vit_tiny(num_classes=1000, img_size=224, patch_size=16):
    """ViT-Tiny: 6 layers, 192 dim, 3 heads (~5.7M params)."""
    return VisionTransformer(
        img_size=img_size, patch_size=patch_size, num_classes=num_classes,
        d_model=192, num_heads=3, num_layers=12, mlp_ratio=4.0,
    )

def vit_small(num_classes=1000, img_size=224, patch_size=16):
    """ViT-Small: 12 layers, 384 dim, 6 heads (~22M params)."""
    return VisionTransformer(
        img_size=img_size, patch_size=patch_size, num_classes=num_classes,
        d_model=384, num_heads=6, num_layers=12, mlp_ratio=4.0,
    )

def vit_base(num_classes=1000, img_size=224, patch_size=16):
    """ViT-Base: 12 layers, 768 dim, 12 heads (~86M params)."""
    return VisionTransformer(
        img_size=img_size, patch_size=patch_size, num_classes=num_classes,
        d_model=768, num_heads=12, num_layers=12, mlp_ratio=4.0,
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    batch_size = 2
    img_size = 224
    num_classes = 10

    configs = {
        "ViT-Tiny": vit_tiny(num_classes=num_classes, img_size=img_size),
        "ViT-Small": vit_small(num_classes=num_classes, img_size=img_size),
    }

    x = torch.randn(batch_size, 3, img_size, img_size)

    for name, model in configs.items():
        model.eval()
        with torch.no_grad():
            logits = model(x)
        print(f"{name:12s} | params: {count_params(model):>12,} | "
              f"input: {list(x.shape)} -> output: {list(logits.shape)}")

    # Detailed shape trace for ViT-Tiny
    print("\nDetailed shape trace (ViT-Tiny):")
    model = configs["ViT-Tiny"]
    model.eval()
    with torch.no_grad():
        patches = model.patch_embed(x)
        print(f"  After patch embedding:   {list(patches.shape)}")
        B = x.shape[0]
        cls_tokens = model.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1)
        print(f"  After CLS token prepend: {list(tokens.shape)}")
        tokens = tokens + model.pos_embed
        print(f"  After position embed:    {list(tokens.shape)}")
        encoded = model.blocks(tokens)
        print(f"  After Transformer:       {list(encoded.shape)}")
        cls_out = model.norm(encoded)[:, 0]
        print(f"  CLS token output:        {list(cls_out.shape)}")
        logits = model.head(cls_out)
        print(f"  Final logits:            {list(logits.shape)}")

    num_patches = (img_size // 16) ** 2
    print(f"\n  Image {img_size}x{img_size} with patch size 16 "
          f"= {num_patches} patches + 1 CLS = {num_patches + 1} tokens")

    print("\nVision Transformer verified successfully!")
