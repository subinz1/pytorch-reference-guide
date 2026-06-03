"""
Model Architectures — ResNet, GPT, ViT
========================================
Covers: complete implementations of common architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=" * 60)
print("1. ResNet (Image Classification)")
print("=" * 60)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.fc(self.pool(x).flatten(1))
        return x

resnet = ResNet(BasicBlock, [2, 2, 2])
x = torch.randn(4, 3, 32, 32)
out = resnet(x)
params = sum(p.numel() for p in resnet.parameters())
print(f"ResNet-14: {params:,} params, input {x.shape} -> output {out.shape}")

print("\n" + "=" * 60)
print("2. GPT (Language Model)")
print("=" * 60)

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.SiLU(), nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )

    def forward(self, x):
        B, L, D = x.shape
        # Self-attention
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv.unbind(0)
        attn = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        x = x + self.out(attn.transpose(1, 2).reshape(B, L, D))
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4, d_ff=512, max_len=512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([GPTBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # Weight tying

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -512:])[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

gpt = GPT(vocab_size=1000)
tokens = torch.randint(0, 1000, (2, 20))
logits = gpt(tokens)
params = sum(p.numel() for p in gpt.parameters())
print(f"Mini-GPT: {params:,} params, input {tokens.shape} -> logits {logits.shape}")

# Generate
generated = gpt.generate(tokens, max_new_tokens=10)
print(f"Generated: {generated.shape}")

print("\n" + "=" * 60)
print("3. Vision Transformer (ViT)")
print("=" * 60)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3, embed_dim=256):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, num_classes=10,
                 embed_dim=256, depth=4, n_heads=4):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        n_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([GPTBlock(embed_dim, n_heads, embed_dim * 4) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, 0]))

vit = ViT()
x = torch.randn(4, 3, 32, 32)
out = vit(x)
params = sum(p.numel() for p in vit.parameters())
print(f"Mini-ViT: {params:,} params, input {x.shape} -> output {out.shape}")

print("\n" + "=" * 60)
print("PARAMETER COMPARISON")
print("=" * 60)

models = {
    "ResNet-14": resnet,
    "Mini-GPT (4L, d=256)": gpt,
    "Mini-ViT (4L, d=256)": vit,
}
for name, m in models.items():
    p = sum(p.numel() for p in m.parameters())
    mem = p * 4 / 1e6
    print(f"  {name:25s}: {p:>10,} params, {mem:>6.1f} MB (FP32)")

print("\nDone!")
