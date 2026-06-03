"""
Exported Graph Inspection
==========================

Shows how to examine the computation graph after export:
- Viewing generated code
- Walking graph nodes
- Listing all ATen operations
- Understanding the graph signature
- Comparing graphs of different models

Run:
    python export_inspection.py
"""

import torch
import torch.nn as nn
from torch.export import export


class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.bn(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class ResidualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(32, 32)
        self.linear2 = nn.Linear(32, 32)
        self.norm = nn.LayerNorm(32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x + residual)
        return x


class AttentionModel(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = x + self.ffn(x)
        return x


def demo_generated_code():
    print("=" * 60)
    print("  1. Generated Code")
    print("=" * 60)

    model = ConvModel().eval()
    exported = export(model, (torch.randn(1, 3, 32, 32),))

    print("  The exported graph compiles to Python-like code:")
    print()
    for line in exported.graph_module.code.strip().split("\n"):
        print(f"    {line}")
    print()


def demo_graph_nodes():
    print("=" * 60)
    print("  2. Graph Nodes")
    print("=" * 60)

    model = ResidualModel().eval()
    exported = export(model, (torch.randn(2, 32),))

    print("  Each node in the graph represents an operation:")
    print(f"  {'Op':<18s} {'Name':<25s} {'Target'}")
    print(f"  {'-'*18} {'-'*25} {'-'*40}")

    for node in exported.graph_module.graph.nodes:
        target_str = str(node.target)
        if len(target_str) > 50:
            target_str = target_str[:50] + "..."
        print(f"  {node.op:<18s} {node.name:<25s} {target_str}")

    print()


def demo_list_operations():
    print("=" * 60)
    print("  3. List All ATen Operations")
    print("=" * 60)

    models = {
        "ConvModel": (ConvModel().eval(), (torch.randn(1, 3, 32, 32),)),
        "ResidualModel": (ResidualModel().eval(), (torch.randn(2, 32),)),
    }

    for name, (model, example) in models.items():
        exported = export(model, example)

        ops = set()
        for node in exported.graph_module.graph.nodes:
            if node.op == "call_function":
                ops.add(str(node.target))

        print(f"  {name} ({len(ops)} unique operations):")
        for op in sorted(ops):
            op_short = op.split(".")[-1] if "." in op else op
            print(f"    {op_short}")
        print()


def demo_graph_signature():
    print("=" * 60)
    print("  4. Graph Signature")
    print("=" * 60)

    model = ConvModel().eval()
    exported = export(model, (torch.randn(1, 3, 32, 32),))

    sig = exported.graph_signature

    # Input specs
    print("  Input specifications:")
    params = []
    buffers = []
    user_inputs = []
    for spec in sig.input_specs:
        kind = spec.kind.name
        if kind == "PARAMETER":
            params.append(spec)
        elif kind == "BUFFER":
            buffers.append(spec)
        elif kind == "USER_INPUT":
            user_inputs.append(spec)

    print(f"    Parameters ({len(params)}):")
    for p in params:
        print(f"      {p.arg} → {p.target}")
    print(f"    Buffers ({len(buffers)}):")
    for b in buffers:
        print(f"      {b.arg} → {b.target}")
    print(f"    User inputs ({len(user_inputs)}):")
    for u in user_inputs:
        print(f"      {u.arg}")

    # Output specs
    print(f"\n  Output specifications ({len(sig.output_specs)}):")
    for spec in sig.output_specs:
        print(f"    {spec.kind.name}: {spec.arg}")

    print()


def demo_state_dict_inspection():
    print("=" * 60)
    print("  5. State Dict Inspection")
    print("=" * 60)

    model = ConvModel().eval()
    exported = export(model, (torch.randn(1, 3, 32, 32),))

    print("  Parameters and buffers in the exported model:")
    total_params = 0
    for key, tensor in exported.state_dict.items():
        numel = tensor.numel()
        total_params += numel
        print(f"    {key:<30s} shape={list(tensor.shape):<20s} numel={numel}")

    print(f"\n  Total parameters: {total_params:,}")
    print()


def demo_attention_graph():
    print("=" * 60)
    print("  6. Attention Model Graph (more complex)")
    print("=" * 60)

    model = AttentionModel(d_model=64, nhead=4).eval()
    exported = export(model, (torch.randn(2, 8, 64),))

    ops = {}
    for node in exported.graph_module.graph.nodes:
        if node.op == "call_function":
            op_name = str(node.target).split(".")[-1]
            ops[op_name] = ops.get(op_name, 0) + 1

    print("  Operation frequency in attention model:")
    for op_name, count in sorted(ops.items(), key=lambda x: -x[1]):
        bar = "#" * count
        print(f"    {op_name:<30s} {count:>3d} {bar}")

    # Count total nodes
    total = sum(1 for _ in exported.graph_module.graph.nodes)
    call_nodes = sum(1 for n in exported.graph_module.graph.nodes if n.op == "call_function")
    print(f"\n  Total graph nodes: {total}")
    print(f"  Call function nodes: {call_nodes}")
    print(f"  Unique operations: {len(ops)}")
    print()


def demo_comparing_models():
    print("=" * 60)
    print("  7. Comparing Model Graphs")
    print("=" * 60)

    models = {
        "ConvModel": (ConvModel().eval(), (torch.randn(1, 3, 32, 32),)),
        "ResidualModel": (ResidualModel().eval(), (torch.randn(2, 32),)),
        "AttentionModel": (AttentionModel().eval(), (torch.randn(2, 8, 64),)),
    }

    print(f"  {'Model':<20s} {'Nodes':>6s} {'Ops':>5s} {'Params':>10s}")
    print(f"  {'-'*20} {'-'*6} {'-'*5} {'-'*10}")

    for name, (model, example) in models.items():
        exported = export(model, example)

        total_nodes = sum(1 for _ in exported.graph_module.graph.nodes)
        unique_ops = len(set(
            str(n.target) for n in exported.graph_module.graph.nodes
            if n.op == "call_function"
        ))
        total_params = sum(p.numel() for p in model.parameters())

        print(f"  {name:<20s} {total_nodes:>6d} {unique_ops:>5d} {total_params:>10,d}")

    print()


def main():
    print("\nExported Graph Inspection")
    print("=" * 60)
    print()

    demo_generated_code()
    demo_graph_nodes()
    demo_list_operations()
    demo_graph_signature()
    demo_state_dict_inspection()
    demo_attention_graph()
    demo_comparing_models()

    print("All inspection demos completed!\n")


if __name__ == "__main__":
    main()
