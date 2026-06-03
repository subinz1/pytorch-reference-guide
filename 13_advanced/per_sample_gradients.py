"""
Per-Sample Gradients with vmap + grad
=====================================

Standard backpropagation gives you the *average* gradient over a batch.
Sometimes you need the gradient for *each sample individually* — for example:
- Differential privacy (DP-SGD): clip per-sample gradients before averaging
- Influence functions: measure how much each training point affects the model
- Fisher information: compute per-sample gradient outer products

This file shows how to compute per-sample gradients efficiently using
torch.func.vmap and torch.func.grad.
"""

import torch
import torch.nn as nn
from torch.func import vmap, grad


def demo_naive_per_sample_gradients():
    """The slow way: loop over samples one at a time."""
    print("=" * 60)
    print("NAIVE PER-SAMPLE GRADIENTS (loop)")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

    batch_size = 8
    X = torch.randn(batch_size, 10)
    Y = torch.randn(batch_size, 1)

    per_sample_grads = []
    for i in range(batch_size):
        model.zero_grad()
        pred = model(X[i:i+1])
        loss = (pred - Y[i:i+1]).pow(2).sum()
        loss.backward()

        sample_grads = {}
        for name, param in model.named_parameters():
            sample_grads[name] = param.grad.clone()
        per_sample_grads.append(sample_grads)

    # Each entry in per_sample_grads is a dict of {param_name: gradient}
    param_name = "0.weight"
    stacked = torch.stack([g[param_name] for g in per_sample_grads])
    print(f"  Per-sample gradient for '{param_name}': {list(stacked.shape)}")
    print(f"  (batch_size, out_features, in_features) = "
          f"({batch_size}, 32, 10)")


def demo_vmap_per_sample_gradients():
    """The fast way: vmap + grad + functional_call."""
    print("\n" + "=" * 60)
    print("EFFICIENT PER-SAMPLE GRADIENTS (vmap + grad)")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

    batch_size = 8
    X = torch.randn(batch_size, 10)
    Y = torch.randn(batch_size, 1)

    # Step 1: Extract parameters as a dict
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    # Step 2: Define a stateless loss function that takes params explicitly
    def compute_loss(params, buffers, x, y):
        # functional_call runs the model with the given params
        pred = torch.func.functional_call(model, (params, buffers), (x.unsqueeze(0),))
        return ((pred - y.unsqueeze(0)) ** 2).sum()

    # Step 3: grad computes gradient of loss w.r.t. params (argument 0)
    grad_fn = grad(compute_loss)

    # Step 4: vmap vectorizes over the batch dimension of x and y
    # in_dims=(None, None, 0, 0) means:
    #   params: not batched (shared across samples)
    #   buffers: not batched
    #   x: batched along dim 0
    #   y: batched along dim 0
    per_sample_grads = vmap(grad_fn, in_dims=(None, None, 0, 0))(
        params, buffers, X, Y,
    )

    print("  Per-sample gradient shapes:")
    for name, g in per_sample_grads.items():
        print(f"    {name:15s}: {list(g.shape)}")

    # Verify: the mean of per-sample gradients should equal the batch gradient
    model.zero_grad()
    pred = model(X)
    loss = ((pred - Y) ** 2).sum()
    loss.backward()

    print("\n  Verification (mean of per-sample grads == batch grad):")
    for name, param in model.named_parameters():
        mean_grad = per_sample_grads[name].mean(dim=0)
        match = torch.allclose(mean_grad, param.grad, atol=1e-5)
        print(f"    {name:15s}: match={match}")

    return per_sample_grads


def demo_gradient_clipping():
    """Simulated DP-SGD: clip per-sample gradients, then average."""
    print("\n" + "=" * 60)
    print("DP-SGD SIMULATION: Per-Sample Gradient Clipping")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

    batch_size = 16
    X = torch.randn(batch_size, 10)
    Y = torch.randn(batch_size, 1)

    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def compute_loss(params, buffers, x, y):
        pred = torch.func.functional_call(model, (params, buffers), (x.unsqueeze(0),))
        return ((pred - y.unsqueeze(0)) ** 2).sum()

    per_sample_grads = vmap(
        grad(compute_loss), in_dims=(None, None, 0, 0),
    )(params, buffers, X, Y)

    # Compute per-sample gradient norms
    per_sample_norms = []
    for i in range(batch_size):
        total_norm_sq = sum(
            per_sample_grads[name][i].pow(2).sum()
            for name in per_sample_grads
        )
        per_sample_norms.append(total_norm_sq.sqrt())
    per_sample_norms = torch.stack(per_sample_norms)

    print(f"  Per-sample gradient norms:")
    print(f"    Min:  {per_sample_norms.min().item():.4f}")
    print(f"    Max:  {per_sample_norms.max().item():.4f}")
    print(f"    Mean: {per_sample_norms.mean().item():.4f}")

    # Clip per-sample gradients to max norm C
    C = 1.0
    clip_factors = torch.clamp(C / per_sample_norms, max=1.0)
    print(f"\n  Clipping to max norm C={C}")
    print(f"  Samples clipped: {(clip_factors < 1.0).sum().item()} / {batch_size}")

    # Apply clipping and average
    clipped_grads = {}
    for name in per_sample_grads:
        # Reshape clip_factors for broadcasting
        shape = [batch_size] + [1] * (per_sample_grads[name].dim() - 1)
        clipped = per_sample_grads[name] * clip_factors.view(shape)
        clipped_grads[name] = clipped.mean(dim=0)

    # In real DP-SGD, you'd also add calibrated Gaussian noise here
    noise_multiplier = 0.1
    for name in clipped_grads:
        noise = torch.randn_like(clipped_grads[name]) * C * noise_multiplier / batch_size
        clipped_grads[name] += noise

    # Verify clipped norms
    clipped_norms = []
    for i in range(batch_size):
        total_norm_sq = sum(
            (per_sample_grads[name][i] * clip_factors[i]).pow(2).sum()
            for name in per_sample_grads
        )
        clipped_norms.append(total_norm_sq.sqrt())
    clipped_norms = torch.stack(clipped_norms)

    print(f"\n  After clipping:")
    print(f"    Max per-sample norm: {clipped_norms.max().item():.4f} (should be <= {C})")

    print("\n  Final clipped+noised gradient shapes:")
    for name, g in clipped_grads.items():
        print(f"    {name:15s}: {list(g.shape)}")


def demo_fisher_information():
    """Compute empirical Fisher information using per-sample gradients."""
    print("\n" + "=" * 60)
    print("FISHER INFORMATION MATRIX (diagonal approximation)")
    print("=" * 60)

    model = nn.Linear(5, 3)
    batch_size = 32
    X = torch.randn(batch_size, 5)

    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {}

    def nll_loss(params, buffers, x):
        logits = torch.func.functional_call(model, (params, buffers), (x.unsqueeze(0),))
        log_probs = torch.log_softmax(logits, dim=-1)
        # Sample from the model's own distribution
        target = torch.multinomial(log_probs.exp().squeeze(0), 1)
        return -log_probs.squeeze(0).gather(0, target).squeeze()

    per_sample_grads = vmap(
        grad(nll_loss), in_dims=(None, None, 0),
    )(params, buffers, X)

    # Diagonal Fisher = E[grad^2]
    print("  Diagonal Fisher information:")
    for name, g in per_sample_grads.items():
        fisher_diag = (g ** 2).mean(dim=0)
        print(f"    {name:15s}: shape={list(fisher_diag.shape)}, "
              f"mean={fisher_diag.mean().item():.6f}")


if __name__ == "__main__":
    demo_naive_per_sample_gradients()
    demo_vmap_per_sample_gradients()
    demo_gradient_clipping()
    demo_fisher_information()
    print("\n" + "=" * 60)
    print("All per-sample gradient demos completed successfully!")
    print("=" * 60)
