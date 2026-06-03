"""
DeviceMesh Creation and Usage
==============================

Demonstrates DeviceMesh: the foundation for modern distributed training.
Shows 1D, 2D, and 3D mesh creation, sub-mesh access, and how meshes
map to process groups.

Run with:
    torchrun --nproc_per_node=4 device_mesh_example.py

For reference viewing without GPUs, the script prints patterns when
CUDA is not available.
"""

import os

import torch
import torch.distributed as dist


def check_environment() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() >= 4


def demo_1d_mesh():
    """1D mesh: all devices in one dimension (simple data parallelism)."""
    from torch.distributed.device_mesh import init_device_mesh

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))

    if rank == 0:
        print("=" * 60)
        print("  1D Mesh: Simple Data Parallelism")
        print("=" * 60)
        print(f"  Mesh shape: ({world_size},)")
        print(f"  Dimension names: ('dp',)")
        print(f"  All {world_size} GPUs in one DP group")
        print()

    dp_mesh = mesh["dp"]
    print(f"  [Rank {rank}] DP group rank: {dp_mesh.get_local_rank()}", flush=True)
    dist.barrier()

    if rank == 0:
        print()


def demo_2d_mesh():
    """2D mesh: DP x TP — data parallelism combined with tensor parallelism."""
    from torch.distributed.device_mesh import init_device_mesh

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dp_size = 2
    tp_size = world_size // dp_size

    mesh = init_device_mesh(
        "cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
    )

    if rank == 0:
        print("=" * 60)
        print("  2D Mesh: DP x TP")
        print("=" * 60)
        print(f"  Mesh shape: ({dp_size}, {tp_size})")
        print(f"  Dimension names: ('dp', 'tp')")
        print()
        print("  Layout:")
        print(f"           TP=0   TP=1")
        for dp in range(dp_size):
            gpus = [dp * tp_size + tp for tp in range(tp_size)]
            print(f"    DP={dp}:  GPU{gpus[0]}   GPU{gpus[1]}")
        print()

    dp_mesh = mesh["dp"]
    tp_mesh = mesh["tp"]

    print(
        f"  [Rank {rank}] DP local_rank={dp_mesh.get_local_rank()}, "
        f"TP local_rank={tp_mesh.get_local_rank()}",
        flush=True,
    )
    dist.barrier()

    if rank == 0:
        print()
        print("  DP groups (ranks that share the same TP position):")
        for tp in range(tp_size):
            dp_ranks = [dp * tp_size + tp for dp in range(dp_size)]
            print(f"    TP={tp}: ranks {dp_ranks}")
        print()
        print("  TP groups (ranks that share the same DP position):")
        for dp in range(dp_size):
            tp_ranks = [dp * tp_size + tp for tp in range(tp_size)]
            print(f"    DP={dp}: ranks {tp_ranks}")
        print()


def demo_mesh_collectives():
    """Show how collectives work on specific mesh dimensions."""
    from torch.distributed.device_mesh import init_device_mesh

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    mesh = init_device_mesh(
        "cuda", (2, world_size // 2), mesh_dim_names=("dp", "tp")
    )

    if rank == 0:
        print("=" * 60)
        print("  Collectives on Mesh Dimensions")
        print("=" * 60)

    # All-reduce within the TP group
    tensor = torch.tensor([float(rank)], device="cuda")
    tp_group = mesh["tp"].get_group()
    dist.all_reduce(tensor, group=tp_group)
    print(
        f"  [Rank {rank}] TP all-reduce result: {tensor.item():.0f} "
        f"(sum of ranks in my TP group)",
        flush=True,
    )
    dist.barrier()

    # All-reduce within the DP group
    tensor = torch.tensor([float(rank)], device="cuda")
    dp_group = mesh["dp"].get_group()
    dist.all_reduce(tensor, group=dp_group)
    print(
        f"  [Rank {rank}] DP all-reduce result: {tensor.item():.0f} "
        f"(sum of ranks in my DP group)",
        flush=True,
    )
    dist.barrier()
    if rank == 0:
        print()


def print_reference_patterns():
    """Print DeviceMesh patterns when GPUs are not available."""
    print("=" * 70)
    print("  DeviceMesh Reference Patterns")
    print("  Requires 4+ GPUs to run live. Showing code patterns.")
    print("=" * 70)

    print("""
1D MESH (Data Parallelism)
==========================

    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cuda", (8,), mesh_dim_names=("dp",))
    # 8 GPUs all in one data-parallel group


2D MESH (DP x TP)
==================

    mesh = init_device_mesh("cuda", (4, 2), mesh_dim_names=("dp", "tp"))
    # 8 GPUs: 4 DP groups of 2, 2 TP groups of 4
    # Layout:
    #        TP=0  TP=1
    # DP=0:  GPU0  GPU1
    # DP=1:  GPU2  GPU3
    # DP=2:  GPU4  GPU5
    # DP=3:  GPU6  GPU7

    dp_mesh = mesh["dp"]  # Sub-mesh for data parallelism
    tp_mesh = mesh["tp"]  # Sub-mesh for tensor parallelism

    # Use sub-meshes with FSDP and TP:
    # parallelize_module(model, tp_mesh, plan)  # TP on tp_mesh
    # fully_shard(model, mesh=dp_mesh)          # FSDP on dp_mesh


3D MESH (DP x TP x PP)
========================

    mesh = init_device_mesh(
        "cuda", (2, 4, 2), mesh_dim_names=("dp", "tp", "pp")
    )
    # 16 GPUs: 2 DP x 4 TP x 2 PP
    dp_mesh = mesh["dp"]
    tp_mesh = mesh["tp"]
    pp_mesh = mesh["pp"]


ACCESSING PROCESS GROUPS
=========================

    # Get the underlying process group for a sub-mesh
    tp_group = mesh["tp"].get_group()
    dist.all_reduce(tensor, group=tp_group)  # All-reduce within TP group only

    # Get this rank's position in a dimension
    tp_rank = mesh["tp"].get_local_rank()
""")


def main():
    if not check_environment():
        print_reference_patterns()
        return

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    rank = dist.get_rank()
    if rank == 0:
        print(f"\nDeviceMesh Demo: {dist.get_world_size()} GPUs\n")

    demo_1d_mesh()
    demo_2d_mesh()
    demo_mesh_collectives()

    if rank == 0:
        print("DeviceMesh demos complete.\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
