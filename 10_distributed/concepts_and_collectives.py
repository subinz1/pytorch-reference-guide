"""
Distributed Concepts and Collective Operations (CPU/Gloo)
=========================================================

Demonstrates fundamental distributed concepts and collective operations
using the Gloo backend on CPU. No GPU required.

Run with:
    torchrun --nproc_per_node=3 concepts_and_collectives.py
"""

import os

import torch
import torch.distributed as dist


def print_rank(msg: str) -> None:
    rank = dist.get_rank()
    print(f"  [Rank {rank}] {msg}", flush=True)


def print_header(title: str) -> None:
    if dist.get_rank() == 0:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    dist.barrier()


def demo_environment_variables():
    """Show the environment variables set by torchrun."""
    print_header("Environment Variables (set by torchrun)")
    env_vars = ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
    info = ", ".join(f"{v}={os.environ.get(v, 'N/A')}" for v in env_vars)
    print_rank(info)
    dist.barrier()


def demo_broadcast():
    """Broadcast: one rank sends data to all others."""
    print_header("Broadcast (src=0)")

    rank = dist.get_rank()
    tensor = torch.tensor([42.0, 7.0]) if rank == 0 else torch.zeros(2)
    print_rank(f"Before: {tensor.tolist()}")

    dist.broadcast(tensor, src=0)
    print_rank(f"After:  {tensor.tolist()}")
    dist.barrier()


def demo_all_reduce():
    """All-reduce: sum tensors across all ranks, result goes to everyone."""
    print_header("All-Reduce (SUM)")

    rank = dist.get_rank()
    tensor = torch.tensor([float(rank + 1), float((rank + 1) * 10)])
    print_rank(f"Before: {tensor.tolist()}")

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print_rank(f"After:  {tensor.tolist()}")
    dist.barrier()


def demo_reduce():
    """Reduce: sum tensors, but result only goes to destination rank."""
    print_header("Reduce (SUM, dst=0)")

    rank = dist.get_rank()
    tensor = torch.tensor([float(rank + 1)])
    print_rank(f"Before: {tensor.tolist()}")

    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    print_rank(f"After:  {tensor.tolist()} {'(has sum)' if rank == 0 else '(unchanged)'}")
    dist.barrier()


def demo_all_gather():
    """All-gather: each rank contributes, everyone gets the full collection."""
    print_header("All-Gather")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local = torch.tensor([float(rank * 10 + 1), float(rank * 10 + 2)])
    print_rank(f"Local tensor: {local.tolist()}")

    gathered = [torch.zeros(2) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    print_rank(f"Gathered: {[t.tolist() for t in gathered]}")
    dist.barrier()


def demo_gather():
    """Gather: all ranks send to one destination."""
    print_header("Gather (dst=0)")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local = torch.tensor([float(rank * 100)])
    print_rank(f"Local: {local.tolist()}")

    if rank == 0:
        gather_list = [torch.zeros(1) for _ in range(world_size)]
        dist.gather(local, gather_list=gather_list, dst=0)
        print_rank(f"Gathered: {[t.tolist() for t in gather_list]}")
    else:
        dist.gather(local, dst=0)
        print_rank("(sent to rank 0)")
    dist.barrier()


def demo_scatter():
    """Scatter: one rank distributes different data to each rank."""
    print_header("Scatter (src=0)")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    output = torch.zeros(2)

    if rank == 0:
        scatter_list = [torch.tensor([float(i), float(i * 10)]) for i in range(world_size)]
        print_rank(f"Scattering: {[t.tolist() for t in scatter_list]}")
        dist.scatter(output, scatter_list=scatter_list, src=0)
    else:
        dist.scatter(output, src=0)

    print_rank(f"Received: {output.tolist()}")
    dist.barrier()


def demo_reduce_scatter():
    """Reduce-scatter: reduce then scatter chunks to each rank."""
    print_header("Reduce-Scatter")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    input_tensor = torch.arange(world_size, dtype=torch.float32) + rank * 10
    print_rank(f"Input: {input_tensor.tolist()}")

    output = torch.zeros(1)
    dist.reduce_scatter_tensor(output, input_tensor, op=dist.ReduceOp.SUM)
    print_rank(f"Output: {output.tolist()}")
    dist.barrier()


def demo_barrier():
    """Barrier: synchronize all ranks."""
    print_header("Barrier (synchronization)")

    import time

    rank = dist.get_rank()
    sleep_time = rank * 0.2
    time.sleep(sleep_time)
    print_rank(f"Arrived at barrier after {sleep_time:.1f}s delay")
    dist.barrier()
    print_rank("Passed barrier (all ranks synchronized)")
    dist.barrier()


def demo_process_groups():
    """Show creating and using sub-groups."""
    print_header("Process Groups (sub-groups)")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    even_ranks = [r for r in range(world_size) if r % 2 == 0]
    odd_ranks = [r for r in range(world_size) if r % 2 != 0]

    even_group = dist.new_group(even_ranks)
    odd_group = dist.new_group(odd_ranks)

    tensor = torch.tensor([float(rank)])

    if rank % 2 == 0:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=even_group)
        print_rank(f"Even group all-reduce result: {tensor.tolist()} (sum of ranks {even_ranks})")
    else:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=odd_group)
        print_rank(f"Odd group all-reduce result: {tensor.tolist()} (sum of ranks {odd_ranks})")

    dist.barrier()


def main():
    dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"\nDistributed setup: {world_size} processes, backend=gloo")
        print("Each demo shows the state before and after the collective.\n")

    demo_environment_variables()
    demo_broadcast()
    demo_all_reduce()
    demo_reduce()
    demo_all_gather()
    demo_gather()
    demo_scatter()
    demo_reduce_scatter()
    demo_barrier()
    demo_process_groups()

    if rank == 0:
        print(f"\n{'='*60}")
        print("  All collective operations demonstrated successfully!")
        print(f"{'='*60}\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
