"""
worker.py — Shared training loop with pluggable PS / Ring AR backend.
CS332 | Distributed SGD Project | Phase 1

Fixes applied:
  1. num_workers=0 in DataLoader  → avoids Docker /dev/shm bus error
  2. Only rank-0 downloads CIFAR-10, barrier ensures others wait
  3. sys.path fix so imports work inside /app/src/
  4. HeartbeatSender wired into training loop
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from metrics import MetricsCollector
from orchestrator import HeartbeatSender

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][Worker %(process)d] %(levelname)s: %(message)s"
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(model_name: str) -> nn.Module:
    if model_name == "resnet50":
        model = torchvision.models.resnet50(weights=None)
        model.fc = nn.Linear(2048, 10)
        return model
    raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Dataset factory
# Only rank 0 downloads; barrier ensures all others wait before loading.
# ---------------------------------------------------------------------------

def build_dataset(dataset_name: str, train: bool, rank: int, world_size: int):
    data_root = "/data/cifar10"

    if dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])

        # Rank 0 downloads; barrier keeps others from starting until done
        if rank == 0:
            log.info("Rank 0: downloading/verifying CIFAR-10...")
            torchvision.datasets.CIFAR10(root=data_root, train=train,
                                          download=True, transform=transform)
            log.info("Rank 0: dataset ready.")

        if world_size > 1:
            dist.barrier()

        return torchvision.datasets.CIFAR10(
            root=data_root, train=train, download=False, transform=transform
        )

    raise ValueError(f"Unknown dataset: {dataset_name}")


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------

class PSBackend:
    def __init__(self, rank: int, world_size: int, tau: int = 0):
        self.rank       = rank
        self.world_size = world_size
        self.tau        = tau
        self.iteration  = 0

    def sync_gradients(self, model: nn.Module) -> float:
        t0 = time.perf_counter()
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size
        if self.tau == 0:
            dist.barrier()
        self.iteration += 1
        return (time.perf_counter() - t0) * 1000.0


class RingARBackend:
    def __init__(self, rank: int, world_size: int):
        self.rank       = rank
        self.world_size = world_size

    def sync_gradients(self, model: nn.Module) -> float:
        t0 = time.perf_counter()
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size
        return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model, loader, optimizer, criterion,
    backend, metrics, device, epoch, rank,
    heartbeat,
):
    model.train()
    total_loss, total_samples = 0.0, 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        iter_start = time.perf_counter()

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        comm_latency_ms = backend.sync_gradients(model)
        optimizer.step()

        iter_time_ms   = (time.perf_counter() - iter_start) * 1000.0
        total_loss    += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        throughput     = inputs.size(0) / ((time.perf_counter() - iter_start) + 1e-9)

        metrics.record_iteration(
            epoch=epoch,
            iteration=batch_idx,
            loss=loss.item(),
            throughput=throughput,
            comm_latency_ms=comm_latency_ms,
            iter_time_ms=iter_time_ms,
        )

        heartbeat.update(
            iteration=batch_idx + (epoch - 1) * len(loader),
            iter_time_ms=iter_time_ms,
        )

        if batch_idx % 20 == 0 and rank == 0:
            log.info(
                f"Epoch {epoch} | Iter {batch_idx}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | Comm: {comm_latency_ms:.1f}ms | "
                f"Throughput: {throughput:.0f} samp/s"
            )

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total   += targets.size(0)
    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend",      type=str,   default="ring_ar",
                        choices=["ps", "ring_ar"])
    parser.add_argument("--model",        type=str,   default="resnet50")
    parser.add_argument("--dataset",      type=str,   default="cifar10")
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=0.01)
    parser.add_argument("--tau",          type=int,   default=0)
    parser.add_argument("--dist-backend", type=str,   default="gloo",
                        choices=["gloo", "nccl"])
    parser.add_argument("--run-id",       type=str,   default="baseline")
    args = parser.parse_args()

    rank        = int(os.environ.get("RANK", 0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")
    orch_host   = os.environ.get("ORCHESTRATOR_HOST", "orchestrator")

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    dist.init_process_group(
        backend=args.dist_backend,
        rank=rank,
        world_size=world_size,
    )
    log.info(f"Rank {rank}/{world_size} init | backend={args.dist_backend}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Heartbeat sender (background thread) ----
    heartbeat = HeartbeatSender(worker_id=rank, orchestrator_host=orch_host)
    heartbeat.start()

    # ---- Model ----
    torch.manual_seed(42)
    model = build_model(args.model).to(device)

    # ---- Data ----
    train_dataset = build_dataset(args.dataset, train=True,
                                  rank=rank, world_size=world_size)
    val_dataset   = build_dataset(args.dataset, train=False,
                                  rank=rank, world_size=world_size)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                       rank=rank, shuffle=True, seed=42)

    # num_workers=0 avoids Docker /dev/shm shared-memory bus error
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=0,
                              pin_memory=False)
    val_loader   = DataLoader(val_dataset, batch_size=256, shuffle=False,
                              num_workers=0)

    # ---- Optimizer / Loss ----
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # ---- Sync backend ----
    if args.backend == "ps":
        sync_backend = PSBackend(rank, world_size, tau=args.tau)
    else:
        sync_backend = RingARBackend(rank, world_size)

    # ---- Metrics ----
    metrics = MetricsCollector(
        run_id=args.run_id,
        rank=rank,
        arch=args.backend,
        world_size=world_size,
    )

    # ---- Training ----
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            sync_backend, metrics, device, epoch, rank, heartbeat,
        )
        scheduler.step()

        if rank == 0:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            log.info(f"Val | Epoch {epoch} | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
            metrics.record_epoch(epoch, train_loss, val_loss, val_acc)

            if epoch % 5 == 0:
                os.makedirs("/results", exist_ok=True)
                ckpt = f"/results/{args.run_id}_ep{epoch}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_acc": val_acc,
                }, ckpt)
                log.info(f"Checkpoint: {ckpt}")

    metrics.finalize()
    heartbeat.stop()
    dist.destroy_process_group()
    log.info("Training complete.")


if __name__ == "__main__":
    main()