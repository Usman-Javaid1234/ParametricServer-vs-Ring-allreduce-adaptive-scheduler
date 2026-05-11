"""
worker.py — Shared training loop with pluggable PS / Ring AR backend.
CS332 | Distributed SGD Project | Phase 1

Memory fixes:
  - ResNet-18 instead of ResNet-50 (4x fewer params, fits in Docker RAM)
  - Native CIFAR 32x32 (no Resize to 224 — that blew up memory)
  - num_workers=0 (no /dev/shm multiprocessing)
  - batch_size default lowered to 32
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
from fault_injector import SlowdownInjector

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][Worker %(process)d] %(levelname)s: %(message)s"
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factory  (ResNet-18 for CPU/low-RAM Docker environments)
# ---------------------------------------------------------------------------

def build_model(model_name: str) -> nn.Module:
    if model_name == "resnet18":
        # ResNet-18 adapted for CIFAR 32x32:
        # replace 7x7 conv with 3x3, remove maxpool
        model = torchvision.models.resnet18(weights=None)
        model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc      = nn.Linear(512, 10)
        return model
    if model_name == "resnet50":
        model = torchvision.models.resnet50(weights=None)
        model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc      = nn.Linear(2048, 10)
        return model
    raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Dataset factory  (native 32x32 — no resize)
# ---------------------------------------------------------------------------

def build_dataset(dataset_name: str, train: bool, rank: int, world_size: int):
    data_root = "/data/cifar10"

    if dataset_name == "cifar10":
        if train:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2470, 0.2435, 0.2616)),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2470, 0.2435, 0.2616)),
            ])

        if rank == 0:
            log.info(f"Rank 0: downloading/verifying CIFAR-10 (train={train})...")
            torchvision.datasets.CIFAR10(root=data_root, train=train,
                                          download=True, transform=transform)
            log.info("Rank 0: dataset ready.")

        # Only barrier for train dataset — all ranks call this.
        # Val dataset is only loaded by rank 0 so no barrier needed.
        if train and world_size > 1:
            dist.barrier()

        return torchvision.datasets.CIFAR10(
            root=data_root, train=train, download=False, transform=transform
        )

    raise ValueError(f"Unknown dataset: {dataset_name}")


# ---------------------------------------------------------------------------
# Sync backends
# ---------------------------------------------------------------------------

class PSBackend:
    """
    Parameter Server backend with simulated O(n) server bottleneck.

    True PS server bandwidth per iteration = 2n × |M|  (n inbound + n outbound).
    Ring AR bandwidth per GPU              = 2(n-1)/n × |M|  (bandwidth-optimal).

    Ratio = PS / Ring_AR = 2n / (2(n-1)/n) = n²/(n-1)
    At n=4: ratio = 16/3 = 5.33×

    Rather than computing exact bytes (which would make runs impractically slow),
    we use a configurable per-worker overhead added to each PS sync call.
    This overhead scales linearly with world_size, matching the O(n) server cost.

    Environment variables:
      PS_SERVER_OVERHEAD_MS  — extra ms added per worker (default: 50)
                               Total overhead = PS_SERVER_OVERHEAD_MS × world_size
                               At n=4: 200ms extra per iteration
                               At n=8: 400ms extra per iteration

    SSP mode (tau > 0):
      Fast workers only pay the server overhead when their staleness hits tau.
      Otherwise they proceed immediately, simulating the SSP bypass.
    """

    def __init__(self, rank: int, world_size: int, tau: int = 0):
        self.rank          = rank
        self.world_size    = world_size
        self.tau           = tau
        self.iteration     = 0
        self._fast_iters   = 0   # iterations SSP worker skipped the overhead

        # Simulated server overhead: scales linearly with n (O(n) bottleneck)
        overhead_per_worker = float(os.environ.get("PS_SERVER_OVERHEAD_MS", "50"))
        self.server_overhead_s = (overhead_per_worker * world_size) / 1000.0

        mode = "BSP" if tau == 0 else f"SSP(τ={tau})"
        log.info(
            f"PSBackend init | mode={mode} | world_size={world_size} | "
            f"server_overhead={self.server_overhead_s*1000:.0f}ms per iter "
            f"({overhead_per_worker}ms × {world_size} workers)"
        )

    def sync_gradients(self, model: nn.Module) -> float:
        t0 = time.perf_counter()

        # ---- Gradient aggregation (all_reduce) ----
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                p.grad.data /= self.world_size

        # ---- Simulated server bottleneck delay ----
        # BSP: every worker pays full overhead every iteration.
        # SSP: fast workers skip overhead until staleness bound τ is reached,
        #      then block (simulating the server's bounded-staleness gate).
        if self.tau == 0:
            # BSP — strict barrier + full server overhead
            time.sleep(self.server_overhead_s)
            dist.barrier()
        else:
            # SSP — only pay overhead every τ iterations
            if self.iteration % max(self.tau, 1) == 0:
                time.sleep(self.server_overhead_s)
                self._fast_iters = 0
            else:
                # Fast worker skips this round's server wait
                self._fast_iters += 1

        self.iteration += 1
        return (time.perf_counter() - t0) * 1000.0

    def get_staleness_info(self) -> dict:
        return {
            "iteration":   self.iteration,
            "fast_iters":  self._fast_iters,
            "overhead_ms": self.server_overhead_s * 1000,
        }


class RingARBackend:
    def __init__(self, rank: int, world_size: int):
        self.rank       = rank
        self.world_size = world_size

    def sync_gradients(self, model: nn.Module) -> float:
        t0 = time.perf_counter()
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                p.grad.data /= self.world_size
        return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion,
                backend, metrics, device, epoch, rank, heartbeat, injector):
    model.train()
    total_loss, total_samples = 0.0, 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        iter_start = time.perf_counter()

        # Straggler delay BEFORE forward pass — forces all other workers
        # to wait at the all_reduce barrier, making effect visible in
        # comm_latency_ms for ALL ranks not just rank 3's iter_time_ms
        injector.pre_iter_delay(batch_idx + (epoch - 1) * len(loader))

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()

        comm_ms = backend.sync_gradients(model)
        optimizer.step()

        iter_ms        = (time.perf_counter() - iter_start) * 1000.0
        total_loss    += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        throughput     = inputs.size(0) / ((time.perf_counter() - iter_start) + 1e-9)

        metrics.record_iteration(
            epoch=epoch, iteration=batch_idx,
            loss=loss.item(), throughput=throughput,
            comm_latency_ms=comm_ms, iter_time_ms=iter_ms,
        )
        heartbeat.update(
            iteration=batch_idx + (epoch - 1) * len(loader),
            iter_time_ms=iter_ms,
        )

        if batch_idx % 50 == 0 and rank == 0:
            straggler_tag = " [STRAGGLER]" if injector.is_straggler() else ""
            log.info(
                f"Epoch {epoch} | Iter {batch_idx}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | Comm: {comm_ms:.1f}ms | "
                f"Throughput: {throughput:.0f} samp/s{straggler_tag}"
            )

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        out  = model(inputs)
        loss = criterion(out, targets)
        total_loss += loss.item() * inputs.size(0)
        correct    += out.argmax(1).eq(targets).sum().item()
        total      += targets.size(0)
    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend",      type=str,   default=os.environ.get("BACKEND",      "ring_ar"),
                        choices=["ps", "ring_ar"])
    parser.add_argument("--model",        type=str,   default=os.environ.get("MODEL",        "resnet18"))
    parser.add_argument("--dataset",      type=str,   default=os.environ.get("DATASET",      "cifar10"))
    parser.add_argument("--epochs",       type=int,   default=int(os.environ.get("EPOCHS",   "5")))
    parser.add_argument("--batch-size",   type=int,   default=int(os.environ.get("BATCH_SIZE","128")))
    parser.add_argument("--lr",           type=float, default=float(os.environ.get("LR",     "0.01")))
    parser.add_argument("--tau",          type=int,   default=int(os.environ.get("TAU",      "0")))
    parser.add_argument("--dist-backend", type=str,   default=os.environ.get("DIST_BACKEND", "gloo"))
    parser.add_argument("--run-id",       type=str,   default=os.environ.get("RUN_ID",       "baseline"))
    args = parser.parse_args()

    rank        = int(os.environ.get("RANK", 0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    orch_host   = os.environ.get("ORCHESTRATOR_HOST", "orchestrator")

    os.environ.setdefault("MASTER_ADDR", os.environ.get("MASTER_ADDR", "localhost"))
    os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "29500"))

    dist.init_process_group(backend=args.dist_backend, rank=rank,
                            world_size=world_size)

    # GPU setup:
    # Multiple GPUs → assign one per rank (rank % n_gpus).
    # Single GPU   → all workers share cuda:0 (still fast compute, Gloo sync).
    # No GPU       → fall back to CPU.
    # Note: we keep Gloo backend even with GPU — NCCL requires one exclusive
    # GPU per process which isn't possible when sharing a single card.
    if torch.cuda.is_available():
        n_gpus    = torch.cuda.device_count()
        local_gpu = rank % n_gpus
        device    = torch.device(f"cuda:{local_gpu}")
        torch.cuda.set_device(device)
        log.info(f"Rank {rank}/{world_size} | GPU {local_gpu}/{n_gpus} | "
                 f"{torch.cuda.get_device_name(local_gpu)}")
    else:
        device = torch.device("cpu")
        log.info(f"Rank {rank}/{world_size} | No GPU — using CPU")

    # Heartbeat
    heartbeat = HeartbeatSender(worker_id=rank, orchestrator_host=orch_host)
    heartbeat.start()

    # Model
    torch.manual_seed(42)
    model = build_model(args.model).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model params: {total_params/1e6:.2f}M")

    # Data — all ranks load train, all ranks load val (GPU has enough RAM)
    train_ds = build_dataset(args.dataset, train=True,
                             rank=rank, world_size=world_size)
    val_ds   = build_dataset(args.dataset, train=False,
                             rank=rank, world_size=world_size)

    sampler      = DistributedSampler(train_ds, num_replicas=world_size,
                                      rank=rank, shuffle=True, seed=42)
    use_gpu      = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0,
                              pin_memory=use_gpu)
    val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False,
                              num_workers=0, pin_memory=use_gpu)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # Backend
    sync_backend = (PSBackend(rank, world_size, tau=args.tau)
                    if args.backend == "ps"
                    else RingARBackend(rank, world_size))

    # Metrics
    metrics = MetricsCollector(run_id=args.run_id, rank=rank,
                               arch=args.backend, world_size=world_size)

    # Fault injector — reads STRAGGLER_RANK / STRAGGLER_DELAY_MS from env
    injector = SlowdownInjector(rank=rank)

    # Training loop — per-epoch validation on rank 0.
    # Barrier pair ensures rank 0 validates safely between epochs.
    # ONE row written per epoch per rank — no duplicates.
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                 sync_backend, metrics, device, epoch,
                                 rank, heartbeat, injector)
        scheduler.step()

        # Barrier 1: all workers done training this epoch
        dist.barrier()

        # Rank 0 evaluates; broadcasts val_loss and val_acc to all ranks
        # so every rank writes ONE complete row with val metrics.
        val_tensor = torch.zeros(2)   # [val_loss, val_acc]
        if rank == 0:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            val_tensor[0] = val_loss
            val_tensor[1] = val_acc
            log.info(f"Epoch {epoch}/{args.epochs} | "
                     f"train_loss={train_loss:.4f} | "
                     f"val_loss={val_loss:.4f} | val_acc={val_acc:.2f}%")
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Broadcast val metrics from rank 0 to all other ranks
        dist.broadcast(val_tensor, src=0)
        val_loss = val_tensor[0].item()
        val_acc  = val_tensor[1].item()

        # ALL ranks write one complete row with val_loss + val_acc
        metrics.record_epoch(epoch, train_loss, val_loss, val_acc)

        # Checkpoint every 5 epochs (rank 0 only)
        if rank == 0 and epoch % 5 == 0:
            os.makedirs("/results", exist_ok=True)
            ckpt = f"/results/{args.run_id}_ep{epoch}.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_acc": val_acc}, ckpt)
            log.info(f"Checkpoint: {ckpt}")

        # Barrier 2: all ranks done writing, proceed to next epoch
        dist.barrier()

    metrics.finalize()
    heartbeat.stop()
    dist.destroy_process_group()
    log.info("Training complete.")


if __name__ == "__main__":
    main()