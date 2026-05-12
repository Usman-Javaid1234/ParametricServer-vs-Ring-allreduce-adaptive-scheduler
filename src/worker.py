"""
worker.py — Distributed SGD Training Loop (Ring AllReduce + QSGD Compression)
CS332 | Distributed SGD Project | Phase 2

Backends available:
  ring_ar            — standard 32-bit Ring AllReduce (baseline)
  compressed_ring_ar — QSGD-quantized Ring AllReduce (N-bit, our contribution)

All PS / Ray PS code removed — this project focuses on compression in Ring AR.

Key change from Phase 1:
  --num-bits N  selects compression level (2, 4, 8, or 32)
  32 = no compression (identical to baseline ring_ar behaviour)

Run (single machine, 2 workers, 4-bit compression):
  torchrun --nproc_per_node=2 worker.py --backend compressed_ring_ar --num-bits 4

Environment variables (Docker / torchrun):
  RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT — set by torchrun automatically
  BACKEND, NUM_BITS, EPOCHS, BATCH_SIZE, LR, RUN_ID — override defaults
  CRASH_RANK, CRASH_ITER  — fault injection (CrashInjector)
  STRAGGLER_RANK, STRAGGLER_DELAY_MS — straggler injection (SlowdownInjector)
  RESULTS_DIR — where CSVs and JSON summaries are written (default: /results)
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
from fault_injector import SlowdownInjector, CrashInjector
from compressed_ring_ar import CompressedRingARBackend

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][Worker %(process)d] %(levelname)s: %(message)s"
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(model_name: str) -> nn.Module:
    if model_name == "resnet18":
        model = torchvision.models.resnet18(weights=None)
        model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc      = nn.Linear(512, 10)
        return model
    raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Dataset factory
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

        # Only rank 0 downloads; others wait at barrier
        if rank == 0:
            log.info("Rank 0: downloading/verifying CIFAR-10...")
            torchvision.datasets.CIFAR10(root=data_root, train=train,
                                          download=True, transform=transform)
            log.info("Rank 0: dataset ready.")

        if train and world_size > 1:
            dist.barrier()

        return torchvision.datasets.CIFAR10(
            root=data_root, train=train, download=False, transform=transform
        )

    raise ValueError(f"Unknown dataset: {dataset_name}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion,
                backend, metrics, device, epoch, rank,
                heartbeat, injector, crash_injector):
    model.train()
    total_loss, total_samples = 0.0, 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        global_iter = batch_idx + (epoch - 1) * len(loader)
        iter_start  = time.perf_counter()

        # ---- Fault injection ----
        crash_injector.maybe_crash(global_iter)
        injector.pre_iter_delay(global_iter)

        # ---- Forward + backward ----
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()

        # ---- Sync gradients (compressed or baseline) ----
        comm_ms = backend.sync_gradients(model)

        # ---- Optimizer step ----
        optimizer.step()

        # ---- Metrics ----
        iter_ms        = (time.perf_counter() - iter_start) * 1000.0
        total_loss    += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        throughput     = inputs.size(0) / ((time.perf_counter() - iter_start) + 1e-9)

        metrics.record_iteration(
            epoch=epoch, iteration=batch_idx,
            loss=loss.item(), throughput=throughput,
            comm_latency_ms=comm_ms, iter_time_ms=iter_ms,
        )
        heartbeat.update(iteration=global_iter, iter_time_ms=iter_ms)

        if batch_idx % 50 == 0 and rank == 0:
            bits_tag = f"[{backend.num_bits}-bit]" if hasattr(backend, "num_bits") else "[32-bit]"
            straggler_tag = " [STRAGGLER]" if injector.is_straggler() else ""
            log.info(
                f"Epoch {epoch} | Iter {batch_idx}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | Comm: {comm_ms:.1f}ms | "
                f"Throughput: {throughput:.0f} samp/s "
                f"{bits_tag}{straggler_tag}"
            )

    return total_loss / max(total_samples, 1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

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
    parser = argparse.ArgumentParser(description="Ring AllReduce + QSGD Compression")
    parser.add_argument("--backend",      type=str,   default=os.environ.get("BACKEND",    "compressed_ring_ar"),
                        choices=["ring_ar", "compressed_ring_ar"])
    parser.add_argument("--num-bits",     type=int,   default=int(os.environ.get("NUM_BITS",   "8")),
                        choices=[2, 4, 8, 32],
                        help="Quantization bit-width. 32 = no compression (baseline).")
    parser.add_argument("--model",        type=str,   default=os.environ.get("MODEL",      "resnet18"))
    parser.add_argument("--dataset",      type=str,   default=os.environ.get("DATASET",    "cifar10"))
    parser.add_argument("--epochs",       type=int,   default=int(os.environ.get("EPOCHS", "10")))
    parser.add_argument("--batch-size",   type=int,   default=int(os.environ.get("BATCH_SIZE", "128")))
    parser.add_argument("--lr",           type=float, default=float(os.environ.get("LR",   "0.01")))
    parser.add_argument("--dist-backend", type=str,   default=os.environ.get("DIST_BACKEND", "gloo"))
    parser.add_argument("--run-id",       type=str,   default=os.environ.get("RUN_ID",     "run"))
    args = parser.parse_args()

    rank       = int(os.environ.get("RANK",       0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    orch_host  = os.environ.get("ORCHESTRATOR_HOST", "localhost")

    # ---- Init distributed process group ----
    os.environ.setdefault("MASTER_ADDR", os.environ.get("MASTER_ADDR", "localhost"))
    os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "29500"))

    dist.init_process_group(
        backend=args.dist_backend,
        rank=rank,
        world_size=world_size,
    )

    # ---- Device ----
    if torch.cuda.is_available():
        n_gpus    = torch.cuda.device_count()
        local_gpu = rank % n_gpus
        device    = torch.device(f"cuda:{local_gpu}")
        torch.cuda.set_device(device)
        log.info(f"Rank {rank}/{world_size} | GPU {local_gpu} | "
                 f"{torch.cuda.get_device_name(local_gpu)}")
    else:
        device = torch.device("cpu")
        log.info(f"Rank {rank}/{world_size} | CPU")

    # ---- Heartbeat ----
    heartbeat = HeartbeatSender(worker_id=rank, orchestrator_host=orch_host)
    heartbeat.start()

    # ---- Model ----
    torch.manual_seed(42)
    model = build_model(args.model).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: {args.model} | Params: {total_params/1e6:.2f}M")

    # ---- Data ----
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

    # ---- Optimizer ----
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # ---- Backend ----
    # Both ring_ar and compressed_ring_ar use CompressedRingARBackend.
    # ring_ar is just compressed_ring_ar with num_bits=32 (no-op quantization).
    num_bits = args.num_bits if args.backend == "compressed_ring_ar" else 32
    sync_backend = CompressedRingARBackend(rank=rank, world_size=world_size,
                                           num_bits=num_bits)

    if rank == 0:
        log.info(f"Backend: {args.backend} | num_bits={num_bits} | "
                 f"world_size={world_size} | epochs={args.epochs}")

    # ---- Fault injectors ----
    injector       = SlowdownInjector(rank=rank)
    crash_injector = CrashInjector(rank=rank)

    # ---- Metrics ----
    run_id  = f"{args.run_id}_w{world_size}_b{num_bits}"
    metrics = MetricsCollector(run_id=run_id, rank=rank,
                               arch=args.backend, world_size=world_size)

    # ---- Training loop ----
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            sync_backend, metrics, device, epoch,
            rank, heartbeat, injector, crash_injector,
        )
        scheduler.step()

        # Barrier: all workers done with this epoch
        dist.barrier()

        # Rank 0 evaluates and broadcasts val metrics
        # val_tensor must be on the same device as the dist backend expects:
        # NCCL → GPU, Gloo → CPU
        val_tensor = torch.zeros(2, device=device)
        if rank == 0:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            val_tensor[0] = val_loss
            val_tensor[1] = val_acc
            log.info(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.2f}% | "
                f"bits={num_bits} | workers={world_size}"
            )

        dist.broadcast(val_tensor, src=0)
        val_loss = val_tensor[0].item()
        val_acc  = val_tensor[1].item()

        metrics.record_epoch(epoch, train_loss, val_loss, val_acc)

        # Checkpoint every 5 epochs (rank 0 only)
        if rank == 0 and epoch % 5 == 0:
            os.makedirs("/results", exist_ok=True)
            ckpt = f"/results/{run_id}_ep{epoch}.pt"
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "val_acc":    val_acc,
                "num_bits":   num_bits,
                "world_size": world_size,
            }, ckpt)
            log.info(f"Checkpoint saved → {ckpt}")

        # Barrier: all ranks proceed to next epoch together
        dist.barrier()

    # ---- Log compression info (rank 0) ----
    if rank == 0:
        info = sync_backend.get_compression_info()
        log.info(f"Compression summary: {info}")

    metrics.finalize()
    heartbeat.stop()
    dist.destroy_process_group()
    log.info(f"Rank {rank}: training complete.")


if __name__ == "__main__":
    main()
