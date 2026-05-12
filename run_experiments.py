#!/usr/bin/env python3
"""
run_experiments.py — Automated experiment grid runner
CS332 | Distributed SGD + QSGD Compression

Runs all combinations of world_size × num_bits sequentially,
then runs the failure scenario experiment.

Experiment grid:
  world_size : 2, 4
  num_bits   : 32, 8, 4, 2
  = 8 core runs × 20 epochs each

Failure scenario:
  world_size=4, num_bits=4, crash_rank=1 at iter 200
  + clean baseline for comparison

Usage:
  python run_experiments.py                    # full grid
  python run_experiments.py --workers 2        # only 2-worker runs
  python run_experiments.py --bits 8 4         # only 8-bit and 4-bit
  python run_experiments.py --skip-failure     # skip failure scenario
  python run_experiments.py --epochs 5         # override epochs (e.g. for testing)
  python run_experiments.py --dry-run          # print commands without running
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime, timedelta


# ── ANSI colours ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def log(msg, colour=RESET):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{colour}{BOLD}[{ts}]{RESET} {colour}{msg}{RESET}", flush=True)


# ── Docker compose command builder ──────────────────────────────────────────

def compose_cmd(
    world_size: int,
    num_bits: int,
    epochs: int,
    run_id: str,
    crash_rank: int = None,
    crash_iter: int = 200,
    batch_size: int = 128,
    lr: float = 0.01,
) -> tuple[list[str], dict]:
    """Build docker compose command and environment for one run."""

    services = ["orchestrator", "worker_0", "worker_1"]
    if world_size >= 4:
        services += ["worker_2", "worker_3"]

    env = {
        **os.environ,
        "WORLD_SIZE":   str(world_size),
        "NUM_BITS":     str(num_bits),
        "EPOCHS":       str(epochs),
        "BATCH_SIZE":   str(batch_size),
        "LR":           str(lr),
        "RUN_ID":       run_id,
        "CRASH_RANK":   str(crash_rank) if crash_rank is not None else "",
        "CRASH_ITER":   str(crash_iter),
    }

    # --exit-code-from worker_0: compose exit code tracks workers, not orchestrator.
    # This prevents orchestrator exit 137 (OOM/killed) from blocking the next run.
    cmd = [
        "docker", "compose", "up",
        "--abort-on-container-exit",
        "--exit-code-from", "worker_0",
    ] + services

    return cmd, env


def compose_down() -> None:
    try:
        subprocess.run(
            ["docker", "compose", "down", "--remove-orphans", "--timeout", "15"],
            capture_output=True,
            timeout=45,
        )
    except subprocess.TimeoutExpired:
        # Force kill if graceful shutdown times out
        subprocess.run(
            ["docker", "compose", "kill"],
            capture_output=True,
        )


# ── Single run ───────────────────────────────────────────────────────────────

def run_one(
    world_size: int,
    num_bits: int,
    epochs: int,
    run_id: str,
    dry_run: bool = False,
    crash_rank: int = None,
    crash_iter: int = 200,
    batch_size: int = 128,
) -> bool:
    """
    Execute one docker compose run. Returns True on success, False on failure.
    """
    tag = f"workers={world_size} bits={num_bits}"
    if crash_rank is not None:
        tag += f" CRASH_RANK={crash_rank}@iter{crash_iter}"

    log(f"Starting: {tag}", CYAN)

    cmd, env = compose_cmd(
        world_size=world_size,
        num_bits=num_bits,
        epochs=epochs,
        run_id=run_id,
        crash_rank=crash_rank,
        crash_iter=crash_iter,
        batch_size=batch_size,
    )

    if dry_run:
        log(f"[DRY RUN] {' '.join(cmd)}", YELLOW)
        for k in ["WORLD_SIZE", "NUM_BITS", "EPOCHS", "RUN_ID", "CRASH_RANK"]:
            log(f"  {k}={env.get(k, '')}", YELLOW)
        return True

    t0 = time.time()
    try:
        result = subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL)
        elapsed = time.time() - t0
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        # returncode 0 = worker_0 exited cleanly (success)
        # returncode 1 = worker crashed (failure)
        # orchestrator exits 137 (killed by compose) — ignored via --exit-code-from
        if result.returncode == 0:
            log(f"✓ Done: {tag} | time={elapsed_str}", GREEN)
            return True
        else:
            log(f"✗ Failed: {tag} | returncode={result.returncode}", RED)
            return False

    except KeyboardInterrupt:
        log("Interrupted by user — stopping current run.", YELLOW)
        compose_down()
        raise

    finally:
        # Force stop all containers before next run, with short timeout
        compose_down()
        time.sleep(5)   # let ports (29500, 29600) fully free up


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run experiment grid")
    parser.add_argument("--workers",       type=int,   nargs="+", default=[2, 4],
                        help="Worker counts to test (default: 2 4)")
    parser.add_argument("--bits",          type=int,   nargs="+", default=[32, 8, 4, 2],
                        help="Bit widths to test (default: 32 8 4 2)")
    parser.add_argument("--epochs",        type=int,   default=20,
                        help="Epochs per run (default: 20)")
    parser.add_argument("--batch-size",    type=int,   default=128)
    parser.add_argument("--run-id",        type=str,   default="exp",
                        help="Base run ID prefix (default: exp)")
    parser.add_argument("--skip-failure",  action="store_true",
                        help="Skip failure scenario experiments")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    total_runs = len(args.workers) * len(args.bits)
    if not args.skip_failure:
        total_runs += 2   # failure + clean baseline

    log(f"Experiment grid: {len(args.workers)} worker configs × "
        f"{len(args.bits)} bit widths = {len(args.workers)*len(args.bits)} runs", BOLD)
    log(f"Epochs per run : {args.epochs}", BOLD)
    log(f"Total runs     : {total_runs}", BOLD)
    log(f"Est. time      : ~{total_runs * args.epochs * 3 // 60}–"
        f"{total_runs * args.epochs * 5 // 60} min total", BOLD)
    print()

    results = []
    run_number = 0

    # ── Core experiment grid ─────────────────────────────────────────────────
    for world_size in args.workers:
        for num_bits in args.bits:
            run_number += 1
            run_id = args.run_id

            log(f"Run {run_number}/{total_runs}: "
                f"workers={world_size} bits={num_bits}", BOLD)

            success = run_one(
                world_size=world_size,
                num_bits=num_bits,
                epochs=args.epochs,
                run_id=run_id,
                dry_run=args.dry_run,
                batch_size=args.batch_size,
            )
            results.append({
                "workers":  world_size,
                "bits":     num_bits,
                "run_id":   f"{run_id}_w{world_size}_b{num_bits}",
                "success":  success,
                "type":     "grid",
            })

            if not success:
                log(f"Run failed — continuing with next configuration.", YELLOW)

    # ── Failure scenario ─────────────────────────────────────────────────────
    if not args.skip_failure:
        print()
        log("─── Failure Scenario Experiments ───", CYAN)

        # Clean baseline (no crash)
        run_number += 1
        log(f"Run {run_number}/{total_runs}: Failure baseline (no crash, workers=4, bits=4)", BOLD)
        success = run_one(
            world_size=4,
            num_bits=4,
            epochs=args.epochs,
            run_id="failure_clean",
            dry_run=args.dry_run,
            batch_size=args.batch_size,
        )
        results.append({
            "workers": 4, "bits": 4,
            "run_id":  "failure_clean_w4_b4",
            "success": success,
            "type":    "failure_baseline",
        })

        # Crash run
        run_number += 1
        log(f"Run {run_number}/{total_runs}: Failure scenario (crash rank=1 at iter=200)", BOLD)
        success = run_one(
            world_size=4,
            num_bits=4,
            epochs=args.epochs,
            run_id="failure_scenario",
            dry_run=args.dry_run,
            crash_rank=1,
            crash_iter=200,
            batch_size=args.batch_size,
        )
        results.append({
            "workers": 4, "bits": 4,
            "run_id":  "failure_scenario_w4_b4",
            "success": success,
            "type":    "failure_crash",
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    log("═══════════════ EXPERIMENT SUMMARY ═══════════════", BOLD)

    passed = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    for r in results:
        status = f"{GREEN}✓{RESET}" if r["success"] else f"{RED}✗{RESET}"
        print(f"  {status}  {r['run_id']:<35} [{r['type']}]")

    print()
    log(f"Passed: {len(passed)}/{len(results)}", GREEN if not failed else YELLOW)

    if failed:
        log("Failed runs:", RED)
        for r in failed:
            log(f"  workers={r['workers']} bits={r['bits']} type={r['type']}", RED)

    log("Results in: ./results/", CYAN)
    log("Next step : python analyze.py", CYAN)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Aborted by user.", YELLOW)
        compose_down()
        sys.exit(1)