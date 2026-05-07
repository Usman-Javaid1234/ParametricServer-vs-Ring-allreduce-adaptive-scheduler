"""
orchestrator.py — Worker registry, heartbeat monitor, straggler detection.
CS332 | Distributed SGD Project | Phase 1

Timeouts tuned for Docker on a single machine where CIFAR-10 download
can take several minutes before workers start sending heartbeats.
"""

import os
import time
import json
import socket
import logging
import threading
import argparse
from typing import Dict
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][Orchestrator] %(levelname)s: %(message)s"
)
log = logging.getLogger(__name__)

HEARTBEAT_PORT     = 29600
HEARTBEAT_INTERVAL = 0.5    # workers send every 500 ms
SUSPECT_TIMEOUT    = 5.0    # 10× interval before suspected
CONFIRM_TIMEOUT    = 10.0   # 20× interval before confirmed failed
STARTUP_GRACE      = 600.0  # 10 min grace — covers CIFAR download time
STRAGGLER_THRESH   = 3.0    # iteration time ratio to flag straggler


@dataclass
class WorkerState:
    worker_id:      int
    last_heartbeat: float = field(default_factory=time.time)
    last_iteration: int   = 0
    last_iter_time: float = 0.0
    status:         str   = "alive"


class Orchestrator:

    def __init__(self, world_size: int, port: int = HEARTBEAT_PORT):
        self.world_size  = world_size
        self.port        = port
        self._workers: Dict[int, WorkerState] = {}
        self._lock       = threading.Lock()
        self._running    = True
        self._start_time = time.time()

        t0 = time.time()
        for wid in range(world_size):
            w = WorkerState(worker_id=wid)
            w.last_heartbeat = t0
            self._workers[wid] = w

        log.info(f"Orchestrator ready | world_size={world_size} | "
                 f"port={port} | startup_grace={STARTUP_GRACE}s")

    # ---- Heartbeat receiver ----

    def _recv_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self.port))
        sock.settimeout(1.0)
        log.info(f"Heartbeat listener on UDP :{self.port}")

        while self._running:
            try:
                data, _ = sock.recvfrom(1024)
                msg = json.loads(data.decode())
                self._process_heartbeat(msg)
            except socket.timeout:
                pass
            except Exception as e:
                log.debug(f"Recv error: {e}")
        sock.close()

    def _process_heartbeat(self, msg: dict):
        wid   = msg.get("worker_id")
        itr   = msg.get("iteration", 0)
        itime = msg.get("iter_time_ms", 0.0)
        if wid is None:
            return
        with self._lock:
            if wid not in self._workers:
                self._workers[wid] = WorkerState(worker_id=wid)
            w = self._workers[wid]
            w.last_heartbeat = time.time()
            w.last_iteration = itr
            w.last_iter_time = itime
            if w.status in ("suspected", "failed"):
                log.info(f"Worker {wid} recovered.")
                w.status = "alive"

    # ---- Failure detector ----

    def _failure_detector_loop(self):
        while self._running:
            time.sleep(HEARTBEAT_INTERVAL)
            elapsed = time.time() - self._start_time
            if elapsed < STARTUP_GRACE:
                # Still in grace period — only log occasionally
                if int(elapsed) % 60 == 0:
                    log.info(f"Startup grace: {int(STARTUP_GRACE - elapsed)}s remaining")
                continue

            now = time.time()
            with self._lock:
                for wid, w in self._workers.items():
                    if w.status == "failed":
                        continue
                    silence = now - w.last_heartbeat
                    if silence > CONFIRM_TIMEOUT and w.status == "suspected":
                        w.status = "failed"
                        log.error(f"Worker {wid} CONFIRMED failed "
                                  f"(silence={silence:.1f}s)")
                        self._on_worker_failure(wid)
                    elif silence > SUSPECT_TIMEOUT and w.status == "alive":
                        w.status = "suspected"
                        log.warning(f"Worker {wid} SUSPECTED failed "
                                    f"(silence={silence:.1f}s)")

    def _on_worker_failure(self, worker_id: int):
        log.error(f"[RECOVERY] Worker {worker_id} failed. "
                  f"Phase 1: logging only. Phase 3: ring/PS recovery.")

    # ---- Straggler detector ----

    def _straggler_detector_loop(self):
        while self._running:
            time.sleep(5.0)
            with self._lock:
                alive = [w for w in self._workers.values()
                         if w.status == "alive" and w.last_iter_time > 0]
            if len(alive) < 2:
                continue
            times  = sorted(w.last_iter_time for w in alive)
            median = times[len(times) // 2]
            for w in alive:
                if w.last_iter_time > STRAGGLER_THRESH * median:
                    log.warning(
                        f"[STRAGGLER] Worker {w.worker_id} | "
                        f"{w.last_iter_time:.0f}ms vs median {median:.0f}ms "
                        f"({w.last_iter_time/median:.1f}x)"
                    )

    # ---- Status ----

    def get_status(self) -> dict:
        with self._lock:
            return {
                wid: {
                    "status":    w.status,
                    "iteration": w.last_iteration,
                    "iter_time_ms": w.last_iter_time,
                    "last_heartbeat_ago": round(time.time() - w.last_heartbeat, 2),
                }
                for wid, w in self._workers.items()
            }

    def alive_workers(self):
        with self._lock:
            return [wid for wid, w in self._workers.items() if w.status == "alive"]

    # ---- Lifecycle ----

    def start(self):
        threading.Thread(target=self._recv_loop,               daemon=True).start()
        threading.Thread(target=self._failure_detector_loop,   daemon=True).start()
        threading.Thread(target=self._straggler_detector_loop, daemon=True).start()
        log.info("Orchestrator threads started.")

    def stop(self):
        self._running = False

    def run_forever(self):
        self.start()
        try:
            while True:
                time.sleep(30)
                s = self.get_status()
                alive = [wid for wid, v in s.items() if v["status"] == "alive"]
                log.info(f"Status | alive={alive}")
        except KeyboardInterrupt:
            log.info("Shutting down.")
            self.stop()


# ---------------------------------------------------------------------------
# HeartbeatSender — runs inside each worker as a daemon thread
# ---------------------------------------------------------------------------

class HeartbeatSender(threading.Thread):

    def __init__(self, worker_id: int,
                 orchestrator_host: str = "orchestrator",
                 orchestrator_port: int = HEARTBEAT_PORT,
                 interval: float = HEARTBEAT_INTERVAL):
        super().__init__(daemon=True)
        self.worker_id  = worker_id
        self.host       = orchestrator_host
        self.port       = orchestrator_port
        self.interval   = interval
        self._iteration = 0
        self._iter_time = 0.0
        self._running   = True
        self._sock      = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def update(self, iteration: int, iter_time_ms: float):
        self._iteration = iteration
        self._iter_time = iter_time_ms

    def run(self):
        while self._running:
            try:
                msg = json.dumps({
                    "worker_id":    self.worker_id,
                    "iteration":    self._iteration,
                    "iter_time_ms": self._iter_time,
                    "timestamp":    time.time(),
                }).encode()
                self._sock.sendto(msg, (self.host, self.port))
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self._running = False
        self._sock.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--port",       type=int, default=HEARTBEAT_PORT)
    args = parser.parse_args()
    Orchestrator(world_size=args.world_size, port=args.port).run_forever()