"""
orchestrator.py — Worker registry, heartbeat monitor, straggler detection.
CS332 | Distributed SGD Project | Phase 1

Runs as a standalone process. Workers send UDP heartbeats every 500ms.
Orchestrator marks workers as suspected-failed after 1500ms silence (3× interval).
"""

import os
import time
import json
import socket
import logging
import threading
import argparse
from typing import Dict, Optional
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][Orchestrator] %(levelname)s: %(message)s"
)
log = logging.getLogger(__name__)

HEARTBEAT_PORT    = 29600
HEARTBEAT_INTERVAL = 0.5       # workers send every 500 ms
FAILURE_TIMEOUT   = 1.5        # 3× interval → suspected failure
STARTUP_GRACE     = 30.0       # seconds to wait before failure detection starts
STRAGGLER_THRESH  = 2.0        # iteration time ratio to declare straggler


# ---------------------------------------------------------------------------
# Worker state
# ---------------------------------------------------------------------------

@dataclass
class WorkerState:
    worker_id:      int
    last_heartbeat: float = field(default_factory=time.time)
    last_iteration: int   = 0
    last_iter_time: float = 0.0
    status:         str   = "alive"     # alive | suspected | failed | straggler


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:

    def __init__(self, world_size: int, port: int = HEARTBEAT_PORT):
        self.world_size = world_size
        self.port       = port
        self._workers: Dict[int, WorkerState] = {}
        self._lock = threading.Lock()
        self._running = True

        # Register all expected workers — heartbeat timer starts from NOW
        t0 = time.time()
        for wid in range(world_size):
            w = WorkerState(worker_id=wid)
            w.last_heartbeat = t0
            self._workers[wid] = w

        self._start_time = time.time()
        log.info(f"Orchestrator ready | world_size={world_size} | port={port} | grace={STARTUP_GRACE}s")

    # -----------------------------------------------------------------------
    # Heartbeat receiver
    # -----------------------------------------------------------------------

    def _recv_loop(self):
        """UDP server — receives heartbeat datagrams from workers."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self.port))
        sock.settimeout(0.5)
        log.info(f"Heartbeat listener on UDP :{self.port}")

        while self._running:
            try:
                data, _ = sock.recvfrom(1024)
                msg = json.loads(data.decode())
                self._process_heartbeat(msg)
            except socket.timeout:
                pass
            except json.JSONDecodeError as e:
                log.warning(f"Bad heartbeat JSON: {e}")

        sock.close()

    def _process_heartbeat(self, msg: dict):
        wid  = msg.get("worker_id")
        itr  = msg.get("iteration", 0)
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
                log.info(f"Worker {wid} recovered (heartbeat received).")
                w.status = "alive"

    # -----------------------------------------------------------------------
    # Failure detector
    # -----------------------------------------------------------------------

    def _failure_detector_loop(self):
        """Periodically checks for missing heartbeats."""
        while self._running:
            time.sleep(HEARTBEAT_INTERVAL)
            # Don't fire during startup — workers need time to boot
            if time.time() - self._start_time < STARTUP_GRACE:
                continue
            now = time.time()
            with self._lock:
                for wid, w in self._workers.items():
                    if w.status == "failed":
                        continue
                    silence = now - w.last_heartbeat
                    if silence > FAILURE_TIMEOUT:
                        if w.status == "alive":
                            w.status = "suspected"
                            log.warning(
                                f"Worker {wid} SUSPECTED failed "
                                f"(silence={silence:.1f}s)"
                            )
                        elif w.status == "suspected":
                            w.status = "failed"
                            log.error(
                                f"Worker {wid} CONFIRMED failed "
                                f"(silence={silence:.1f}s). Initiating recovery."
                            )
                            self._on_worker_failure(wid)

    def _on_worker_failure(self, worker_id: int):
        """
        Hook called when a worker is confirmed failed.
        Phase 1: just logs.
        Phase 3: will trigger ring reformation or PS worker removal.
        """
        log.error(
            f"[RECOVERY] Worker {worker_id} failed. "
            f"Phase 1: logging only. Phase 3: ring/PS recovery."
        )

    # -----------------------------------------------------------------------
    # Straggler detector
    # -----------------------------------------------------------------------

    def _straggler_detector_loop(self):
        """
        Detects stragglers by comparing per-worker iteration times.
        Emits a STRAGGLER_ALERT log when a worker is >2× median speed.
        """
        while self._running:
            time.sleep(2.0)
            with self._lock:
                alive = [
                    w for w in self._workers.values()
                    if w.status == "alive" and w.last_iter_time > 0
                ]
            if len(alive) < 2:
                continue
            iter_times = [w.last_iter_time for w in alive]
            median_t   = sorted(iter_times)[len(iter_times) // 2]
            for w in alive:
                if w.last_iter_time > STRAGGLER_THRESH * median_t:
                    log.warning(
                        f"[STRAGGLER_ALERT] Worker {w.worker_id} | "
                        f"iter_time={w.last_iter_time:.0f}ms | "
                        f"median={median_t:.0f}ms | "
                        f"ratio={w.last_iter_time/median_t:.1f}x"
                    )

    # -----------------------------------------------------------------------
    # Status API
    # -----------------------------------------------------------------------

    def get_status(self) -> dict:
        with self._lock:
            return {
                wid: {
                    "status":      w.status,
                    "iteration":   w.last_iteration,
                    "iter_time_ms": w.last_iter_time,
                    "last_heartbeat_ago": round(time.time() - w.last_heartbeat, 2),
                }
                for wid, w in self._workers.items()
            }

    def alive_workers(self):
        with self._lock:
            return [wid for wid, w in self._workers.items() if w.status == "alive"]

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self):
        threading.Thread(target=self._recv_loop,              daemon=True).start()
        threading.Thread(target=self._failure_detector_loop,  daemon=True).start()
        threading.Thread(target=self._straggler_detector_loop, daemon=True).start()
        log.info("Orchestrator threads started.")

    def stop(self):
        self._running = False

    def run_forever(self):
        """Block while printing periodic status."""
        self.start()
        try:
            while True:
                time.sleep(10)
                status = self.get_status()
                alive = [wid for wid, s in status.items() if s["status"] == "alive"]
                log.info(f"Status | alive={alive} | full={json.dumps(status)}")
        except KeyboardInterrupt:
            log.info("Orchestrator shutting down.")
            self.stop()


# ---------------------------------------------------------------------------
# Heartbeat sender (runs inside each worker process)
# ---------------------------------------------------------------------------

class HeartbeatSender(threading.Thread):
    """
    Sends UDP heartbeats from a worker to the orchestrator.
    Instantiated inside worker.py training loop.
    """

    def __init__(
        self,
        worker_id: int,
        orchestrator_host: str = "orchestrator",
        orchestrator_port: int = HEARTBEAT_PORT,
        interval: float = HEARTBEAT_INTERVAL,
    ):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.host      = orchestrator_host
        self.port      = orchestrator_port
        self.interval  = interval
        self._iteration = 0
        self._iter_time = 0.0
        self._running   = True
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def update(self, iteration: int, iter_time_ms: float):
        self._iteration = iteration
        self._iter_time = iter_time_ms

    def run(self):
        while self._running:
            msg = json.dumps({
                "worker_id":   self.worker_id,
                "iteration":   self._iteration,
                "iter_time_ms": self._iter_time,
                "timestamp":   time.time(),
            }).encode()
            try:
                self._sock.sendto(msg, (self.host, self.port))
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self._running = False
        self._sock.close()


# ---------------------------------------------------------------------------
# Standalone orchestrator process entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--port",       type=int, default=HEARTBEAT_PORT)
    args = parser.parse_args()

    orc = Orchestrator(world_size=args.world_size, port=args.port)
    orc.run_forever()