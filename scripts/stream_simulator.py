"""
stream_simulator.py — Simulate real-time turbofan sensor data streaming.

Reads the C-MAPSS dataset row-by-row grouped by engine and emits each
reading at a configurable interval (1–2 seconds) over HTTP POST or
WebSocket, making the static dataset behave like a live IoT sensor feed.

Features
────────
• HTTP POST mode (default) or WebSocket mode
• Pause / resume via keyboard (spacebar / Enter)
• Configurable emission rate and engine selection
• Graceful shutdown with Ctrl+C
• Detailed logging of every emitted row

Usage
─────
    # Terminal 1 — start the receiver server:
    python3 stream_server.py

    # Terminal 2 — start the simulator:
    python3 stream_simulator.py                              # HTTP POST, all engines
    python3 stream_simulator.py --mode websocket             # WebSocket mode
    python3 stream_simulator.py --engine 1 --engine 5        # specific engines
    python3 stream_simulator.py --interval 0.5               # faster emission
    python3 stream_simulator.py --data-dir data              # custom data path
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Column schema (must match data_pipeline.py / config.py)
# ═══════════════════════════════════════════════════════════════════════

INDEX_COLS = ["unit_number", "time_cycles"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS


# ═══════════════════════════════════════════════════════════════════════
# Stream Controller (pause / resume)
# ═══════════════════════════════════════════════════════════════════════

class StreamController:
    """
    Thread-safe pause/resume controller.

    Monitors keyboard input in a background thread.
    Press Enter to toggle pause/resume.
    """

    def __init__(self):
        self._paused = False
        self._stopped = False
        self._lock = asyncio.Lock()
        self._stats = {
            "emitted": 0,
            "engines_completed": 0,
            "start_time": None,
        }

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def stopped(self) -> bool:
        return self._stopped

    def toggle_pause(self) -> None:
        self._paused = not self._paused
        state = "⏸  PAUSED" if self._paused else "▶  RESUMED"
        logger.info(state)

    def stop(self) -> None:
        self._stopped = True
        logger.info("⏹  Stream stopped")

    def record_emit(self) -> None:
        if self._stats["start_time"] is None:
            self._stats["start_time"] = time.time()
        self._stats["emitted"] += 1

    def record_engine_complete(self) -> None:
        self._stats["engines_completed"] += 1

    @property
    def summary(self) -> Dict:
        elapsed = 0
        if self._stats["start_time"]:
            elapsed = time.time() - self._stats["start_time"]
        return {
            **self._stats,
            "elapsed_seconds": round(elapsed, 1),
            "rate": round(self._stats["emitted"] / max(elapsed, 0.001), 2),
        }


# ═══════════════════════════════════════════════════════════════════════
# Data Loader
# ═══════════════════════════════════════════════════════════════════════

def load_engine_data(
    data_dir: Path,
    engine_ids: Optional[List[int]] = None,
) -> Dict[int, pd.DataFrame]:
    """
    Load train_FD001.txt and group by engine.

    Parameters
    ----------
    data_dir : Path
        Directory containing train_FD001.txt
    engine_ids : list[int] | None
        If provided, only load these engines.

    Returns
    -------
    dict[int, DataFrame]  — engine_id → rows for that engine
    """
    filepath = data_dir / "train_FD001.txt"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{filepath.resolve()}'. "
            f"Run generate_sample_data.py or download the real data."
        )

    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=ALL_COLS, engine="python")
    logger.info("Loaded %d rows from %s", len(df), filepath.name)

    if engine_ids:
        df = df[df["unit_number"].isin(engine_ids)]
        logger.info("Filtered to engines: %s (%d rows)", engine_ids, len(df))

    engines = {}
    for uid, group in df.groupby("unit_number"):
        engines[uid] = group.sort_values("time_cycles").reset_index(drop=True)

    logger.info("Loaded %d engines", len(engines))
    return engines


def row_to_payload(row: pd.Series) -> Dict:
    """Convert a DataFrame row to the JSON payload format."""
    sensor_values = {}
    for col in SETTING_COLS + SENSOR_COLS:
        sensor_values[col] = round(float(row[col]), 4)

    return {
        "unit_number": int(row["unit_number"]),
        "time_cycles": int(row["time_cycles"]),
        "sensor_values": sensor_values,
    }


# ═══════════════════════════════════════════════════════════════════════
# HTTP POST Sender
# ═══════════════════════════════════════════════════════════════════════

async def send_http(
    session,
    url: str,
    payload: Dict,
) -> bool:
    """Send a single reading via HTTP POST. Returns True on success."""
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status == 200:
                return True
            else:
                body = await resp.text()
                logger.warning("HTTP %d: %s", resp.status, body[:200])
                return False
    except Exception as e:
        logger.error("HTTP POST failed: %s", e)
        return False


# ═══════════════════════════════════════════════════════════════════════
# WebSocket Sender
# ═══════════════════════════════════════════════════════════════════════

async def send_ws(ws, payload: Dict) -> bool:
    """Send a single reading via WebSocket. Returns True on success."""
    try:
        await ws.send_str(json.dumps(payload))
        resp = await ws.receive_json()
        return resp.get("status") == "accepted"
    except Exception as e:
        logger.error("WebSocket send failed: %s", e)
        return False


# ═══════════════════════════════════════════════════════════════════════
# Main Streaming Loop
# ═══════════════════════════════════════════════════════════════════════

async def stream_data(
    engines: Dict[int, pd.DataFrame],
    controller: StreamController,
    mode: str = "http",
    server_url: str = "http://localhost:8000",
    interval: float = 1.0,
    jitter: float = 0.5,
) -> None:
    """
    Stream engine data row-by-row to the backend server.

    Parameters
    ----------
    engines : dict[int, DataFrame]
    controller : StreamController
    mode : 'http' | 'websocket'
    server_url : str
        Base URL of the receiver server.
    interval : float
        Base delay between emissions (seconds).
    jitter : float
        Random ± jitter added to interval.
    """
    import aiohttp
    import random

    logger.info("━" * 50)
    logger.info("  📡 Starting sensor stream")
    logger.info("  Mode     : %s", mode.upper())
    logger.info("  Server   : %s", server_url)
    logger.info("  Interval : %.1f ± %.1fs", interval, jitter)
    logger.info("  Engines  : %d", len(engines))
    logger.info("━" * 50)
    logger.info("  Press Enter to pause/resume. Ctrl+C to stop.")
    logger.info("━" * 50)

    async with aiohttp.ClientSession() as session:
        ws = None

        # Connect WebSocket if needed
        if mode == "websocket":
            if server_url.startswith("ws"):
                ws_url = server_url
            else:
                ws_url = server_url.replace("http", "ws") + "/ws/sensor"
                
            try:
                ws = await session.ws_connect(ws_url)
                logger.info("WebSocket connected to %s", ws_url)
            except Exception as e:
                logger.error("Failed to connect WebSocket: %s", e)
                return

        http_url = server_url + "/predict/stream"
        engine_ids = sorted(engines.keys())

        try:
            for engine_id in engine_ids:
                if controller.stopped:
                    break

                engine_df = engines[engine_id]
                total_cycles = len(engine_df)

                logger.info(
                    "\n🔧 Engine %d — %d cycles",
                    engine_id, total_cycles,
                )

                for idx, (_, row) in enumerate(engine_df.iterrows()):
                    if controller.stopped:
                        break

                    # Pause check
                    while controller.paused and not controller.stopped:
                        await asyncio.sleep(0.2)

                    payload = row_to_payload(row)

                    # Send
                    if mode == "websocket" and ws is not None:
                        success = await send_ws(ws, payload)
                    else:
                        success = await send_http(session, http_url, payload)

                    status = "✓" if success else "✗"
                    controller.record_emit()

                    logger.info(
                        "  %s  Engine %3d  Cycle %3d/%d  [%d total]",
                        status, engine_id,
                        int(row["time_cycles"]), total_cycles,
                        controller.summary["emitted"],
                    )

                    # Wait with jitter
                    delay = interval + random.uniform(-jitter, jitter)
                    delay = max(0.1, delay)
                    await asyncio.sleep(delay)

                controller.record_engine_complete()
                logger.info(
                    "  ✅ Engine %d complete (%d/%d engines done)",
                    engine_id,
                    controller.summary["engines_completed"],
                    len(engines),
                )

        finally:
            if ws is not None:
                await ws.close()

    # Print summary
    s = controller.summary
    logger.info("\n" + "━" * 50)
    logger.info("  📊 Stream Summary")
    logger.info("━" * 50)
    logger.info("  Readings emitted  : %d", s["emitted"])
    logger.info("  Engines completed : %d", s["engines_completed"])
    logger.info("  Total time        : %.1fs", s["elapsed_seconds"])
    logger.info("  Avg rate          : %.2f readings/sec", s["rate"])
    logger.info("━" * 50)


# ═══════════════════════════════════════════════════════════════════════
# Keyboard listener (pause/resume)
# ═══════════════════════════════════════════════════════════════════════

async def keyboard_listener(controller: StreamController) -> None:
    """Listen for Enter keypress to toggle pause in a non-blocking way."""
    loop = asyncio.get_event_loop()
    try:
        while not controller.stopped:
            # Read a line from stdin asynchronously
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if line is not None:
                controller.toggle_pause()
    except (EOFError, OSError):
        pass


# ═══════════════════════════════════════════════════════════════════════
# CLI & Entry Point
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate real-time turbofan sensor streaming.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Directory containing train_FD001.txt (default: data/)",
    )
    parser.add_argument(
        "--mode", choices=["http", "websocket"], default="http",
        help="Transport mode: 'http' (POST) or 'websocket' (default: http)",
    )
    parser.add_argument(
        "--server", default="http://localhost:8000",
        help="Backend server URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="Base interval between emissions in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--jitter", type=float, default=0.5,
        help="Random jitter ± seconds added to interval (default: 0.5)",
    )
    parser.add_argument(
        "--engine", type=int, action="append", dest="engines",
        help="Stream specific engine(s). Repeat for multiple: --engine 1 --engine 5",
    )
    parser.add_argument(
        "--max-engines", type=int, default=None,
        help="Limit number of engines to stream (default: all)",
    )
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    # Load data
    engines = load_engine_data(args.data_dir, args.engines)

    if args.max_engines:
        engine_ids = sorted(engines.keys())[: args.max_engines]
        engines = {k: engines[k] for k in engine_ids}
        logger.info("Limited to first %d engines", args.max_engines)

    controller = StreamController()

    # Handle Ctrl+C
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, controller.stop)

    # Run stream + keyboard listener concurrently
    await asyncio.gather(
        stream_data(
            engines=engines,
            controller=controller,
            mode=args.mode,
            server_url=args.server,
            interval=args.interval,
            jitter=args.jitter,
        ),
        keyboard_listener(controller),
        return_exceptions=True,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
