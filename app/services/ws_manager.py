"""
ws_manager.py — Production WebSocket connection manager.

Provides:
    • Per-engine subscription rooms (clients subscribe to engines)
    • Broadcast to all dashboard/viewer clients
    • Heartbeat keepalive with configurable interval
    • Async message queue per connection (backpressure protection)
    • Connection lifecycle tracking with metrics
    • Thread-safe operations via asyncio.Lock

Architecture
────────────
    Connections are organized into two categories:

    1. DATA PRODUCERS — sensors / simulators pushing readings via `/ws/sensor`
       → readings are buffered, inference runs, predictions broadcast

    2. DASHBOARD VIEWERS — frontends subscribing via `/ws/live`
       → receive prediction broadcasts for subscribed engines
       → can subscribe/unsubscribe to specific engines dynamically

    Protocol (viewer):
        Client → Server:  {"action": "subscribe",   "engines": [1, 5, 10]}
        Client → Server:  {"action": "unsubscribe", "engines": [5]}
        Client → Server:  {"action": "subscribe_all"}
        Client → Server:  {"action": "ping"}

        Server → Client:  {"type": "prediction", ...}
        Server → Client:  {"type": "engine_list", "engines": [...]}
        Server → Client:  {"type": "pong",    "server_time": "..."}
        Server → Client:  {"type": "welcome", "session_id": "..."}
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

HEARTBEAT_INTERVAL = 15       # seconds between server pings
HEARTBEAT_TIMEOUT = 45        # seconds before considering client dead
MESSAGE_QUEUE_SIZE = 256       # max queued messages per connection
BROADCAST_BATCH_DELAY = 0.005  # small delay to batch broadcasts


# ═══════════════════════════════════════════════════════════════════════
# Connection wrapper
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class WSConnection:
    """Wraps a WebSocket with session state and send queue."""

    ws: WebSocket
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    role: str = "viewer"                 # "viewer" or "producer"
    subscribed_engines: Set[int] = field(default_factory=set)
    subscribe_all: bool = False
    connected_at: float = field(default_factory=time.time)
    last_pong: float = field(default_factory=time.time)
    messages_sent: int = 0
    messages_received: int = 0
    _queue: asyncio.Queue = field(default=None, repr=False)

    def __post_init__(self):
        self._queue = asyncio.Queue(maxsize=MESSAGE_QUEUE_SIZE)

    @property
    def peer(self) -> str:
        try:
            return self.ws.client.host if self.ws.client else "unknown"
        except Exception:
            return "unknown"

    @property
    def uptime(self) -> float:
        return round(time.time() - self.connected_at, 1)

    def is_subscribed_to(self, engine_id: int) -> bool:
        """Check if this viewer should receive data for the given engine."""
        return self.subscribe_all or engine_id in self.subscribed_engines

    async def send(self, data: dict) -> bool:
        """
        Enqueue a message for delivery. Returns False if queue is full
        (client is too slow — backpressure protection).
        """
        try:
            self._queue.put_nowait(data)
            return True
        except asyncio.QueueFull:
            logger.warning(
                "Queue full for %s [%s] — dropping message",
                self.session_id, self.peer,
            )
            return False

    async def send_direct(self, data: dict) -> bool:
        """Send immediately, bypassing the queue. For control messages."""
        try:
            await self.ws.send_json(data)
            self.messages_sent += 1
            return True
        except Exception:
            return False

    def to_info(self) -> dict:
        return {
            "session_id": self.session_id,
            "role": self.role,
            "peer": self.peer,
            "subscribed_engines": sorted(self.subscribed_engines),
            "subscribe_all": self.subscribe_all,
            "uptime_seconds": self.uptime,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
        }


# ═══════════════════════════════════════════════════════════════════════
# WebSocket Connection Manager
# ═══════════════════════════════════════════════════════════════════════

class WebSocketManager:
    """
    Manages all WebSocket connections with per-engine routing,
    heartbeat keepalive, and async message delivery.
    """

    def __init__(self):
        self._connections: Dict[str, WSConnection] = {}
        self._lock = asyncio.Lock()
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._stats = {
            "total_connections": 0,
            "total_broadcasts": 0,
            "total_messages": 0,
        }

    # ── Lifecycle ───────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the heartbeat background task."""
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            logger.info("WebSocket manager started (heartbeat=%ds)", HEARTBEAT_INTERVAL)

    async def shutdown(self) -> None:
        """Gracefully close all connections and stop heartbeat."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            for conn in list(self._connections.values()):
                try:
                    await conn.ws.close(code=1001, reason="Server shutting down")
                except Exception:
                    pass
            self._connections.clear()

        logger.info("WebSocket manager shut down")

    # ── Connect / Disconnect ────────────────────────────────────────

    async def connect(
        self, ws: WebSocket, role: str = "viewer",
    ) -> WSConnection:
        """Accept a WebSocket and register it."""
        await ws.accept()
        conn = WSConnection(ws=ws, role=role)

        async with self._lock:
            self._connections[conn.session_id] = conn
            self._stats["total_connections"] += 1

        # Send welcome with session info
        await conn.send_direct({
            "type": "welcome",
            "session_id": conn.session_id,
            "role": role,
            "server_time": datetime.now().isoformat(),
            "heartbeat_interval": HEARTBEAT_INTERVAL,
        })

        logger.info(
            "WS connect [%s] %s as %s (total: %d)",
            conn.session_id, conn.peer, role, len(self._connections),
        )
        return conn

    async def disconnect(self, conn: WSConnection) -> None:
        """Unregister a connection."""
        async with self._lock:
            self._connections.pop(conn.session_id, None)

        logger.info(
            "WS disconnect [%s] %s (%d msgs, %.0fs uptime, total: %d)",
            conn.session_id, conn.peer,
            conn.messages_sent, conn.uptime,
            len(self._connections),
        )

    # ── Message Delivery ────────────────────────────────────────────

    async def drain_queue(self, conn: WSConnection) -> None:
        """
        Continuously drain the connection's send queue.
        Run this as a background task per connection.
        """
        try:
            while True:
                data = await conn._queue.get()
                try:
                    await conn.ws.send_json(data)
                    conn.messages_sent += 1
                except Exception:
                    break
        except asyncio.CancelledError:
            pass

    # ── Broadcast ───────────────────────────────────────────────────

    async def broadcast_prediction(
        self, engine_id: int, prediction: dict,
    ) -> int:
        """
        Broadcast a prediction to all viewers subscribed to this engine.

        Parameters
        ----------
        engine_id : int
        prediction : dict
            The prediction payload (PredictionResponse as dict).

        Returns
        -------
        int — Number of clients that received the broadcast.
        """
        message = {
            "type": "prediction",
            **prediction,
        }

        sent_count = 0
        async with self._lock:
            for conn in self._connections.values():
                if conn.role == "viewer" and conn.is_subscribed_to(engine_id):
                    if await conn.send(message):
                        sent_count += 1

        self._stats["total_broadcasts"] += 1
        self._stats["total_messages"] += sent_count
        return sent_count

    async def broadcast_buffering(
        self, engine_id: int, buffer_info: dict,
    ) -> int:
        """Broadcast buffering status to subscribed viewers."""
        message = {
            "type": "buffering",
            **buffer_info,
        }

        sent_count = 0
        async with self._lock:
            for conn in self._connections.values():
                if conn.role == "viewer" and conn.is_subscribed_to(engine_id):
                    if await conn.send(message):
                        sent_count += 1
        return sent_count

    async def broadcast_to_all(self, message: dict) -> int:
        """Broadcast a message to ALL viewer connections."""
        sent_count = 0
        async with self._lock:
            for conn in self._connections.values():
                if conn.role == "viewer":
                    if await conn.send(message):
                        sent_count += 1
        return sent_count

    # ── Subscription Management ─────────────────────────────────────

    async def handle_viewer_message(
        self, conn: WSConnection, data: dict,
    ) -> Optional[dict]:
        """
        Process a control message from a viewer client.

        Returns a response dict to send back, or None.
        """
        action = data.get("action", "")
        conn.messages_received += 1

        if action == "subscribe":
            engines = data.get("engines", [])
            if isinstance(engines, list):
                conn.subscribed_engines.update(engines)
                logger.info(
                    "[%s] subscribed to engines: %s",
                    conn.session_id, sorted(conn.subscribed_engines),
                )
                return {
                    "type": "subscribed",
                    "engines": sorted(conn.subscribed_engines),
                    "subscribe_all": conn.subscribe_all,
                }

        elif action == "unsubscribe":
            engines = data.get("engines", [])
            if isinstance(engines, list):
                conn.subscribed_engines.difference_update(engines)
                logger.info(
                    "[%s] unsubscribed from engines: %s → remaining: %s",
                    conn.session_id, engines, sorted(conn.subscribed_engines),
                )
                return {
                    "type": "subscribed",
                    "engines": sorted(conn.subscribed_engines),
                    "subscribe_all": conn.subscribe_all,
                }

        elif action == "subscribe_all":
            conn.subscribe_all = True
            logger.info("[%s] subscribed to ALL engines", conn.session_id)
            return {
                "type": "subscribed",
                "engines": "all",
                "subscribe_all": True,
            }

        elif action == "unsubscribe_all":
            conn.subscribe_all = False
            conn.subscribed_engines.clear()
            return {
                "type": "subscribed",
                "engines": [],
                "subscribe_all": False,
            }

        elif action == "ping":
            conn.last_pong = time.time()
            return {
                "type": "pong",
                "server_time": datetime.now().isoformat(),
                "session_id": conn.session_id,
            }

        elif action == "status":
            return {
                "type": "session_status",
                **conn.to_info(),
            }

        else:
            return {
                "type": "error",
                "message": f"Unknown action: '{action}'. "
                           f"Valid: subscribe, unsubscribe, subscribe_all, "
                           f"unsubscribe_all, ping, status",
            }

        return None

    # ── Heartbeat ───────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        """Periodically ping all clients and prune dead ones."""
        while True:
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL)

                now = time.time()
                dead: List[str] = []

                async with self._lock:
                    for sid, conn in self._connections.items():
                        # Send server-side ping
                        try:
                            await conn.ws.send_json({
                                "type": "heartbeat",
                                "server_time": datetime.now().isoformat(),
                            })
                        except Exception:
                            dead.append(sid)
                            continue

                        # Check if client has been unresponsive
                        if now - conn.last_pong > HEARTBEAT_TIMEOUT:
                            logger.warning(
                                "Client [%s] %s timed out (%.0fs)",
                                sid, conn.peer, now - conn.last_pong,
                            )
                            dead.append(sid)

                # Remove dead connections outside the lock
                for sid in dead:
                    conn = self._connections.get(sid)
                    if conn:
                        try:
                            await conn.ws.close(code=1001, reason="Heartbeat timeout")
                        except Exception:
                            pass
                        await self.disconnect(conn)

                if dead:
                    logger.info("Pruned %d dead connections", len(dead))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat error: %s", e)

    # ── Metrics ─────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return connection manager statistics."""
        roles = {"viewer": 0, "producer": 0}
        for conn in self._connections.values():
            roles[conn.role] = roles.get(conn.role, 0) + 1

        return {
            "active_connections": len(self._connections),
            "viewers": roles.get("viewer", 0),
            "producers": roles.get("producer", 0),
            **self._stats,
        }

    def get_connections(self) -> List[dict]:
        """Return info about all active connections."""
        return [conn.to_info() for conn in self._connections.values()]
