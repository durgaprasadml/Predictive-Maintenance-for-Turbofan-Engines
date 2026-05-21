"""
api.py — FastAPI backend for real-time RUL prediction.

Endpoints
─────────
    POST   /predict           →  Single-sequence RUL prediction
    POST   /predict/batch     →  Batch RUL prediction
    POST   /predict/stream    →  Single reading with auto sliding window
    WS     /ws/sensor         →  Producer: push sensor data, receive predictions
    WS     /ws/live           →  Viewer: subscribe to engines, receive broadcasts
    GET    /health            →  Service health check
    GET    /engines           →  List engines with active sliding windows
    GET    /engines/{id}      →  Engine detail + current buffer state
    DELETE /engines/{id}      →  Clear sliding window for an engine
    GET    /ws/stats          →  WebSocket connection statistics

Architecture
────────────
    • Model loaded once at startup via lifespan context.
    • Per-engine sliding window buffers (last 30 cycles) maintained in
      memory for WebSocket streaming — no repeated payload needed.
    • Thread-safe buffer operations via asyncio.Lock.
    • Health status derived from predicted RUL:
        RUL ≥ 50  →  Healthy
        15 ≤ RUL < 50 →  Warning
        RUL < 15  →  Critical

Usage
─────
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import joblib
import numpy as np
from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

MODEL_DIR = Path("outputs/models")
META_PATH = MODEL_DIR / "metadata.json"
SEQUENCE_LENGTH = 30  # sliding window size

# Health status thresholds (in cycles)
HEALTHY_THRESHOLD = 50
WARNING_THRESHOLD = 15


def classify_status(rul: float) -> str:
    """Map predicted RUL to a health status label."""
    if rul >= HEALTHY_THRESHOLD:
        return "Healthy"
    elif rul >= WARNING_THRESHOLD:
        return "Warning"
    else:
        return "Critical"


# ═══════════════════════════════════════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════════════════════════════════════

class SensorReading(BaseModel):
    """A single row of sensor data from one engine cycle."""
    unit_number: int = Field(..., ge=1, description="Engine unit ID")
    time_cycles: int = Field(..., ge=1, description="Current cycle number")
    sensor_values: Dict[str, float] = Field(
        ...,
        description="Sensor and setting values keyed by column name",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "unit_number": 1,
                "time_cycles": 42,
                "sensor_values": {
                    "setting_1": 0.0023,
                    "setting_2": 0.0003,
                    "setting_3": 100.0,
                    "s_2": 642.15,
                    "s_3": 1589.70,
                    "s_4": 1400.60,
                    "s_6": 21.61,
                    "s_7": 554.36,
                    "s_8": 2388.02,
                    "s_9": 9046.19,
                    "s_11": 47.47,
                    "s_12": 521.66,
                    "s_13": 2388.03,
                    "s_14": 8138.62,
                    "s_15": 8.4195,
                    "s_17": 392.0,
                    "s_20": 39.06,
                    "s_21": 23.42,
                },
            }
        }


class SequenceInput(BaseModel):
    """A complete pre-formed sequence for direct prediction."""
    unit_number: int = Field(..., ge=1)
    sequence: List[List[float]] = Field(
        ...,
        description="2D array of shape (30, 18) — pre-normalized sensor values",
    )

    @validator("sequence")
    def validate_shape(cls, v):
        if len(v) != SEQUENCE_LENGTH:
            raise ValueError(
                f"Sequence must have {SEQUENCE_LENGTH} timesteps, got {len(v)}"
            )
        if v and len(v[0]) != 18:
            raise ValueError(f"Each timestep must have 18 features, got {len(v[0])}")
        return v


class BatchInput(BaseModel):
    """Batch of sequences for bulk prediction."""
    sequences: List[SequenceInput]


class PredictionResponse(BaseModel):
    """Standard prediction response."""
    unit_number: int
    predicted_rul: float
    status: str
    confidence_band: Dict[str, float] = Field(
        default_factory=dict,
        description="Approximate confidence interval",
    )
    timestamp: str
    model_used: str


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int
    latency_ms: float


class EngineState(BaseModel):
    """Current sliding window state for an engine."""
    unit_number: int
    buffer_size: int
    latest_cycle: Optional[int]
    latest_rul: Optional[float]
    latest_status: Optional[str]
    is_ready: bool = Field(description="True when buffer has >= 30 cycles")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    input_shape: List[int]
    feature_columns: List[str]
    active_engines: int
    uptime_seconds: float


# ═══════════════════════════════════════════════════════════════════════
# Sliding Window Buffer Manager
# ═══════════════════════════════════════════════════════════════════════

class EngineBufferManager:
    """
    Maintains per-engine sliding window buffers for streaming inference.

    Each engine has a deque of the last `window_size` feature vectors.
    When a new sensor reading arrives, it's appended to the buffer.
    Once the buffer reaches `window_size`, a complete sequence is
    available for prediction.
    """

    def __init__(self, window_size: int, feature_cols: List[str]):
        self._window_size = window_size
        self._feature_cols = feature_cols
        self._buffers: Dict[int, Deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self._metadata: Dict[int, Dict[str, Any]] = defaultdict(dict)
        self._lock = asyncio.Lock()

    async def add_reading(
        self, reading: SensorReading,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Add a sensor reading to the engine's buffer.

        Returns
        -------
        (sequence | None, metadata)
            sequence: np.ndarray of shape (window_size, n_features) if buffer
                      is full, else None.
            metadata: dict with buffer state info.
        """
        async with self._lock:
            uid = reading.unit_number

            # Extract feature vector in the correct column order
            feature_vec = self._extract_features(reading.sensor_values)
            self._buffers[uid].append(feature_vec)

            self._metadata[uid] = {
                "latest_cycle": reading.time_cycles,
                "buffer_size": len(self._buffers[uid]),
                "updated_at": datetime.now().isoformat(),
            }

            is_ready = len(self._buffers[uid]) >= self._window_size
            sequence = None
            if is_ready:
                sequence = np.array(
                    list(self._buffers[uid]), dtype=np.float32,
                )

            return sequence, {
                "buffer_size": len(self._buffers[uid]),
                "is_ready": is_ready,
                "unit_number": uid,
            }

    def _extract_features(self, sensor_values: Dict[str, float]) -> np.ndarray:
        """Extract features in the correct order from sensor dict."""
        vec = []
        for col in self._feature_cols:
            val = sensor_values.get(col)
            if val is None:
                raise ValueError(
                    f"Missing feature '{col}' in sensor_values. "
                    f"Expected: {self._feature_cols}"
                )
            vec.append(float(val))
        return np.array(vec, dtype=np.float32)

    async def get_engine_state(self, uid: int) -> Optional[Dict]:
        async with self._lock:
            if uid not in self._buffers:
                return None
            return {
                "unit_number": uid,
                "buffer_size": len(self._buffers[uid]),
                **self._metadata.get(uid, {}),
            }

    async def get_all_engines(self) -> List[Dict]:
        async with self._lock:
            states = []
            for uid in sorted(self._buffers.keys()):
                states.append({
                    "unit_number": uid,
                    "buffer_size": len(self._buffers[uid]),
                    "is_ready": len(self._buffers[uid]) >= self._window_size,
                    **self._metadata.get(uid, {}),
                })
            return states

    async def clear_engine(self, uid: int) -> bool:
        async with self._lock:
            if uid in self._buffers:
                del self._buffers[uid]
                self._metadata.pop(uid, None)
                return True
            return False

    async def clear_all(self) -> int:
        async with self._lock:
            count = len(self._buffers)
            self._buffers.clear()
            self._metadata.clear()
            return count


# ═══════════════════════════════════════════════════════════════════════
# Application state (populated at startup)
# ═══════════════════════════════════════════════════════════════════════

class AppState:
    predictor = None
    buffer_manager: Optional[EngineBufferManager] = None
    ws_manager = None       # WebSocketManager instance
    feature_cols: List[str] = []
    start_time: float = 0.0
    scaler: Optional[Any] = None

state = AppState()


# ═══════════════════════════════════════════════════════════════════════
# Lifespan: load model at startup
# ═══════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the pre-trained model and metadata once at startup."""
    from app.models.inference import RULPredictor
    from app.services.ws_manager import WebSocketManager

    logger.info("Loading model from %s ...", MODEL_DIR)
    state.predictor = RULPredictor.load(MODEL_DIR)
    state.start_time = time.time()
    
    # Load the MinMaxScaler for inference
    scaler_path = MODEL_DIR / "scaler.joblib"
    if scaler_path.exists():
        state.scaler = joblib.load(scaler_path)
        logger.info("Loaded scaler from %s", scaler_path)
    else:
        logger.warning("No scaler found at %s. Predictions may be saturated if inputs are unscaled.", scaler_path)

    # Load feature column order from metadata
    if META_PATH.exists():
        with open(META_PATH) as f:
            meta = json.load(f)
        state.feature_cols = meta.get("feature_cols", [])
    else:
        state.feature_cols = [
            "setting_1", "setting_2", "setting_3",
            *[f"s_{i}" for i in [2,3,4,6,7,8,9,11,12,13,14,15,17,20,21]],
        ]

    state.buffer_manager = EngineBufferManager(
        window_size=SEQUENCE_LENGTH,
        feature_cols=state.feature_cols,
    )

    # Initialize WebSocket manager with heartbeat
    state.ws_manager = WebSocketManager()
    await state.ws_manager.start()

    logger.info(
        "✅ Model loaded: %s (input=%s, %d features)",
        state.predictor.name,
        state.predictor._expected_shape,
        len(state.feature_cols),
    )
    logger.info("Feature columns: %s", state.feature_cols)

    yield

    logger.info("Shutting down WebSocket manager...")
    await state.ws_manager.shutdown()


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_prediction(unit_number: int, sequence: np.ndarray) -> PredictionResponse:
    """Run inference and build a response object."""
    # Ensure data is normalized before prediction to prevent model saturation
    if getattr(state, "scaler", None) is not None:
        sequence = state.scaler.transform(sequence)
        
    rul = state.predictor.predict_rul(sequence)
    status = classify_status(rul)

    # Simple confidence band (±10% heuristic for demo; replace with
    # proper quantile regression or MC dropout in production)
    margin = max(rul * 0.10, 2.0)

    return PredictionResponse(
        unit_number=unit_number,
        predicted_rul=round(rul, 2),
        status=status,
        confidence_band={
            "lower": round(max(0, rul - margin), 2),
            "upper": round(rul + margin, 2),
        },
        timestamp=datetime.now().isoformat(),
        model_used=state.predictor.name,
    )


# ═══════════════════════════════════════════════════════════════════════
# POST /predict — Single sequence
# ═══════════════════════════════════════════════════════════════════════

router = APIRouter()

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict RUL from a complete sensor sequence",
    tags=["Prediction"],
)
async def predict_rul(body: SequenceInput):
    """
    Submit a complete 30×18 sensor sequence and receive a RUL prediction.

    The input sequence must be pre-normalized using the pipeline's
    MinMaxScaler. Use this endpoint for one-shot inference when you
    already have a full window of data.
    """
    sequence = np.array(body.sequence, dtype=np.float32)
    return _make_prediction(body.unit_number, sequence)


# ═══════════════════════════════════════════════════════════════════════
# POST /predict/batch — Batch prediction
# ═══════════════════════════════════════════════════════════════════════

@router.post(
    "/predict/batch",
    response_model=BatchResponse,
    summary="Batch prediction for multiple engines",
    tags=["Prediction"],
)
async def predict_batch(body: BatchInput):
    """Submit multiple sequences and get all predictions in one call."""
    t0 = time.time()
    predictions = []
    for item in body.sequences:
        seq = np.array(item.sequence, dtype=np.float32)
        pred = _make_prediction(item.unit_number, seq)
        predictions.append(pred)

    return BatchResponse(
        predictions=predictions,
        count=len(predictions),
        latency_ms=round((time.time() - t0) * 1000, 2),
    )


# ═══════════════════════════════════════════════════════════════════════
# POST /predict/stream — Single reading with sliding window
# ═══════════════════════════════════════════════════════════════════════

@router.post(
    "/predict/stream",
    summary="Submit a single sensor reading (sliding window auto-managed)",
    tags=["Streaming"],
)
async def predict_stream(reading: SensorReading):
    """
    Submit a single cycle's sensor data. The server maintains a
    per-engine sliding window buffer. Once 30 cycles accumulate,
    predictions begin automatically.

    This is the HTTP equivalent of the WebSocket `/ws` endpoint —
    useful when WebSocket is not available.
    """
    sequence, meta = await state.buffer_manager.add_reading(reading)

    if sequence is None:
        return {
            "unit_number": reading.unit_number,
            "status": "buffering",
            "buffer_size": meta["buffer_size"],
            "cycles_needed": SEQUENCE_LENGTH - meta["buffer_size"],
            "message": (
                f"Collecting data — {meta['buffer_size']}/{SEQUENCE_LENGTH} "
                f"cycles buffered"
            ),
        }

    pred = _make_prediction(reading.unit_number, sequence)
    pred_dict = pred.dict()

    # Also broadcast to WS viewers so dashboards see HTTP-sourced predictions
    if state.ws_manager:
        await state.ws_manager.broadcast_prediction(
            reading.unit_number, pred_dict,
        )

    return pred_dict


# ═══════════════════════════════════════════════════════════════════════
# WebSocket /ws/sensor — Producer: push sensor data
# ═══════════════════════════════════════════════════════════════════════

@router.websocket("/ws/sensor")
async def ws_sensor_producer(ws: WebSocket):
    """
    Producer WebSocket — sensors/simulators push readings here.

    Each reading is buffered in the per-engine sliding window.
    Once full, inference runs, the producer gets the prediction back,
    AND it's broadcast to all subscribed viewer clients.

    Protocol
    ────────
    Client → Server:  sensor reading JSON
    Server → Client:  prediction or buffering status
    Server → Viewers: broadcast prediction to subscribed dashboards
    """
    conn = await state.ws_manager.connect(ws, role="producer")
    drain_task = asyncio.create_task(state.ws_manager.drain_queue(conn))

    try:
        while True:
            raw = await ws.receive_text()
            conn.messages_received += 1
            conn.last_pong = time.time()  # any message counts as alive

            try:
                data = json.loads(raw)
                reading = SensorReading(**data)
            except (json.JSONDecodeError, Exception) as e:
                await conn.send_direct({
                    "type": "error",
                    "message": f"Invalid payload: {str(e)}",
                })
                continue

            # Add to sliding window
            sequence, meta = await state.buffer_manager.add_reading(reading)

            if sequence is None:
                buffer_msg = {
                    "type": "buffering",
                    "unit_number": reading.unit_number,
                    "time_cycles": reading.time_cycles,
                    "buffer_size": meta["buffer_size"],
                    "cycles_needed": SEQUENCE_LENGTH - meta["buffer_size"],
                }
                await conn.send_direct(buffer_msg)
                # Also notify subscribed viewers
                await state.ws_manager.broadcast_buffering(
                    reading.unit_number, buffer_msg,
                )
            else:
                pred = _make_prediction(reading.unit_number, sequence)
                pred_dict = pred.dict()

                # Send back to producer
                await conn.send_direct({
                    "type": "prediction",
                    **pred_dict,
                })

                # Broadcast to all subscribed viewers
                await state.ws_manager.broadcast_prediction(
                    reading.unit_number, pred_dict,
                )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WS sensor error [%s]: %s", conn.session_id, e)
    finally:
        drain_task.cancel()
        await state.ws_manager.disconnect(conn)


# ═══════════════════════════════════════════════════════════════════════
# WebSocket /ws/live — Viewer: subscribe to engine predictions
# ═══════════════════════════════════════════════════════════════════════

@router.websocket("/ws/live")
async def ws_live_viewer(ws: WebSocket):
    """
    Viewer WebSocket — dashboards subscribe here to receive
    live prediction broadcasts.

    Protocol
    ────────
    Client → Server:
        {"action": "subscribe",     "engines": [1, 5, 10]}
        {"action": "unsubscribe",   "engines": [5]}
        {"action": "subscribe_all"}
        {"action": "ping"}
        {"action": "status"}

    Server → Client:
        {"type": "welcome",     "session_id": "...", ...}
        {"type": "subscribed",  "engines": [1, 10]}
        {"type": "prediction",  "unit_number": 1, ...}  ← broadcast
        {"type": "buffering",   "unit_number": 1, ...}  ← broadcast
        {"type": "heartbeat",   "server_time": "..."}   ← keepalive
        {"type": "pong",        "server_time": "..."}
    """
    conn = await state.ws_manager.connect(ws, role="viewer")
    drain_task = asyncio.create_task(state.ws_manager.drain_queue(conn))

    try:
        while True:
            raw = await ws.receive_text()
            conn.last_pong = time.time()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await conn.send_direct({
                    "type": "error",
                    "message": "Invalid JSON",
                })
                continue

            response = await state.ws_manager.handle_viewer_message(conn, data)
            if response:
                await conn.send_direct(response)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WS viewer error [%s]: %s", conn.session_id, e)
    finally:
        drain_task.cancel()
        await state.ws_manager.disconnect(conn)


# ═══════════════════════════════════════════════════════════════════════
# WebSocket /ws — Legacy endpoint (backwards compatible)
# ═══════════════════════════════════════════════════════════════════════

@router.websocket("/ws")
async def ws_legacy(ws: WebSocket):
    """Legacy endpoint — acts as a producer (same as /ws/sensor)."""
    await ws_sensor_producer(ws)


# ═══════════════════════════════════════════════════════════════════════
# WebSocket stats endpoint
# ═══════════════════════════════════════════════════════════════════════

@router.get(
    "/ws/stats",
    summary="WebSocket connection statistics",
    tags=["WebSocket"],
)
async def ws_stats():
    """Return active WebSocket connections and broadcast metrics."""
    return {
        "stats": state.ws_manager.get_stats(),
        "connections": state.ws_manager.get_connections(),
    }


# ═══════════════════════════════════════════════════════════════════════
# Engine management endpoints
# ═══════════════════════════════════════════════════════════════════════

@router.get(
    "/engines",
    summary="List all engines with active sliding windows",
    tags=["Engine Management"],
)
async def list_engines():
    """Return all engines that have buffered data."""
    engines = await state.buffer_manager.get_all_engines()
    return {"engines": engines, "count": len(engines)}


@router.get(
    "/engines/{unit_number}",
    summary="Get buffer state for a specific engine",
    tags=["Engine Management"],
)
async def get_engine(unit_number: int):
    """Return sliding window state for a single engine."""
    engine_state = await state.buffer_manager.get_engine_state(unit_number)
    if engine_state is None:
        raise HTTPException(404, f"No data buffered for engine {unit_number}")
    return engine_state


@router.delete(
    "/engines/{unit_number}",
    summary="Clear sliding window for an engine",
    tags=["Engine Management"],
)
async def clear_engine(unit_number: int):
    """Reset the sliding window buffer for a specific engine."""
    cleared = await state.buffer_manager.clear_engine(unit_number)
    if not cleared:
        raise HTTPException(404, f"No data buffered for engine {unit_number}")
    return {"message": f"Buffer cleared for engine {unit_number}"}


@router.delete(
    "/engines",
    summary="Clear all sliding window buffers",
    tags=["Engine Management"],
)
async def clear_all_engines():
    """Reset all engine buffers."""
    count = await state.buffer_manager.clear_all()
    return {"message": f"Cleared {count} engine buffers"}


# ═══════════════════════════════════════════════════════════════════════
# Health check
# ═══════════════════════════════════════════════════════════════════════

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    tags=["System"],
)
async def health_check():
    """Return service health, model info, and uptime."""
    engines = await state.buffer_manager.get_all_engines()
    ws_stats = state.ws_manager.get_stats() if state.ws_manager else {}
    return HealthResponse(
        status="ok",
        model_loaded=state.predictor is not None,
        model_name=state.predictor.name if state.predictor else "none",
        input_shape=list(state.predictor._expected_shape) if state.predictor else [],
        feature_columns=state.feature_cols,
        active_engines=len(engines),
        uptime_seconds=round(time.time() - state.start_time, 1),
    )


# ═══════════════════════════════════════════════════════════════════════
# Feature info endpoint
# ═══════════════════════════════════════════════════════════════════════

@router.get(
    "/features",
    summary="List expected feature columns",
    tags=["System"],
)
async def get_features():
    """
    Return the ordered list of feature columns the model expects.
    Use this to construct valid `sensor_values` dictionaries.
    """
    return {
        "feature_columns": state.feature_cols,
        "count": len(state.feature_cols),
        "sequence_length": SEQUENCE_LENGTH,
        "note": (
            "sensor_values must contain all listed columns. "
            "Values should be raw (unnormalized) — the scaler is "
            "applied internally if configured."
        ),
    }
