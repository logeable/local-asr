from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True)
class SessionEvent:
    state: Literal["starting", "running", "stopping", "stopped", "error"]
    device_name: str
    device_mode: str
    model_name: str
    samplerate: int
    blocksize: int
    chunk_size: tuple[int, int, int]
    encoder_look_back: int
    decoder_look_back: int
    ts: float = field(default_factory=time.time)


@dataclass(slots=True)
class TranscriptEvent:
    level: Literal["partial", "stable", "final"]
    text: str
    ts: float = field(default_factory=time.time)


@dataclass(slots=True)
class MetricsEvent:
    queue_current: int
    queue_max: int
    audio_chunks_received: int
    audio_overflows: int
    inference_calls: int
    inference_empty_results: int
    final_sentences: int
    avg_infer_ms: float
    last_infer_ms: float
    rtf_current: float
    rtf_avg: float
    chunk_rate: float
    ts: float = field(default_factory=time.time)


@dataclass(slots=True)
class LogEvent:
    level: Literal["INFO", "WARN", "ERROR", "DEBUG"]
    message: str
    ts: float = field(default_factory=time.time)


@dataclass(slots=True)
class DebugEvent:
    raw_chunk_text: str
    stable_text: str
    partial_text: str
    cache_summary: str
    last_infer_ms: float
    ts: float = field(default_factory=time.time)
