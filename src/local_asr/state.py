from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

from .events import DebugEvent, LogEvent, MetricsEvent, SessionEvent, TranscriptEvent


@dataclass(slots=True)
class SessionState:
    state: str = "idle"
    device_name: str = "-"
    device_mode: str = "-"
    model_name: str = "-"
    samplerate: int = 0
    blocksize: int = 0
    chunk_size: tuple[int, int, int] = (0, 0, 0)
    encoder_look_back: int = 0
    decoder_look_back: int = 0
    started_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class TranscriptState:
    partial: str = ""
    stable_total: str = ""
    stable_lines: Deque[str] = field(default_factory=lambda: deque(maxlen=12))


@dataclass(slots=True)
class MetricsState:
    queue_current: int = 0
    queue_max: int = 0
    audio_chunks_received: int = 0
    audio_overflows: int = 0
    silence_flushes: int = 0
    inference_calls: int = 0
    inference_empty_results: int = 0
    final_sentences: int = 0
    avg_infer_ms: float = 0.0
    last_infer_ms: float = 0.0
    rtf_current: float = 0.0
    rtf_avg: float = 0.0
    chunk_rate: float = 0.0


@dataclass(slots=True)
class DebugState:
    raw_chunk_text: str = ""
    stable_text: str = ""
    partial_text: str = ""
    cache_summary: str = ""
    last_infer_ms: float = 0.0


@dataclass(slots=True)
class UIState:
    session: SessionState = field(default_factory=SessionState)
    transcript: TranscriptState = field(default_factory=TranscriptState)
    metrics: MetricsState = field(default_factory=MetricsState)
    debug: DebugState = field(default_factory=DebugState)
    logs: Deque[LogEvent] = field(default_factory=lambda: deque(maxlen=20))

    def apply_session(self, event: SessionEvent) -> None:
        self.session.state = event.state
        self.session.device_name = event.device_name
        self.session.device_mode = event.device_mode
        self.session.model_name = event.model_name
        self.session.samplerate = event.samplerate
        self.session.blocksize = event.blocksize
        self.session.chunk_size = event.chunk_size
        self.session.encoder_look_back = event.encoder_look_back
        self.session.decoder_look_back = event.decoder_look_back
        if event.state == "starting":
            self.session.started_at = event.ts

    def apply_transcript(self, event: TranscriptEvent) -> None:
        if event.level == "partial":
            self.transcript.partial = event.text
        elif event.level == "stable":
            stable_delta = event.text[len(self.transcript.stable_total) :] if event.text.startswith(self.transcript.stable_total) else event.text
            self.transcript.stable_total = event.text
            if stable_delta:
                self.transcript.stable_lines.append(stable_delta)
        elif event.level == "final":
            self.transcript.partial = ""
            self.transcript.stable_total = ""
            self.transcript.stable_lines.clear()

    def apply_metrics(self, event: MetricsEvent) -> None:
        self.metrics.queue_current = event.queue_current
        self.metrics.queue_max = event.queue_max
        self.metrics.audio_chunks_received = event.audio_chunks_received
        self.metrics.audio_overflows = event.audio_overflows
        self.metrics.silence_flushes = event.silence_flushes
        self.metrics.inference_calls = event.inference_calls
        self.metrics.inference_empty_results = event.inference_empty_results
        self.metrics.final_sentences = event.final_sentences
        self.metrics.avg_infer_ms = event.avg_infer_ms
        self.metrics.last_infer_ms = event.last_infer_ms
        self.metrics.rtf_current = event.rtf_current
        self.metrics.rtf_avg = event.rtf_avg
        self.metrics.chunk_rate = event.chunk_rate

    def apply_debug(self, event: DebugEvent) -> None:
        self.debug.raw_chunk_text = event.raw_chunk_text
        self.debug.stable_text = event.stable_text
        self.debug.partial_text = event.partial_text
        self.debug.cache_summary = event.cache_summary
        self.debug.last_infer_ms = event.last_infer_ms

    def apply_log(self, event: LogEvent) -> None:
        self.logs.appendleft(event)
