from __future__ import annotations

import queue
import threading
import time

from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .events import DebugEvent, LogEvent, MetricsEvent, SessionEvent, TranscriptEvent
from .state import UIState

RuntimeEvent = SessionEvent | TranscriptEvent | MetricsEvent | LogEvent | DebugEvent


class TUIRunner:
    def __init__(self, event_queue: "queue.Queue[RuntimeEvent]") -> None:
        self.event_queue = event_queue
        self.state = UIState()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="local-asr-tui", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)

    def _run(self) -> None:
        layout = build_layout()
        with Live(layout, refresh_per_second=8, screen=True) as live:
            while not self.stop_event.is_set():
                self._drain_events()
                render_layout(layout, self.state)
                live.refresh()
                time.sleep(0.125)
            self._drain_events()
            render_layout(layout, self.state)
            live.refresh()

    def _drain_events(self) -> None:
        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(event, SessionEvent):
                self.state.apply_session(event)
            elif isinstance(event, TranscriptEvent):
                self.state.apply_transcript(event)
            elif isinstance(event, MetricsEvent):
                self.state.apply_metrics(event)
            elif isinstance(event, DebugEvent):
                self.state.apply_debug(event)
            elif isinstance(event, LogEvent):
                self.state.apply_log(event)


def build_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="status", size=3),
        Layout(name="body", ratio=3),
        Layout(name="bottom", ratio=2),
    )
    layout["body"].split_row(
        Layout(name="stable", ratio=2),
        Layout(name="partial", ratio=1),
        Layout(name="metrics", ratio=1),
    )
    layout["bottom"].split_row(Layout(name="logs", ratio=2), Layout(name="debug", ratio=1))
    return layout


def render_layout(layout: Layout, state: UIState) -> None:
    layout["status"].update(render_status(state))
    layout["stable"].update(render_stable(state))
    layout["partial"].update(render_partial(state))
    layout["metrics"].update(render_metrics(state))
    layout["logs"].update(render_logs(state))
    layout["debug"].update(render_debug(state))


def render_status(state: UIState) -> Panel:
    uptime = max(time.time() - state.session.started_at, 0.0)
    status = Text()
    status.append(f"state={state.session.state}  ", style="bold green")
    status.append(f"device={state.session.device_name}  ")
    status.append(f"backend={state.session.device_mode}  ")
    status.append(f"model={state.session.model_name}  ")
    status.append(f"sr={state.session.samplerate}  ")
    status.append(f"block={state.session.blocksize}  ")
    status.append(f"chunk={','.join(map(str, state.session.chunk_size))}  ")
    status.append(f"uptime={format_duration(uptime)}")
    return Panel(status, title="Status")


def render_stable(state: UIState) -> Panel:
    stable_lines = list(state.transcript.stable_lines)
    return Panel("\n".join(stable_lines) if stable_lines else "-", title="Stable")


def render_partial(state: UIState) -> Panel:
    return Panel(state.transcript.partial or "-", title="Partial")


def render_metrics(state: UIState) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="cyan")
    table.add_column(justify="right")
    metrics = state.metrics
    for key, value in [
        ("queue", f"{metrics.queue_current}/{metrics.queue_max}"),
        ("audio_chunks", str(metrics.audio_chunks_received)),
        ("overflows", str(metrics.audio_overflows)),
        ("silence_flushes", str(metrics.silence_flushes)),
        ("infer_calls", str(metrics.inference_calls)),
        ("empty_results", str(metrics.inference_empty_results)),
        ("final_sentences", str(metrics.final_sentences)),
        ("last_infer_ms", f"{metrics.last_infer_ms:.1f}"),
        ("avg_infer_ms", f"{metrics.avg_infer_ms:.1f}"),
        ("rtf_current", f"{metrics.rtf_current:.2f}"),
        ("rtf_avg", f"{metrics.rtf_avg:.2f}"),
        ("chunk_rate", f"{metrics.chunk_rate:.2f}/s"),
    ]:
        table.add_row(key, value)
    return Panel(table, title="Metrics")


def render_logs(state: UIState) -> Panel:
    if not state.logs:
        return Panel("-", title="Logs")
    lines = []
    for event in list(state.logs)[:10]:
        ts = time.strftime("%H:%M:%S", time.localtime(event.ts))
        lines.append(Text.assemble((f"[{ts}] ", "dim"), (f"{event.level} ", level_style(event.level)), event.message))
    return Panel(Group(*lines), title="Logs")


def render_debug(state: UIState) -> Panel:
    debug = state.debug
    content = Group(
        Text.assemble(("Raw chunk\n", "bold magenta"), (debug.raw_chunk_text or "-", "white")),
        Text.assemble(("\nStable cache\n", "bold magenta"), (debug.stable_text or "-", "white")),
        Text.assemble(("\nPartial cache\n", "bold magenta"), (debug.partial_text or "-", "white")),
        Text.assemble(("\nCache summary\n", "bold magenta"), (debug.cache_summary or "-", "white")),
    )
    return Panel(content, title="Debug")


def level_style(level: str) -> str:
    return {
        "INFO": "green",
        "WARN": "yellow",
        "ERROR": "red",
        "DEBUG": "cyan",
    }.get(level, "white")


def format_duration(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
