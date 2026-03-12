from __future__ import annotations

import argparse
from pathlib import Path
import queue
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .aggregate import TranscriptAggregator
from .events import DebugEvent, LogEvent, MetricsEvent, SessionEvent, TranscriptEvent
from .tui import RuntimeEvent, TUIRunner

DEFAULT_ASR_MODEL = "paraformer-zh-streaming"
DEFAULT_ENCODER_LOOK_BACK = 6
DEFAULT_DECODER_LOOK_BACK = 2

if TYPE_CHECKING:
    import numpy as np
    import sounddevice as sd
    from funasr import AutoModel


@dataclass(slots=True)
class AudioChunk:
    data: bytes
    overflowed: bool = False


@dataclass(slots=True)
class RuntimeStats:
    started_at: float
    queue_max: int = 0
    audio_chunks_received: int = 0
    audio_overflows: int = 0
    inference_calls: int = 0
    inference_empty_results: int = 0
    final_sentences: int = 0
    total_infer_ms: float = 0.0
    last_infer_ms: float = 0.0
    last_rtf: float = 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-asr",
        description="Open a local audio input device and run offline real-time ASR.",
    )
    subparsers = parser.add_subparsers(dest="command")

    devices_parser = subparsers.add_parser("devices", help="List local audio input devices.")
    devices_parser.set_defaults(handler=handle_devices)

    warmup_parser = subparsers.add_parser(
        "warmup",
        help="Download the default FunASR model into the local cache without opening the microphone.",
    )
    add_model_arguments(warmup_parser)
    add_punctuation_arguments(warmup_parser)
    warmup_parser.set_defaults(handler=handle_warmup)

    recognize_parser = subparsers.add_parser("recognize", help="Run real-time recognition from the microphone.")
    add_model_arguments(recognize_parser)
    add_punctuation_arguments(recognize_parser)
    recognize_parser.add_argument(
        "--device",
        default=None,
        help="Input device index or exact name. Omit to use the default input device.",
    )
    recognize_parser.add_argument("--samplerate", type=int, default=16000, help="Input sample rate.")
    recognize_parser.add_argument(
        "--blocksize",
        type=int,
        default=960,
        help="Frames per audio callback. Defaults to 60ms at 16kHz.",
    )
    recognize_parser.add_argument(
        "--chunk-size",
        default="0,12,6",
        help="FunASR online chunk size, formatted as a,b,c. Default: 0,12,6.",
    )
    recognize_parser.add_argument(
        "--encoder-look-back",
        type=int,
        default=DEFAULT_ENCODER_LOOK_BACK,
        help="Encoder look-back chunks for the streaming model.",
    )
    recognize_parser.add_argument(
        "--decoder-look-back",
        type=int,
        default=DEFAULT_DECODER_LOOK_BACK,
        help="Decoder look-back chunks for the streaming model.",
    )
    recognize_parser.add_argument(
        "--hide-intermediate",
        action="store_true",
        help="Do not print intermediate streaming text chunks in plain mode.",
    )
    recognize_parser.add_argument(
        "--device-mode",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Torch device used by FunASR. Default picks mps/cuda/cpu automatically.",
    )
    recognize_parser.add_argument(
        "--ncpu",
        type=int,
        default=4,
        help="CPU thread count forwarded to FunASR.",
    )
    recognize_parser.add_argument(
        "--disable-update-check",
        action="store_true",
        help="Disable FunASR version update checks.",
    )
    recognize_parser.add_argument(
        "--ui",
        choices=["plain", "tui"],
        default="tui",
        help="Output mode. Use `tui` for the multi-panel terminal view.",
    )
    recognize_parser.add_argument(
        "--final-output",
        type=Path,
        default=None,
        help="Optional file path. If set, final transcript lines are appended there.",
    )
    recognize_parser.set_defaults(handler=handle_recognize)

    punctuate_parser = subparsers.add_parser("punctuate", help="Run punctuation restoration on text.")
    add_punctuation_arguments(punctuate_parser, default_model="ct-punc")
    punctuate_parser.add_argument("--text", default=None, help="Input text to punctuate.")
    punctuate_parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Optional input text file. If set, file content is punctuated instead of --text.",
    )
    punctuate_parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional output file. If set, punctuated text is written there.",
    )
    punctuate_parser.set_defaults(handler=handle_punctuate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "command", None):
        args = parser.parse_args(["recognize", *sys.argv[1:]])
    args.handler(args)


def handle_devices(_: argparse.Namespace) -> None:
    import sounddevice as sd

    devices = sd.query_devices()
    default_input, _ = sd.default.device
    print("Available input devices:")
    for index, device in enumerate(devices):
        max_input = int(device.get("max_input_channels", 0))
        if max_input <= 0:
            continue
        marker = "*" if index == default_input else " "
        name = str(device.get("name", "unknown"))
        samplerate = int(device.get("default_samplerate", 0))
        print(f"{marker} [{index}] {name} | input_channels={max_input} | default_samplerate={samplerate}")


def handle_warmup(args: argparse.Namespace) -> None:
    model = build_asr_model(args)
    print(f"FunASR model initialized: {model.kwargs.get('model')}")
    if getattr(args, "punc_model", None):
        punc_model = build_punc_model(args)
        print(f"Punctuation model initialized: {punc_model.kwargs.get('model')}")


def handle_punctuate(args: argparse.Namespace) -> None:
    source_text = read_punctuation_input(args)
    model = build_punc_model(args)
    punctuated_text = punctuate_text(model, source_text)
    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(punctuated_text + "\n", encoding="utf-8")
    print(punctuated_text)


def handle_recognize(args: argparse.Namespace) -> None:
    import numpy as np
    import sounddevice as sd

    model = build_asr_model(args)
    punc_model = build_punc_model(args) if args.punc_model else None
    device = parse_device(args.device)
    device_name = resolve_device_name(device)
    resolved_device_mode = detect_torch_device(args.device_mode)
    chunk_size = parse_chunk_size(args.chunk_size)
    chunk_stride = chunk_size[1] * 960
    if args.samplerate != 16000:
        raise ValueError("The default streaming model expects 16kHz audio. Use --samplerate 16000.")

    audio_queue: queue.Queue[AudioChunk] = queue.Queue()
    event_queue: "queue.Queue[RuntimeEvent]" = queue.Queue()
    stats = RuntimeStats(started_at=time.time())
    aggregator = TranscriptAggregator()
    tui_runner = TUIRunner(event_queue) if args.ui == "tui" else None

    def callback(indata: bytes, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        del frames, time_info
        stats.audio_chunks_received += 1
        if status.input_overflow:
            stats.audio_overflows += 1
        audio_queue.put(AudioChunk(bytes(indata), bool(status.input_overflow)))
        publish_metrics(event_queue, stats, audio_queue.qsize())

    if tui_runner is not None:
        tui_runner.start()
    else:
        print(f"Using FunASR model: {args.model}")
        print(f"Chunk size: {list(chunk_size)} | stride_samples={chunk_stride} | device={resolved_device_mode}")
        print("Start speaking. Press Ctrl+C to stop.")

    publish(
        event_queue,
        SessionEvent(
            state="starting",
            device_name=device_name,
            device_mode=resolved_device_mode,
            model_name=args.model,
            samplerate=args.samplerate,
            blocksize=args.blocksize,
            chunk_size=chunk_size,
            encoder_look_back=args.encoder_look_back,
            decoder_look_back=args.decoder_look_back,
        ),
    )
    publish(event_queue, LogEvent(level="INFO", message="Microphone stream initializing"))

    try:
        with sd.RawInputStream(
            samplerate=args.samplerate,
            blocksize=args.blocksize,
            device=device,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            publish(
                event_queue,
                SessionEvent(
                    state="running",
                    device_name=device_name,
                    device_mode=resolved_device_mode,
                    model_name=args.model,
                    samplerate=args.samplerate,
                    blocksize=args.blocksize,
                    chunk_size=chunk_size,
                    encoder_look_back=args.encoder_look_back,
                    decoder_look_back=args.decoder_look_back,
                ),
            )
            publish(event_queue, LogEvent(level="INFO", message="Microphone stream running"))
            render_stream(
                model,
                audio_queue,
                event_queue=event_queue,
                stats=stats,
                aggregator=aggregator,
                chunk_stride=chunk_stride,
                chunk_size=chunk_size,
                encoder_look_back=args.encoder_look_back,
                decoder_look_back=args.decoder_look_back,
                final_output=args.final_output,
                punc_model=punc_model,
                show_intermediate=not args.hide_intermediate,
                ui_mode=args.ui,
                np_module=np,
            )
    except KeyboardInterrupt:
        publish(event_queue, LogEvent(level="INFO", message="Stopped by user"))
    finally:
        publish(
            event_queue,
            SessionEvent(
                state="stopped",
                device_name=device_name,
                device_mode=resolved_device_mode,
                model_name=args.model,
                samplerate=args.samplerate,
                blocksize=args.blocksize,
                chunk_size=chunk_size,
                encoder_look_back=args.encoder_look_back,
                decoder_look_back=args.decoder_look_back,
            ),
        )
        if tui_runner is not None:
            time.sleep(0.2)
            tui_runner.stop()
        else:
            print("\nStopped.")


def render_stream(
    model: Any,
    audio_queue: "queue.Queue[AudioChunk]",
    *,
    event_queue: "queue.Queue[RuntimeEvent]",
    stats: RuntimeStats,
    aggregator: TranscriptAggregator,
    chunk_stride: int,
    chunk_size: tuple[int, int, int],
    encoder_look_back: int,
    decoder_look_back: int,
    final_output: Path | None,
    show_intermediate: bool,
    ui_mode: str,
    np_module: Any,
) -> None:
    cache: dict[str, Any] = {}
    pending = bytearray()
    try:
        while True:
            chunk = audio_queue.get()
            stats.queue_max = max(stats.queue_max, audio_queue.qsize())
            if chunk.overflowed:
                message = "Input overflow detected; consider increasing --blocksize"
                publish(event_queue, LogEvent(level="WARN", message=message))
                if ui_mode == "plain":
                    print(f"\n[warn] {message}.")
            pending.extend(chunk.data)
            step_bytes = chunk_stride * 2
            while len(pending) >= step_bytes:
                current = bytes(pending[:step_bytes])
                del pending[:step_bytes]
                emit_stream_result(
                    model=model,
                    pcm_bytes=current,
                    cache=cache,
                    event_queue=event_queue,
                    stats=stats,
                    aggregator=aggregator,
                    audio_queue_size=audio_queue.qsize(),
                    chunk_size=chunk_size,
                    encoder_look_back=encoder_look_back,
                decoder_look_back=decoder_look_back,
                is_final=False,
                final_output=final_output,
                punc_model=punc_model,
                show_intermediate=show_intermediate,
                ui_mode=ui_mode,
                np_module=np_module,
                )
    except KeyboardInterrupt:
        flush_remaining_audio(
            model=model,
            pending=pending,
            cache=cache,
            event_queue=event_queue,
            stats=stats,
            aggregator=aggregator,
            chunk_stride=chunk_stride,
            chunk_size=chunk_size,
            encoder_look_back=encoder_look_back,
            decoder_look_back=decoder_look_back,
            final_output=final_output,
            punc_model=punc_model,
            ui_mode=ui_mode,
            np_module=np_module,
        )
        raise


def emit_stream_result(
    *,
    model: Any,
    pcm_bytes: bytes,
    cache: dict[str, Any],
    event_queue: "queue.Queue[RuntimeEvent]",
    stats: RuntimeStats,
    aggregator: TranscriptAggregator,
    audio_queue_size: int,
    chunk_size: tuple[int, int, int],
    encoder_look_back: int,
    decoder_look_back: int,
    is_final: bool,
    final_output: Path | None,
    punc_model: Any | None,
    show_intermediate: bool,
    ui_mode: str,
    np_module: Any,
) -> None:
    audio = np_module.frombuffer(pcm_bytes, dtype=np_module.int16).astype(np_module.float32) / 32768.0
    start = time.perf_counter()
    results = model.generate(
        input=audio,
        cache=cache,
        is_final=is_final,
        chunk_size=list(chunk_size),
        encoder_chunk_look_back=encoder_look_back,
        decoder_chunk_look_back=decoder_look_back,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    stats.inference_calls += 1
    stats.last_infer_ms = elapsed_ms
    stats.total_infer_ms += elapsed_ms
    audio_seconds = len(audio) / 16000.0
    stats.last_rtf = (elapsed_ms / 1000.0) / audio_seconds if audio_seconds > 0 else 0.0

    if not isinstance(results, list):
        stats.inference_empty_results += 1
        for event in aggregator.feed("", is_final_chunk=is_final):
            handle_transcript_event(event_queue, stats, event, ui_mode, show_intermediate, final_output, punc_model)
        publish_metrics(event_queue, stats, audio_queue_size)
        return

    emitted_text = False
    for result in results:
        text = str(result.get("text", "")).strip()
        if not text:
            continue
        emitted_text = True
        for event in aggregator.feed(text, is_final_chunk=is_final):
            handle_transcript_event(event_queue, stats, event, ui_mode, show_intermediate, final_output, punc_model)

    if not emitted_text:
        stats.inference_empty_results += 1
        for event in aggregator.feed("", is_final_chunk=is_final):
            handle_transcript_event(event_queue, stats, event, ui_mode, show_intermediate, final_output, punc_model)

    publish(
        event_queue,
        DebugEvent(
            raw_chunk_text=aggregator.last_text,
            stable_text=aggregator.stable_text,
            partial_text=aggregator.partial_text,
            cache_summary=summarize_cache(cache),
            last_infer_ms=elapsed_ms,
        ),
    )
    publish_metrics(event_queue, stats, audio_queue_size)


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=DEFAULT_ASR_MODEL,
        help="FunASR streaming ASR model name or local path.",
    )
    parser.add_argument(
        "--model-revision",
        default="master",
        help="Model revision passed to FunASR.",
    )


def add_punctuation_arguments(parser: argparse.ArgumentParser, *, default_model: str | None = None) -> None:
    parser.add_argument(
        "--punc-model",
        default=default_model,
        help="Optional punctuation restoration model, for example `ct-punc`.",
    )
    parser.add_argument(
        "--punc-model-revision",
        default="master",
        help="Punctuation model revision passed to FunASR.",
    )


def build_asr_model(args: argparse.Namespace) -> Any:
    from funasr import AutoModel

    return AutoModel(
        model=args.model,
        model_revision=args.model_revision,
        device=detect_torch_device(getattr(args, "device_mode", "auto")),
        disable_update=getattr(args, "disable_update_check", True),
        disable_pbar=True,
        disable_log=True,
        ncpu=getattr(args, "ncpu", 4),
    )


def build_punc_model(args: argparse.Namespace) -> Any:
    from funasr import AutoModel

    if not getattr(args, "punc_model", None):
        raise ValueError("No punctuation model configured.")
    return AutoModel(
        model=args.punc_model,
        model_revision=args.punc_model_revision,
        device=detect_torch_device(getattr(args, "device_mode", "auto")),
        disable_update=getattr(args, "disable_update_check", True),
        disable_pbar=True,
        disable_log=True,
        ncpu=getattr(args, "ncpu", 4),
    )


def detect_torch_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_chunk_size(raw_value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != 3:
        raise ValueError("--chunk-size must contain exactly three comma-separated integers.")
    return tuple(int(part) for part in parts)


def parse_device(raw_device: str | None) -> int | str | None:
    if raw_device is None:
        return None
    try:
        return int(raw_device)
    except ValueError:
        return raw_device


def resolve_device_name(device: int | str | None) -> str:
    import sounddevice as sd

    if device is None:
        default_input, _ = sd.default.device
        device = default_input
    try:
        info = sd.query_devices(device)
        return str(info.get("name", device))
    except Exception:
        return str(device)


def flush_remaining_audio(
    *,
    model: Any,
    pending: bytearray,
    cache: dict[str, Any],
    event_queue: "queue.Queue[RuntimeEvent]",
    stats: RuntimeStats,
    aggregator: TranscriptAggregator,
    chunk_stride: int,
    chunk_size: tuple[int, int, int],
    encoder_look_back: int,
    decoder_look_back: int,
    final_output: Path | None,
    punc_model: Any | None,
    ui_mode: str,
    np_module: Any,
) -> None:
    if not pending:
        return
    step_bytes = chunk_stride * 2
    padded = bytes(pending) + b"\x00" * max(0, step_bytes - len(pending))
    emit_stream_result(
        model=model,
        pcm_bytes=padded[:step_bytes],
        cache=cache,
        event_queue=event_queue,
        stats=stats,
        aggregator=aggregator,
        audio_queue_size=0,
        chunk_size=chunk_size,
        encoder_look_back=encoder_look_back,
        decoder_look_back=decoder_look_back,
        is_final=True,
        final_output=final_output,
        punc_model=punc_model,
        show_intermediate=True,
        ui_mode=ui_mode,
        np_module=np_module,
    )


def publish(event_queue: "queue.Queue[RuntimeEvent]", event: RuntimeEvent) -> None:
    try:
        event_queue.put_nowait(event)
    except queue.Full:
        pass


def publish_metrics(event_queue: "queue.Queue[RuntimeEvent]", stats: RuntimeStats, queue_current: int) -> None:
    now = time.time()
    elapsed = max(now - stats.started_at, 1e-6)
    publish(
        event_queue,
        MetricsEvent(
            queue_current=queue_current,
            queue_max=max(stats.queue_max, queue_current),
            audio_chunks_received=stats.audio_chunks_received,
            audio_overflows=stats.audio_overflows,
            inference_calls=stats.inference_calls,
            inference_empty_results=stats.inference_empty_results,
            final_sentences=stats.final_sentences,
            avg_infer_ms=stats.total_infer_ms / stats.inference_calls if stats.inference_calls else 0.0,
            last_infer_ms=stats.last_infer_ms,
            rtf_current=stats.last_rtf,
            rtf_avg=((stats.total_infer_ms / 1000.0) / elapsed),
            chunk_rate=stats.audio_chunks_received / elapsed,
        ),
    )


def handle_transcript_event(
    event_queue: "queue.Queue[RuntimeEvent]",
    stats: RuntimeStats,
    event: TranscriptEvent,
    ui_mode: str,
    show_intermediate: bool,
    final_output: Path | None,
    punc_model: Any | None,
) -> None:
    publish(event_queue, event)
    if event.level == "final":
        stats.final_sentences += 1
        final_text = punctuate_text(punc_model, event.text) if punc_model is not None else event.text
        if final_output is not None:
            append_final_output(final_output, final_text)
        if ui_mode == "plain":
            print(f"[final] {final_text}")
    elif event.level == "partial" and show_intermediate and ui_mode == "plain":
        print(f"[partial] {event.text}")


def summarize_cache(cache: dict[str, Any]) -> str:
    if not cache:
        return "-"
    parts = []
    for key, value in sorted(cache.items()):
        if isinstance(value, dict):
            parts.append(f"{key}={len(value)}")
        elif isinstance(value, (list, tuple)):
            parts.append(f"{key}[{len(value)}]")
        elif hasattr(value, "shape"):
            parts.append(f"{key}{tuple(value.shape)}")
        else:
            parts.append(f"{key}={type(value).__name__}")
    return ", ".join(parts)


def append_final_output(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def punctuate_text(model: Any | None, text: str) -> str:
    if model is None:
        return text
    results = model.generate(input=text)
    if not isinstance(results, list) or not results:
        return text
    punctuated = str(results[0].get("text", "")).strip()
    return punctuated or text


def read_punctuation_input(args: argparse.Namespace) -> str:
    if args.input_file is not None:
        return args.input_file.read_text(encoding="utf-8").strip()
    if args.text:
        return args.text.strip()
    raise ValueError("Provide either --text or --input-file for `punctuate`.")
