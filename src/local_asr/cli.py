from __future__ import annotations

import argparse
import queue
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
    warmup_parser.set_defaults(handler=handle_warmup)

    recognize_parser = subparsers.add_parser("recognize", help="Run real-time recognition from the microphone.")
    add_model_arguments(recognize_parser)
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
        help="FunASR online chunk size, formatted as a,b,c. Default: 0,10,5.",
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
        help="Do not print intermediate streaming text chunks.",
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
    recognize_parser.set_defaults(handler=handle_recognize)

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


def handle_recognize(args: argparse.Namespace) -> None:
    import numpy as np
    import sounddevice as sd

    model = build_asr_model(args)
    device = parse_device(args.device)
    chunk_size = parse_chunk_size(args.chunk_size)
    chunk_stride = chunk_size[1] * 960
    if args.samplerate != 16000:
        raise ValueError("The default streaming model expects 16kHz audio. Use --samplerate 16000.")

    audio_queue: queue.Queue[AudioChunk] = queue.Queue()

    def callback(indata: bytes, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        del frames, time_info
        audio_queue.put(AudioChunk(bytes(indata), bool(status.input_overflow)))

    print(f"Using FunASR model: {args.model}")
    print(f"Chunk size: {list(chunk_size)} | stride_samples={chunk_stride} | device={args.device_mode}")
    print("Start speaking. Press Ctrl+C to stop.")

    try:
        with sd.RawInputStream(
            samplerate=args.samplerate,
            blocksize=args.blocksize,
            device=device,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            render_stream(
                model,
                audio_queue,
                chunk_stride=chunk_stride,
                chunk_size=chunk_size,
                encoder_look_back=args.encoder_look_back,
                decoder_look_back=args.decoder_look_back,
                show_intermediate=not args.hide_intermediate,
                np_module=np,
            )
    except KeyboardInterrupt:
        print("\nStopped.")


def render_stream(
    model: Any,
    audio_queue: "queue.Queue[AudioChunk]",
    *,
    chunk_stride: int,
    chunk_size: tuple[int, int, int],
    encoder_look_back: int,
    decoder_look_back: int,
    show_intermediate: bool,
    np_module: Any,
) -> None:
    cache: dict[str, Any] = {}
    pending = bytearray()
    last_text = ""
    try:
        while True:
            chunk = audio_queue.get()
            if chunk.overflowed:
                print("\n[warn] input overflow detected; consider increasing --blocksize.")
            pending.extend(chunk.data)
            step_bytes = chunk_stride * 2
            while len(pending) >= step_bytes:
                current = bytes(pending[:step_bytes])
                del pending[:step_bytes]
                last_text = emit_stream_result(
                    model=model,
                    pcm_bytes=current,
                    cache=cache,
                    chunk_size=chunk_size,
                    encoder_look_back=encoder_look_back,
                    decoder_look_back=decoder_look_back,
                    is_final=False,
                    show_intermediate=show_intermediate,
                    last_text=last_text,
                    np_module=np_module,
                )
    except KeyboardInterrupt:
        flush_remaining_audio(
            model=model,
            pending=pending,
            cache=cache,
            chunk_stride=chunk_stride,
            chunk_size=chunk_size,
            encoder_look_back=encoder_look_back,
            decoder_look_back=decoder_look_back,
            last_text=last_text,
            np_module=np_module,
        )
        raise


def emit_stream_result(
    *,
    model: Any,
    pcm_bytes: bytes,
    cache: dict[str, Any],
    chunk_size: tuple[int, int, int],
    encoder_look_back: int,
    decoder_look_back: int,
    is_final: bool,
    show_intermediate: bool,
    last_text: str,
    np_module: Any,
) -> str:
    audio = np_module.frombuffer(pcm_bytes, dtype=np_module.int16).astype(np_module.float32) / 32768.0
    results = model.generate(
        input=audio,
        cache=cache,
        is_final=is_final,
        chunk_size=list(chunk_size),
        encoder_chunk_look_back=encoder_look_back,
        decoder_chunk_look_back=decoder_look_back,
    )
    if not isinstance(results, list):
        return last_text
    for result in results:
        text = str(result.get("text", "")).strip()
        if not text or text == last_text:
            continue
        tag = "final" if is_final else "stream"
        if show_intermediate or is_final:
            print(f"[{tag}] {text}")
        last_text = text
    return last_text


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


def flush_remaining_audio(
    *,
    model: Any,
    pending: bytearray,
    cache: dict[str, Any],
    chunk_stride: int,
    chunk_size: tuple[int, int, int],
    encoder_look_back: int,
    decoder_look_back: int,
    last_text: str,
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
        chunk_size=chunk_size,
        encoder_look_back=encoder_look_back,
        decoder_look_back=decoder_look_back,
        is_final=True,
        show_intermediate=True,
        last_text=last_text,
        np_module=np_module,
    )
