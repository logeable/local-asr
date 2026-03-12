# local-asr

基于 `uv` 的本地实时语音识别示例项目。

实现内容：

- 打开本地音频输入设备
- 从麦克风实时采集音频
- 使用 `FunASR` 在线流式模型做实时识别
- 支持列出设备、预热下载模型、实时打印流式识别结果

## 环境要求

- Python `3.11`
- macOS / Linux / Windows
- 可用的麦克风输入设备

如果你在 macOS 上运行，第一次启动时需要给终端或 Codex 桌面应用授予麦克风权限。

## 安装依赖

```bash
uv sync
```

## 查看本地音频设备

```bash
uv run local-asr devices
```

## 预热下载默认模型

第一次运行会自动下载默认 FunASR 中文流式模型。也可以先预热下载：

```bash
uv run local-asr warmup
```

默认模型是：

```bash
iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online
```

## 启动实时识别

使用默认输入设备：

```bash
uv run local-asr recognize
```

指定设备：

```bash
uv run local-asr recognize --device 0
```

指定模型：

```bash
uv run local-asr recognize --model iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online
```

隐藏中间流式输出：

```bash
uv run local-asr recognize --hide-intermediate
```

指定运行设备：

```bash
uv run local-asr recognize --device-mode mps
```

调整流式分块参数：

```bash
uv run local-asr recognize --chunk-size 0,10,5 --encoder-look-back 4 --decoder-look-back 1
```

## 命令帮助

```bash
uv run local-asr --help
uv run local-asr recognize --help
```
