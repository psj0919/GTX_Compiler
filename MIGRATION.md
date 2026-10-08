# YOLOv8n, Laya, ResNet18 migration

This independent repository is `psj0919/GTX_Compiler`. Its vision submodule is
`psj0919/vision.cpp`; the original repositories and both migration backups remain intact.

The migration includes the backup's YOLO meshgrid emitter, ConvTranspose2d emitter
and weight layout handling, Laya dependency lock and converter, encoder/head graphs,
request preparation/decoding, validation tools and sample requests. The vision submodule
builds the remote YOLOv8n graph and generated ResNet18 graph into `vision-cli`, with
`yolo` and `resnet` commands and explicit `-b gtx` selection. Existing architecture
sources and the bundled llama commit remain in place.

GGUF/checkpoint files, `.venv`, build directories and generated `output/` are ignored
artifacts. They are preserved in the backups; Git does not upload these large files.

```bash
cd ~/supergate
git clone --recurse-submodules https://github.com/psj0919/GTX_Compiler.git GTX_Compiler-owned
cd GTX_Compiler-owned
python3 tools/restore_migration_artifacts.py \
  ~/supergate/gtx-migration-backup-20261008-141225
uv sync --extra laya
cd ../gtx_ggml_FPGA
git pull --ff-only
export GTX_COMPILER_ROOT="$HOME/supergate/GTX_Compiler-owned"
bash tools/build_unified.sh
source tools/gtx_unified_env.sh
```

All runtimes use `gtx_ggml_FPGA/build-unified/bin`, one GTX module and shared GGML
with `GGML_MAX_NAME=128`. Do not mix these libraries with old standalone llama builds.

## CPU commands

Unset `GGML_BACKEND_PATH` for Laya CPU execution because its runners select the best
registered device automatically. The existing Laya checkpoint revision remains pinned.

```bash
cd ~/supergate/GTX_Compiler-owned
env -u GGML_BACKEND_PATH "$GTX_UNIFIED_BIN/vision-cli" yolo \
  -m vision.cpp/models/Yolov8n-F16.gguf -i vision.cpp/docs/media/input.jpg \
  -o output/yolov8n/cpu.png -b cpu
env -u GGML_BACKEND_PATH "$GTX_UNIFIED_BIN/vision-cli" resnet \
  -m output/resnet18/ResNet.gguf -i vision.cpp/docs/media/input.jpg \
  -o output/resnet18/cpu-logits.txt -b cpu --log-ops
env -u GGML_BACKEND_PATH .venv/bin/python tools/run_laya.py \
  --model-dir output/laya-multilingual --runner-dir "$GTX_UNIFIED_BIN" \
  --batch output/laya-request/batch.json --output output/laya-request/result-cpu-new.json
```

ResNet directly resizes to RGB 224x224 and applies ImageNet mean/std once. The earlier
remote CLI mistakenly applied byte scaling twice; this is corrected. Its fixed 7x7
final average pool supports this input size. Class indexes are zero based.

## FPGA commands (user validation required)

After reboot and firmware upload, use the common module and explicit GTX selection.
Vision submits the graph to GTX without an added CPU fallback scheduler.

```bash
sudo env GGML_BACKEND_PATH="$GGML_BACKEND_PATH" LD_LIBRARY_PATH="$GTX_UNIFIED_BIN" \
  GTX_TRANSPORT=xdma GTX_XDMA_USER=/dev/xdma0_user \
  GTX_XDMA_H2C=/dev/xdma0_h2c_0 GTX_XDMA_C2H=/dev/xdma0_c2h_0 \
  GTX_KERNEL_DIR="$GTX_KERNEL_DIR" GTX_KERNEL_WORK_ADDR=0x40000 \
  GTX_KERNEL_WORK_SLOT_COUNT=28 GTX_LOG_OPS=1 VISP_FLASH_ATTENTION=0 \
  "$GTX_UNIFIED_BIN/vision-cli" resnet \
  -m output/resnet18/ResNet.gguf -i vision.cpp/docs/media/input.jpg \
  -o output/resnet18/gtx-logits.txt -b gtx
```

For YOLO, replace the executable arguments with `yolo -m vision.cpp/models/Yolov8n-F16.gguf
-i vision.cpp/docs/media/input.jpg -o output/yolov8n/gtx.png -b gtx`.

```bash
sudo env GGML_BACKEND_PATH="$GGML_BACKEND_PATH" LD_LIBRARY_PATH="$GTX_UNIFIED_BIN" \
  GTX_TRANSPORT=xdma GTX_XDMA_USER=/dev/xdma0_user \
  GTX_XDMA_H2C=/dev/xdma0_h2c_0 GTX_XDMA_C2H=/dev/xdma0_c2h_0 \
  GTX_KERNEL_DIR="$GTX_KERNEL_DIR" GTX_KERNEL_WORK_ADDR=0xc0000 \
  GTX_KERNEL_WORK_SLOT_COUNT=28 GTX_ENABLE_TILING_SCHED=1 \
  GTX_LOG_OPS=1 VISP_FLASH_ATTENTION=0 \
  "$PWD/.venv/bin/python" tools/run_laya.py \
  --model-dir output/laya-multilingual --runner-dir "$GTX_UNIFIED_BIN" \
  --batch output/laya-request/batch.json --output output/laya-request/result-gtx-new.json
```

Work addresses preserve the prior YOLO/ResNet and Laya command settings; use the
addresses required by the uploaded firmware. Do not replace firmware implicitly.

## Local validation, 2026-10-08

- Built GTX module, vision CLI, llama CLI/completion, both Laya runners.
- Registered GTX0 via `llama-cli --list-devices`, without initializing FPGA transport.
- YOLO: 705,600 CPU float values exactly match the recorded old runtime output.
- Laya: four prepared requests, all hidden tensors and logits exactly match the remote
  backup CPU baseline; decoded JSON is identical. The math sources are copied unchanged.
- ResNet: 1,000 finite logits match the corrected CPU reference, top class 756,
  logit 10.4060049. This is not a comparison against the previously incorrect preprocessing.
- Actual GTX inference and full Qwen/MobileLLM/Whisper/vision FPGA regression are
  **unverified** and will be run remotely by the user. No kernels, firmware or intrinsics changed.

Keep backups until those regressions pass. See `gtx_ggml_FPGA/docs/gtx_unified_backend.md`
for shared ABI details and model baseline commands.
