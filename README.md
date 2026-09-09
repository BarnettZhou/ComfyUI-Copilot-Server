# ComfyUI Copilot Server

在 Mac 上使用 ComfyUI 源码版运行的轻量 HTTP 服务，给局域网内的主 ComfyUI（PC 端）
分担两类重活：**远程文本编码**（CLIPTextEncode / Krea2EditGroundedEncode）和
**SeedVR2 图像放大**（VAE + DiT 采样 + 颜色修复全流程，SeedVR2 无需 text encoder）。

一个进程、一个端口；`exclusive_models: true`（默认）时两类服务错峰加载，加载一类
会释放另一类的模型。

## 配置与启动

复制 `example.config.yaml` 为 `config.yaml`，填写 ComfyUI 根目录、监听地址，
以及 `encoders` / `upscalers`（至少一段）。encoder 的 `id` 必须与 Windows 端
text encoder 文件名一致；upscaler 需要 SeedVR2 的 DiT 权重和专用 VAE 文件路径。

```bash
cd ~/comfyui/ComfyUI-Copilot-Server
cp example.config.yaml config.yaml   # 按需修改
./start.sh
```

`start.sh` 默认使用 `~/comfyui/venv/bin/python3`；可通过 `COMFY_ROOT`、`PYTHON`
和 `CONFIG` 环境变量覆盖。

## 端点

文本编码（与 ComfyUI-Text-Encoder-Server 协议一致，Windows 端客户端零改动）：

- `POST /v1/ping` → `{"ok", "encoder_ids", "upscaler_ids"}`
- `POST /v1/load_clip` / `/v1/release` / `/v1/encode_text` / `/v1/encode_grounded`

SeedVR2 放大：

- `POST /v1/load_upscaler`，JSON `{"upscaler_id": "..."}` → `{"ok", "loaded"}`
- `POST /v1/release_upscaler` → `{"ok": true}`
- `POST /v1/upscale_seedvr2`，body 为 `torch.save` 字节流（`application/octet-stream`）：

```python
{
  "image": IMAGE tensor (B,H,W,C) float16 [0,1],  # 已放大到目标分辨率的输入图
  "upscaler_id": "seedvr2_7b",   # 只配了一个 upscaler 时可省略
  "seed": int, "steps": int, "cfg": float,
  "sampler_name": str, "scheduler": str,
  "denoise": float,              # 通常 1.0
  "color_fix": "lab" | "wavelet" | "adain" | "none",
  "tile": int,                   # VAE 分块大小（像素），<=0 或缺省表示整图不分块
  "overlap": int,                # 分块重叠像素，默认 64
}
```

响应 body 为 `torch.save` 的 IMAGE tensor（float16）；出错时 HTTP 400 + JSON
`{"ok": false, "error": ...}`，显存不足（OOM）时额外带 `"oom": true`，
调用方可据此改用更小的 `tile` 重试。

轻量放大（无需配置，body 同样是 `torch.save` 字节流，响应同样是 IMAGE tensor）：

- `POST /v1/upscale_resize`：`{"image", "scale": 2.0, "method": "lanczos"}`，
  method 可选 `nearest-exact / bilinear / area / bicubic / lanczos`
- `POST /v1/upscale_model`：`{"image", "scale": 2.0, "method": "lanczos",
  "model": "4x-UltraSharp.pth", "tile": 512, "overlap": 32}`；model 为
  ComfyUI `models/upscale_models/` 下的文件名（spandrel 系模型），先用模型原生倍数
  放大，再按 `method` 插值到 `scale` 倍目标尺寸；`tile <= 0` 表示不分块。
- `/v1/ping` 响应中带 `upscale_models`（可用模型列表）和 `resize_methods`。

## 本地冒烟

```bash
~/comfyui/venv/bin/python3 scripts/test_upscale.py in.png out.png --mode seedvr2 --steps 1
~/comfyui/venv/bin/python3 scripts/test_upscale.py in.png out.png --mode resize --scale 2.0
~/comfyui/venv/bin/python3 scripts/test_upscale.py in.png out.png --mode model --model 4x-UltraSharp.pth --scale 2.0
```
