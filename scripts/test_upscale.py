"""本地冒烟脚本：把一张图发给 /v1/upscale_seedvr2 并保存结果。

用法（用 ComfyUI venv 的 python 运行）:
    ~/comfyui/venv/bin/python3 scripts/test_upscale.py 输入图 [输出图] [--server 127.0.0.1:50051] [--steps 4]
"""
import argparse, io, urllib.request
from pathlib import Path
import torch
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input"); parser.add_argument("output", nargs="?", default="upscaled.png")
    parser.add_argument("--server", default="127.0.0.1:50051")
    parser.add_argument("--upscaler-id", default=None)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--cfg", type=float, default=1.0); parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="simple"); parser.add_argument("--denoise", type=float, default=1.0)
    parser.add_argument("--color-fix", default="lab", choices=["lab", "wavelet", "adain", "none"])
    args = parser.parse_args()

    image = Image.open(args.input).convert("RGB")
    tensor = torch.from_numpy(__import__("numpy").array(image)).float().div(255).unsqueeze(0).half()
    payload = {"image": tensor, "upscaler_id": args.upscaler_id, "seed": args.seed, "steps": args.steps,
               "cfg": args.cfg, "sampler_name": args.sampler, "scheduler": args.scheduler,
               "denoise": args.denoise, "color_fix": args.color_fix}
    buf = io.BytesIO(); torch.save(payload, buf)
    request = urllib.request.Request(f"http://{args.server}/v1/upscale_seedvr2", data=buf.getvalue(), headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(request) as response:
        result = torch.load(io.BytesIO(response.read()), map_location="cpu", weights_only=True)
    array = result[0].float().mul(255).round().clamp(0, 255).byte().numpy()
    Image.fromarray(array).save(args.output)
    print(f"已保存: {Path(args.output).resolve()} {tuple(result.shape)}")

if __name__ == "__main__": main()
