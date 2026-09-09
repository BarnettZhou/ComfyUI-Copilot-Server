"""SeedVR2 图像放大服务：preprocess → VAE encode → 采样 → VAE decode → 颜色修复。"""
import time
from pathlib import Path

COLOR_FIX_METHODS = ("lab", "wavelet", "adain", "none")

class Upscaler:
    def __init__(self, config, torch, folder_paths, nodes, lock):
        self.torch, self.folder_paths, self.nodes, self.lock = torch, folder_paths, nodes, lock
        self.upscalers = config["upscalers"]
        self.current_id, self.model, self.vae = None, None, None
        self._seedvr = None
    def _nodes(self):
        if self._seedvr is None:
            import comfy_extras.nodes_seedvr
            self._seedvr = comfy_extras.nodes_seedvr
        return self._seedvr
    def release(self):
        with self.lock:
            self.model = None; self.vae = None; self.current_id = None
    def load_upscaler(self, upscaler_id):
        with self.lock:
            if upscaler_id is None and len(self.upscalers) == 1:
                upscaler_id = next(iter(self.upscalers))
            item = self.upscalers.get(upscaler_id)
            if item is None: raise ValueError(f"未知 upscaler_id: {upscaler_id}")
            if self.current_id == upscaler_id and self.model is not None:
                print(f"[copilot] 复用 upscaler: {upscaler_id}", flush=True)
                return False
            dit_path, vae_path = Path(item["dit_path"]), Path(item["vae_path"])
            if not dit_path.is_file(): raise FileNotFoundError(f"找不到 SeedVR2 DiT: {dit_path}")
            if not vae_path.is_file(): raise FileNotFoundError(f"找不到 SeedVR2 VAE: {vae_path}")
            print(f"[copilot] 加载 upscaler: {upscaler_id}", flush=True)
            started = time.perf_counter()
            self.model = None; self.vae = None
            self.folder_paths.add_model_folder_path("diffusion_models", str(dit_path.parent), is_default=True)
            self.folder_paths.add_model_folder_path("vae", str(vae_path.parent), is_default=True)
            self.model = self.nodes.UNETLoader().load_unet(dit_path.name, item["weight_dtype"])[0]
            self.vae = self.nodes.VAELoader().load_vae(vae_path.name)[0]
            self.current_id = upscaler_id
            print(f"[copilot] upscaler 就绪: {upscaler_id} ({time.perf_counter() - started:.2f}s)", flush=True)
            return True
    def upscale(self, image, params):
        with self.lock, self.torch.inference_mode():
            self.load_upscaler(params.get("upscaler_id"))
            seedvr = self._nodes()
            color_fix = params.get("color_fix", "lab")
            if color_fix not in COLOR_FIX_METHODS: raise ValueError(f"未知 color_fix: {color_fix}")
            started = time.perf_counter()
            image = image.to(self.torch.float32)
            tile, overlap = int(params.get("tile", 0) or 0), int(params.get("overlap", 64))
            if tile < overlap * 4: overlap = tile // 4
            tiled = tile > 0
            print(f"[copilot] 开始 SeedVR2 放大: {tuple(image.shape)} steps={params['steps']} cfg={params['cfg']} tile={tile or '整图'}", flush=True)
            pre = seedvr.SeedVR2Preprocess.execute(image).result[0]
            if tiled:
                latent = {"samples": self.vae.encode_tiled(pre, tile_x=tile, tile_y=tile, overlap=overlap)}
            else:
                latent = self.nodes.VAEEncode().encode(self.vae, pre)[0]
            positive, negative = seedvr.SeedVR2Conditioning.execute(self.model, latent).result
            out = self.nodes.common_ksampler(self.model, params["seed"], params["steps"], params["cfg"], params["sampler_name"], params["scheduler"], positive, negative, latent, denoise=params.get("denoise", 1.0))[0]
            if tiled:
                compression = self.vae.spacial_compression_decode()
                decoded = self.vae.decode_tiled(out["samples"], tile_x=tile // compression, tile_y=tile // compression, overlap=overlap // compression)
                if len(decoded.shape) == 5:
                    decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
            else:
                decoded = self.nodes.VAEDecode().decode(self.vae, out)[0]
            result = seedvr.SeedVR2PostProcessing.execute(decoded, image, color_fix).result[0]
            print(f"[copilot] SeedVR2 放大完成: {tuple(result.shape)} ({time.perf_counter() - started:.2f}s)", flush=True)
            return result
