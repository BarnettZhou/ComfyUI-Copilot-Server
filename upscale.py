"""轻量图像放大服务：普通 resize 和 upscale model（spandrel 系，如 4x-UltraSharp）。"""
import time

METHODS = ("nearest-exact", "bilinear", "area", "bicubic", "lanczos")

class ImageUpscaler:
    def __init__(self, torch, folder_paths, nodes, model_management, lock):
        self.torch, self.folder_paths, self.nodes, self.lock = torch, folder_paths, nodes, lock
        self.model_management = model_management
        self.model_name, self.model = None, None
    def list_models(self):
        return self.folder_paths.get_filename_list("upscale_models")
    def _check_method(self, method):
        if method not in METHODS: raise ValueError(f"未知插值方式: {method}，可选: {', '.join(METHODS)}")
    def resize(self, image, scale, method):
        self._check_method(method)
        started = time.perf_counter()
        result = self.nodes.ImageScaleBy().upscale(image.to(self.torch.float32), method, float(scale))[0]
        print(f"[copilot] resize 放大: {tuple(image.shape)} x{scale} {method} -> {tuple(result.shape)} ({time.perf_counter() - started:.2f}s)", flush=True)
        return result
    def _load_model(self, model_name):
        if self.model is not None and self.model_name == model_name:
            print(f"[copilot] 复用 upscale model: {model_name}", flush=True)
            return self.model
        import comfy_extras.nodes_upscale_model
        started = time.perf_counter()
        print(f"[copilot] 加载 upscale model: {model_name}", flush=True)
        self.model = comfy_extras.nodes_upscale_model.UpscaleModelLoader.execute(model_name).result[0]
        self.model_name = model_name
        print(f"[copilot] upscale model 就绪: {model_name} scale={self.model.scale}x ({time.perf_counter() - started:.2f}s)", flush=True)
        return self.model
    def model_upscale(self, image, scale, method, model_name, tile, overlap):
        self._check_method(method)
        import comfy.utils
        with self.lock, self.torch.inference_mode():
            model = self._load_model(model_name)
            started = time.perf_counter()
            image = image.to(self.torch.float32)
            memory_required = (512 * 512 * 3) * image.element_size() * max(model.scale, 1.0) * 384.0 + image.nelement() * image.element_size()
            self.model_management.load_models_gpu([model.patcher], memory_required=memory_required, force_full_load=True)
            in_img = image.movedim(-1, -3).to(model.patcher.load_device)
            tile = int(tile) if tile and int(tile) > 0 else max(in_img.shape[2], in_img.shape[3])
            while True:
                try:
                    out = comfy.utils.tiled_scale(in_img, lambda a: model(a.float()), tile_x=tile, tile_y=tile, overlap=int(overlap), upscale_amount=model.scale, output_device=self.model_management.intermediate_device())
                    break
                except Exception as exc:
                    self.model_management.raise_non_oom(exc)
                    tile //= 2
                    if tile < 128: raise
                    print(f"[copilot] upscale model OOM，tile 减半到 {tile}", flush=True)
            result = self.torch.clamp(out.movedim(-3, -1), min=0, max=1.0)
            target_w, target_h = round(image.shape[2] * float(scale)), round(image.shape[1] * float(scale))
            if (result.shape[2], result.shape[1]) != (target_w, target_h):
                result = comfy.utils.common_upscale(result.movedim(-1, 1), target_w, target_h, method, "disabled").movedim(1, -1)
            print(f"[copilot] upscale model 放大: {model_name} tile={tile} overlap={overlap} -> {tuple(result.shape)} ({time.perf_counter() - started:.2f}s)", flush=True)
            return result
