"""ComfyUI 副驾驶服务：远程 text encoder + SeedVR2 图像放大，运行在 Mac 源码版 ComfyUI Python 中。"""
import argparse, base64, io, json, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import yaml

def unpack(value, torch):
    return torch.load(io.BytesIO(base64.b64decode(value)), map_location="cpu", weights_only=True)
def pack(value, torch):
    out = io.BytesIO(); torch.save(value, out)
    return base64.b64encode(out.getvalue()).decode("ascii")

def _load_named(items, required_keys, optional_keys, section):
    if isinstance(items, dict): items = [{"id": key, **(value or {})} for key, value in items.items()]
    entries = {}
    for item in items or []:
        if not isinstance(item, dict) or not item.get("id") or any(not item.get(key) for key in required_keys):
            raise ValueError(f"每个 {section} 项必须包含 id 和 {', '.join(required_keys)}")
        entry = {key: str(Path(item[key]).expanduser().resolve()) for key in required_keys}
        entry.update({key: str(item[key]) for key in optional_keys if item.get(key) is not None})
        entries[str(item["id"])] = entry
    return entries

def load_config(path):
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict): raise ValueError("配置顶层必须是映射")
    comfy, listen = raw.get("comfyui") or {}, raw.get("listen") or {}
    encoders = _load_named(raw.get("encoders"), ["clip_path"], ["clip_type"], "encoders")
    for item in encoders.values(): item.setdefault("clip_type", "krea2")
    upscalers = _load_named(raw.get("upscalers"), ["dit_path", "vae_path"], ["weight_dtype"], "upscalers")
    for item in upscalers.values(): item.setdefault("weight_dtype", "default")
    if not encoders and not upscalers: raise ValueError("至少需要配置一个 encoder 或 upscaler")
    return {"comfy_root": str(Path(comfy.get("root", "~/comfyui")).expanduser().resolve()), "host": str(listen.get("host", "127.0.0.1")), "port": int(listen.get("port", 50051)), "exclusive": bool(raw.get("exclusive_models", True)), "encoders": encoders, "upscalers": upscalers}

class Manager:
    def __init__(self, config):
        root = Path(config["comfy_root"]); sys.path.insert(0, str(root))
        import comfy.options; comfy.options.args_parsing = False
        import comfy.cli_args; comfy.cli_args.args.disable_xformers = True; comfy.cli_args.args.use_pytorch_cross_attention = True
        import comfy.model_management, folder_paths, nodes, torch
        from text_encoder import Encoder
        from seedvr2 import Upscaler
        from upscale import ImageUpscaler, METHODS
        self.torch, self.model_management = torch, comfy.model_management
        self.lock, self.exclusive = threading.RLock(), config["exclusive"]
        self.encoder = Encoder(config, torch, folder_paths, nodes, self.lock) if config["encoders"] else None
        self.upscaler = Upscaler(config, torch, folder_paths, nodes, self.lock) if config["upscalers"] else None
        self.image_upscaler = ImageUpscaler(torch, folder_paths, nodes, comfy.model_management, self.lock)
        self.resize_methods = METHODS
    def _release_other(self, loading):
        if not self.exclusive: return
        other = self.upscaler if loading == "encoder" else self.encoder
        if other is not None: other.release()
        self.model_management.unload_all_models(); self.model_management.soft_empty_cache()
    def load_clip(self, encoder_id, clip_type):
        if self.encoder is None: raise ValueError("未配置 encoders")
        with self.lock:
            if self.encoder.current_id != encoder_id: self._release_other("encoder")
            return self.encoder.load_clip(encoder_id, clip_type)
    def release_encoder(self):
        if self.encoder is not None: self.encoder.release()
    def load_upscaler(self, upscaler_id):
        if self.upscaler is None: raise ValueError("未配置 upscalers")
        with self.lock:
            if upscaler_id is None and len(self.upscaler.upscalers) == 1:
                upscaler_id = next(iter(self.upscaler.upscalers))
            if self.upscaler.current_id != upscaler_id: self._release_other("upscaler")
            return self.upscaler.load_upscaler(upscaler_id)
    def release_upscaler(self):
        if self.upscaler is not None: self.upscaler.release()
    def upscale(self, image, params):
        if self.upscaler is None: raise ValueError("未配置 upscalers")
        with self.lock:
            self.load_upscaler(params.get("upscaler_id"))
            return self.upscaler.upscale(image, params)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="config.yaml"); args = parser.parse_args(); config = load_config(args.config); manager = Manager(config)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def send_json(self, payload, status=200):
            data = json.dumps(payload, ensure_ascii=False).encode(); self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def send_binary(self, payload):
            out = io.BytesIO(); manager.torch.save(payload.to(manager.torch.float16), out); data = out.getvalue()
            self.send_response(200); self.send_header("Content-Type", "application/octet-stream"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0")); raw = self.rfile.read(length)
                if self.path == "/v1/upscale_seedvr2":
                    body = manager.torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
                    image = body.pop("image"); self.send_binary(manager.upscale(image, body)); return
                if self.path == "/v1/upscale_resize":
                    body = manager.torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
                    self.send_binary(manager.image_upscaler.resize(body.pop("image"), body.get("scale", 2.0), body.get("method", "lanczos"))); return
                if self.path == "/v1/upscale_model":
                    body = manager.torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
                    self.send_binary(manager.image_upscaler.model_upscale(body.pop("image"), body.get("scale", 2.0), body.get("method", "lanczos"), body["model"], body.get("tile", 512), body.get("overlap", 32))); return
                body = json.loads(raw or b"{}")
                if self.path == "/v1/ping": self.send_json({"ok": True, "encoder_ids": list(manager.encoder.encoders) if manager.encoder else [], "upscaler_ids": list(manager.upscaler.upscalers) if manager.upscaler else [], "upscale_models": manager.image_upscaler.list_models(), "resize_methods": manager.resize_methods}); return
                if self.path == "/v1/load_clip": self.send_json({"ok": True, "loaded": manager.load_clip(body.get("encoder_id"), body.get("clip_type"))}); return
                if self.path == "/v1/release":
                    manager.release_encoder()
                    print("[copilot] 已释放 encoder", flush=True)
                    self.send_json({"ok": True}); return
                if self.path == "/v1/load_upscaler": self.send_json({"ok": True, "loaded": manager.load_upscaler(body.get("upscaler_id"))}); return
                if self.path == "/v1/release_upscaler":
                    manager.release_upscaler()
                    print("[copilot] 已释放 upscaler", flush=True)
                    self.send_json({"ok": True}); return
                if self.path == "/v1/encode_text":
                    p, n = manager.encoder.text(body["prompt"], body.get("negative_prompt", ""), body.get("cfg", 1), body.get("reuse_negative_at_cfg_one", False)); self.send_json({"ok": True, "positive": pack(p, manager.torch), "negative": pack(n, manager.torch)}); return
                if self.path == "/v1/encode_grounded":
                    image = unpack(body["image"], manager.torch); image_b = unpack(body["image_b"], manager.torch) if body.get("image_b") else None; p, n = manager.encoder.grounded(body["prompt"], body.get("negative_prompt", ""), image, image_b, body.get("grounding_px", 768), body.get("cfg", 1), body.get("reuse_negative_at_cfg_one", False)); self.send_json({"ok": True, "positive": pack(p, manager.torch), "negative": pack(n, manager.torch)}); return
                raise ValueError("未知请求路径")
            except Exception as exc:
                print(f"[copilot] 请求失败: {type(exc).__name__}: {exc}", flush=True)
                self.send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 400)
    print(f"copilot listening on {config['host']}:{config['port']}", flush=True); ThreadingHTTPServer((config["host"], config["port"]), Handler).serve_forever()
if __name__ == "__main__": main()
