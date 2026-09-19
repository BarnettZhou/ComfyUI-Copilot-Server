"""浏览器端图像放大测试页：表单提交 → 单 worker 队列串行执行 → 结果存 .cache/。"""
import base64, io, queue, random, re, threading, time, uuid
from pathlib import Path

MODES = ("seedvr2", "resize", "model")
COLOR_FIX = ("lab", "wavelet", "adain", "none")
MAX_DIFFUSION_PIXELS = 4_000_000  # SeedVR2 扩散分辨率（宽×高）上限，超出会被整机 swap 拖死，请降低放大倍数或输入尺寸
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

class WebUI:
    def __init__(self, manager, root):
        self.manager, self.root = manager, Path(root)
        self.cache = Path(".cache"); self.cache.mkdir(exist_ok=True)
        self.jobs, self.queue = {}, queue.Queue()
        self.lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()

    def page(self):
        return (self.root / "web" / "index.html").read_bytes()

    def options(self):
        m = self.manager
        return {"modes": list(MODES),
                "upscaler_ids": list(m.upscaler.upscalers) if m.upscaler else [],
                "upscale_models": m.image_upscaler.list_models(),
                "resize_methods": list(m.resize_methods),
                "samplers": list(m.samplers), "schedulers": list(m.schedulers),
                "color_fix": list(COLOR_FIX)}

    def submit(self, body):
        mode = body.get("mode")
        if mode not in MODES: raise ValueError(f"未知模式: {mode}")
        data = str(body.get("image") or "")
        if "," in data: data = data.split(",", 1)[1]
        if not data: raise ValueError("缺少图片")
        raw = base64.b64decode(data)
        params = body.get("params") or {}
        if mode == "model" and not params.get("model"): raise ValueError("model 模式需要选择放大模型")
        if mode == "seedvr2":
            from PIL import Image
            w, h = Image.open(io.BytesIO(raw)).size
            scale = float(params.get("scale", 0.0) or 0.0)
            factor = scale if scale > 0 else 1.0
            if (w * factor) * (h * factor) > MAX_DIFFUSION_PIXELS:
                raise ValueError(f"扩散分辨率 {round(w * factor)}x{round(h * factor)} 超过上限 {MAX_DIFFUSION_PIXELS // 10000} 万像素，请降低放大倍数或输入尺寸")
        job = {"id": uuid.uuid4().hex[:12], "mode": mode, "params": params, "name": str(body.get("name") or "image"),
               "status": "pending", "created": time.time(), "started": None, "elapsed": None,
               "result": None, "error": None, "oom": False, "raw": raw}
        with self.lock: self.jobs[job["id"]] = job
        self.queue.put(job["id"])
        return {"ok": True, "job_id": job["id"]}

    def list_jobs(self):
        with self.lock: jobs = list(self.jobs.values())
        pending = [j["id"] for j in jobs if j["status"] == "pending"]
        out = [{k: v for k, v in j.items() if k != "raw"} | {"position": pending.index(j["id"]) + 1 if j["id"] in pending else 0} for j in jobs]
        out.sort(key=lambda j: -j["created"])
        return {"ok": True, "jobs": out}

    def cache_file(self, name):
        if not _SAFE_NAME.fullmatch(name): return None
        path = (self.cache / name).resolve()
        if path.parent != self.cache.resolve() or not path.is_file(): return None
        return path

    def _worker(self):
        while True:
            job_id = self.queue.get()
            with self.lock: job = self.jobs.get(job_id)
            if job is None: continue
            job.update(status="running", started=time.time())
            try:
                result = self._run(job)
                filename = f"{job['id']}.png"
                self._save_png(result, self.cache / filename)
                job.update(status="done", result=filename, elapsed=time.time() - job["started"])
                print(f"[copilot] web 任务完成: {job['id']} {job['mode']} ({job['elapsed']:.2f}s)", flush=True)
            except Exception as exc:
                if self.manager.model_management.is_oom(exc):
                    self.manager.model_management.soft_empty_cache(); job["oom"] = True
                job.update(status="error", error=f"{type(exc).__name__}: {exc}", elapsed=time.time() - job["started"])
                print(f"[copilot] web 任务失败: {job['id']} {type(exc).__name__}: {exc}", flush=True)
            finally:
                job.pop("raw", None)

    def _run(self, job):
        import numpy as np
        from PIL import Image
        image = Image.open(io.BytesIO(job["raw"])).convert("RGB")
        tensor = self.manager.torch.from_numpy(np.array(image)).float().div(255).unsqueeze(0)
        p = job["params"]
        if job["mode"] == "seedvr2":
            seed = int(p.get("seed", -1) or 0)
            if seed < 0:
                seed = random.SystemRandom().randint(0, 2**63 - 1)
                job["params"]["seed"] = str(seed)
            return self.manager.upscale(tensor, {
                "upscaler_id": p.get("upscaler_id") or None,
                "seed": seed, "steps": max(1, int(p.get("steps", 1))),
                "cfg": float(p.get("cfg", 1.0)), "sampler_name": p.get("sampler_name", "euler"),
                "scheduler": p.get("scheduler", "simple"), "denoise": float(p.get("denoise", 1.0)),
                "color_fix": p.get("color_fix", "lab"),
                "sharpen": float(p.get("sharpen", 0.0) or 0.0),
                "scale": float(p.get("scale", 0.0) or 0.0), "method": p.get("method", "lanczos"),
                "tile": int(p.get("tile", 0) or 0), "overlap": int(p.get("overlap", 64))})
        if job["mode"] == "resize":
            return self.manager.image_upscaler.resize(tensor, float(p.get("scale", 2.0)), p.get("method", "lanczos"), float(p.get("sharpen", 0.0) or 0.0))
        return self.manager.image_upscaler.model_upscale(tensor, float(p.get("scale", 2.0)),
                p.get("method", "lanczos"), p["model"], int(p.get("tile", 512)), int(p.get("overlap", 32)), float(p.get("sharpen", 0.0) or 0.0))

    def _save_png(self, tensor, path):
        from PIL import Image
        array = tensor[0].float().mul(255).round().clamp(0, 255).byte().cpu().numpy()
        Image.fromarray(array).save(path)
