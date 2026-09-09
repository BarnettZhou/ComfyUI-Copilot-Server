"""远程文本编码服务：CLIPTextEncode 和 Krea2EditGroundedEncode。从 ComfyUI-Text-Encoder-Server 迁移。"""
import importlib.util, sys, time
from pathlib import Path

class Encoder:
    def __init__(self, config, torch, folder_paths, nodes, lock):
        self.torch, self.folder_paths, self.nodes, self.lock = torch, folder_paths, nodes, lock
        self.encoders, self.current_id, self.clip = config["encoders"], None, None
        self.krea2edit = None
    def release(self):
        with self.lock:
            self.clip = None; self.current_id = None
    def load_clip(self, encoder_id, clip_type):
        with self.lock:
            item = self.encoders.get(encoder_id)
            if item is None:
                for candidate in self.encoders.values():
                    if Path(candidate["clip_path"]).name == encoder_id:
                        item = candidate
                        encoder_id = next(key for key, value in self.encoders.items() if value is candidate)
                        break
            if item is None: raise ValueError(f"未知 encoder_id: {encoder_id}")
            if self.current_id == encoder_id and self.clip is not None:
                print(f"[copilot] 复用 encoder: {encoder_id}", flush=True)
                return False
            path = Path(item["clip_path"])
            if not path.is_file(): raise FileNotFoundError(f"找不到 text encoder: {path}")
            print(f"[copilot] 加载 encoder: {encoder_id}", flush=True)
            started = time.perf_counter()
            self.clip = None; self.folder_paths.add_model_folder_path("text_encoders", str(path.parent), is_default=True)
            self.clip = self.nodes.CLIPLoader().load_clip(path.name, clip_type or item["clip_type"])[0]; self.current_id = encoder_id
            print(f"[copilot] encoder 就绪: {encoder_id} ({time.perf_counter() - started:.2f}s)", flush=True)
            return True
    def _nodes(self):
        if self.krea2edit is not None: return self.krea2edit
        root = Path(self.folder_paths.__file__).resolve().parent / "custom_nodes" / "comfyui-krea2edit"; entry = root / "__init__.py"
        spec = importlib.util.spec_from_file_location("remote_krea2edit", entry, submodule_search_locations=[str(root)])
        if spec is None or spec.loader is None: raise RuntimeError(f"无法加载节点: {entry}")
        module = importlib.util.module_from_spec(spec); sys.modules["remote_krea2edit"] = module; spec.loader.exec_module(module); self.krea2edit = module
        return module
    def text(self, prompt, negative_prompt, cfg, reuse):
        with self.lock, self.torch.inference_mode():
            if self.clip is None: raise RuntimeError("尚未加载 encoder，请先调用 /v1/load_clip")
            started = time.perf_counter(); print("[copilot] 开始普通文本编码", flush=True)
            enc = self.nodes.CLIPTextEncode(); positive = enc.encode(self.clip, prompt)[0]; negative = positive if reuse and cfg == 1.0 else enc.encode(self.clip, negative_prompt)[0]
            print(f"[copilot] 普通文本编码完成 ({time.perf_counter() - started:.2f}s)", flush=True); return positive, negative
    def grounded(self, prompt, negative_prompt, image, image_b, grounding_px, cfg, reuse):
        with self.lock, self.torch.inference_mode():
            if self.clip is None: raise RuntimeError("尚未加载 encoder，请先调用 /v1/load_clip")
            started = time.perf_counter(); kind = "双图" if image_b is not None else "单图"; print(f"[copilot] 开始{kind}编辑编码", flush=True)
            node = self._nodes().NODE_CLASS_MAPPINGS["Krea2EditGroundedEncode"](); positive = node.encode(self.clip, prompt, image=image, image_b=image_b, grounding_px=int(grounding_px), system_prompt="")[0]
            if reuse and cfg == 1.0: negative = positive
            elif not negative_prompt: negative = self.nodes.ConditioningZeroOut().zero_out(positive)[0]
            else: negative = node.encode(self.clip, negative_prompt, image=image, image_b=image_b, grounding_px=int(grounding_px), system_prompt="")[0]
            print(f"[copilot] {kind}编辑编码完成 ({time.perf_counter() - started:.2f}s)", flush=True); return positive, negative
