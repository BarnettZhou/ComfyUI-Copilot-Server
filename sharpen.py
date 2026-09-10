"""USM 锐化后处理：像素域高斯差值，无模型依赖，供 seedvr2 / resize / upscale model 复用。"""
import math


def sharpen_images(torch, images, strength, radius=2.0):
    """images 为 (B,H,W,C) 浮点 0-1 图像；strength<=0 时原样返回。

    out = clamp(img + strength * (img - gaussian_blur(img)))；边缘用 reflect
    padding 避免暗边。耗时相对扩散推理可忽略。
    """
    strength = float(strength or 0.0)
    if strength <= 0:
        return images
    if strength > 1:
        raise ValueError(f"sharpen 必须在 0 到 1 之间: {strength}")
    sigma = max(float(radius), 0.1)
    half = max(1, int(math.ceil(sigma * 2)))
    size = half * 2 + 1
    coords = torch.arange(size, dtype=torch.float32, device=images.device) - half
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    x = images.movedim(-1, 1).to(torch.float32)
    channels = int(x.shape[1])
    weight = kernel_2d.expand(channels, 1, size, size).contiguous()
    padded = torch.nn.functional.pad(x, (half, half, half, half), mode="reflect")
    blurred = torch.nn.functional.conv2d(padded, weight, groups=channels)
    sharp = (x + strength * (x - blurred)).clamp(0.0, 1.0)
    return sharp.movedim(1, -1)
