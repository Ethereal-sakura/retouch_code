from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _normalize_image(arr: np.ndarray) -> np.ndarray:
    """将任意格式的图像数组统一转换为 float32、shape=(H, W, 3)、值域 [0, 1] 的 RGB 数组。
    
    处理以下情况：
    - 灰度图（2D）→ 复制 3 通道
    - RGBA（4 通道）→ 丢弃 Alpha 通道
    - 单通道 3D → 复制 3 通道
    - float 类型 → 若最大值 > 1 则按实际最大值归一化，否则直接转 float32
    - uint16 → 除以 65535
    - uint8 及其他整数 → 除以 255
    """
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
        if arr.max(initial=0.0) > 1.0:
            arr /= max(float(arr.max(initial=1.0)), 1.0)
    elif arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    else:
        arr = arr.astype(np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def load_image(path: str | Path) -> np.ndarray:
    """从磁盘加载图像，返回 float32 线性/sRGB 数组（取决于文件本身的色彩空间）。
    
    支持格式：JPG、PNG、TIF/TIFF。
    返回值：shape=(H, W, 3)，dtype=float32，值域 [0, 1]。
    注意：不做 sRGB→linear 转换，由 engine.py 根据 input_color_space 参数决定是否转换。
    """
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported input format: {path.suffix}")
    arr = np.asarray(iio.imread(path))
    return _normalize_image(arr)


def save_image(path: str | Path, image: np.ndarray, quality: int = 95) -> None:
    """将 float32 图像（值域 [0,1]）保存到磁盘。
    
    - JPG：使用 PIL 编码，quality 参数控制压缩质量，subsampling=0 保留色度细节
    - PNG：使用 PIL 编码，8 位无损
    - TIF/TIFF：使用 imageio 编码，16 位无损（uint16，值域 0~65535）
    
    输出前会将图像 clip 到 [0, 1]，转换时加 0.5 进行四舍五入。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    image = np.clip(image.astype(np.float32), 0.0, 1.0)

    if suffix in {".jpg", ".jpeg"}:
        Image.fromarray((image * 255.0 + 0.5).astype(np.uint8)).save(
            path, quality=int(quality), subsampling=0, optimize=True
        )
        return
    if suffix == ".png":
        Image.fromarray((image * 255.0 + 0.5).astype(np.uint8)).save(path)
        return
    if suffix in {".tif", ".tiff"}:
        iio.imwrite(path, (image * 65535.0 + 0.5).astype(np.uint16))
        return
    raise ValueError(f"Unsupported output format: {suffix}")
