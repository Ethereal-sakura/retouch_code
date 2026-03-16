"""Image I/O and encoding utilities."""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk as float32 [0, 1] RGB numpy array.

    Delegates to image_engine's load_image for format support (TIF, JPG, PNG).
    """
    # Add image_engine to path if needed
    engine_path = Path(__file__).parent.parent.parent / "image_engine"
    if str(engine_path) not in sys.path:
        sys.path.insert(0, str(engine_path))
    from rapidraw_basic_color.io import load_image as _load
    return _load(path)


def save_image(path: str | Path, image: np.ndarray, quality: int = 85) -> None:
    """Save a float32 [0, 1] image to disk."""
    engine_path = Path(__file__).parent.parent.parent / "image_engine"
    if str(engine_path) not in sys.path:
        sys.path.insert(0, str(engine_path))
    from rapidraw_basic_color.io import save_image as _save
    _save(path, image, quality=quality)


def np_to_pil(img: np.ndarray) -> Image.Image:
    """Convert float32 [0, 1] numpy array to PIL Image."""
    return Image.fromarray((np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8))


def pil_to_np(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to float32 [0, 1] numpy array."""
    return np.array(img).astype(np.float32) / 255.0


def resize_image(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a float32 [0, 1] image to (width, height)."""
    pil = np_to_pil(img)
    pil = pil.resize(size, Image.BILINEAR)
    return pil_to_np(pil)


def encode_image_base64(img: np.ndarray, format: str = "JPEG", quality: int = 85) -> str:
    """Encode a float32 [0, 1] numpy image to base64 string.

    Parameters
    ----------
    img : np.ndarray
        Float32 image, shape (H, W, 3), values in [0, 1].
    format : str
        Image format: "JPEG" or "PNG".
    quality : int
        JPEG quality (ignored for PNG).

    Returns
    -------
    str
        Base64-encoded image string (no data URI prefix).
    """
    pil = np_to_pil(img)
    buf = io.BytesIO()
    if format.upper() == "JPEG":
        pil.save(buf, format="JPEG", quality=quality, subsampling=0)
    else:
        pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def make_thumbnail(img: np.ndarray, max_size: int = 512) -> np.ndarray:
    """Resize image so the longest side is at most max_size pixels."""
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return resize_image(img, (new_w, new_h))
