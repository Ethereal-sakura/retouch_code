"""Quality metrics for trajectory evaluation.

Provides PSNR, SSIM, LPIPS, and DeltaE metrics.
LPIPS is loaded lazily to avoid slow startup when not needed.
"""

from __future__ import annotations

import numpy as np
import cv2
import torch
from PIL import Image
from skimage.color import rgb2lab, deltaE_cie76


_lpips_model = None


def _get_lpips():
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net="vgg")
    return _lpips_model


def _pil_to_np(img: Image.Image | np.ndarray, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Resize and convert to float32 numpy array in [0, 1]."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    img = img.resize(size, Image.BILINEAR)
    return np.array(img).astype(np.float32) / 255.0


def _pil_to_tensor(img: Image.Image | np.ndarray, size: tuple[int, int] = (64, 64)) -> torch.Tensor:
    np_img = _pil_to_np(img, size)
    return torch.from_numpy(np_img).permute(2, 0, 1)


def compute_psnr(
    src: Image.Image | np.ndarray,
    tar: Image.Image | np.ndarray,
    size: tuple[int, int] = (64, 64),
) -> float:
    src_t = _pil_to_tensor(src, size).unsqueeze(0)
    tar_t = _pil_to_tensor(tar, size).unsqueeze(0)
    mse = torch.mean((src_t - tar_t) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return round(10 * np.log10(1.0 / mse), 2)


def _ssim_channel(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM for a single channel, both in [0, 1]."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.T)

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    s1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    s2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    s12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * s12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (s1_sq + s2_sq + c2)
    )
    return float(ssim_map.mean())


def compute_ssim(
    src: Image.Image | np.ndarray,
    tar: Image.Image | np.ndarray,
    size: tuple[int, int] = (64, 64),
) -> float:
    src_np = _pil_to_np(src, size)
    tar_np = _pil_to_np(tar, size)
    ssims = [_ssim_channel(src_np[..., i], tar_np[..., i]) for i in range(3)]
    return round(float(np.mean(ssims)), 3)


def compute_lpips(
    src: Image.Image | np.ndarray,
    tar: Image.Image | np.ndarray,
    size: tuple[int, int] = (64, 64),
    device: str = "cpu",
) -> float:
    model = _get_lpips().to(device)
    src_t = (_pil_to_tensor(src, size).unsqueeze(0) / 1.0 * 2 - 1).to(device)
    tar_t = (_pil_to_tensor(tar, size).unsqueeze(0) / 1.0 * 2 - 1).to(device)
    with torch.no_grad():
        val = model(src_t, tar_t)
    return round(float(val.cpu().item()), 3)


def compute_delta_e(
    src: Image.Image | np.ndarray,
    tar: Image.Image | np.ndarray,
    size: tuple[int, int] = (64, 64),
) -> float:
    src_np = _pil_to_np(src, size)
    tar_np = _pil_to_np(tar, size)
    de = deltaE_cie76(rgb2lab(src_np), rgb2lab(tar_np))
    return round(float(np.array(de).mean()), 2)


def compute_metrics(
    src: Image.Image | np.ndarray,
    tar: Image.Image | np.ndarray,
    size: tuple[int, int] = (64, 64),
    use_lpips: bool = False,
    device: str = "cpu",
) -> dict:
    """Compute all quality metrics between two images.

    Returns dict with keys: psnr, ssim, delta_e, and optionally lpips.
    """
    result = {
        "psnr": compute_psnr(src, tar, size),
        "ssim": compute_ssim(src, tar, size),
        "delta_e": compute_delta_e(src, tar, size),
    }
    if use_lpips:
        result["lpips"] = compute_lpips(src, tar, size, device)
    return result
