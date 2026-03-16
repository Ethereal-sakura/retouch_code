import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

import lpips
import clip
from skimage.color import rgb2lab, deltaE_cie76


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load("ViT-B/32", device=device)
text = clip.tokenize([
    "a dark light photo", "a bright light photo",             # Exposure
    "a low-contrast photo", "a high-contrast photo",          # Contrast
    "a desaturated colours photo", "a vivid colours photo",   # Saturation
    "a cool-toned photo", "a warm-toned photo",               # Temperature
    ]).to(device)
text_feat = model.encode_text(text)

def img_preprocess(img, device):
    global preprocess
    return preprocess(img).unsqueeze(0).to(device)

def get_clip_score(src_imgs, tar_imgs, device):
    global model, text_feat
    src_imgs = [img_preprocess(img, device) for img in src_imgs]
    src_imgs = torch.cat(src_imgs, dim=0)

    tar_imgs = [img_preprocess(img, device) for img in tar_imgs]
    tar_imgs = torch.cat(tar_imgs, dim=0)

    all_imgs = torch.cat([src_imgs, tar_imgs], dim=0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(all_imgs)  # (N, D)
        text_features = text_feat  # (2, D)
 
        if text_feat.dtype == torch.int:  # tokenized
            text_features = model.encode_text(text_feat)
 
        logits_per_image = image_features @ text_features.T
        src_logits = logits_per_image[:len(src_imgs), :]
        tar_logits = logits_per_image[len(src_imgs):, :]
    return src_logits.cpu(), tar_logits.cpu()
          
def get_idx(src_imgs: Image.Image, tar_imgs: Image.Image, device='cpu', s_type='clip', size=(64, 64)):
    if s_type=='psnr':
        psnr_matrix = get_psnr_matrix(src_imgs, tar_imgs, size, device)
        if len(tar_imgs) > 1:
            _, indices = torch.topk(psnr_matrix, k=3, dim=1, largest=True, sorted=False)
            top3_values = torch.gather(psnr_matrix, 1, indices)
            average_top3 = top3_values.mean(dim=1, keepdim=True)
            idx = torch.argmax(average_top3).item()
        else:
            idx = torch.argmax(psnr_matrix).item()
        return idx
    if s_type=='clip':
        src_logits, tar_logits = get_clip_score(src_imgs, tar_imgs, device)
        src_probs = F.softmax(src_logits, dim=1)
        tar_probs = F.softmax(tar_logits, dim=1)
        mean_probs = tar_probs.mean(dim=0, keepdim=True)    # [1, 2]
        
        kl_each = F.kl_div(
            src_probs.log(),                # log q(x): candidate log-prob
            mean_probs.expand_as(src_probs), # p(x): reference mean prob
            reduction="none"
        ).sum(dim=1)   # [M]
        idx = torch.argmin(kl_each).item()
        return idx    

def pil_to_tensor(img, size=(64, 64)):
    img = img.resize(size)
    img = np.array(img).astype(np.float32)
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img

def pil_to_np(img, size=(64, 64)):
    img = img.resize(size)
    img = np.array(img).astype(np.float32)
    return img

def get_psnr_matrix(src_imgs: list, tar_imgs: list, size=(64, 64), device='cpu'):
    src_imgs_tesnors = [pil_to_tensor(img, size) / 255.0 for img in src_imgs]
    tar_imgs_tensors = [pil_to_tensor(img, size) / 255.0 for img in tar_imgs]
    src_imgs_tensor = torch.stack(src_imgs_tesnors).to(device)
    tar_imgs_tensor = torch.stack(tar_imgs_tensors).to(device)
    
    mse = torch.mean((src_imgs_tensor.unsqueeze(1) - tar_imgs_tensor.unsqueeze(0)) ** 2, dim=(2, 3, 4))  
    psnr_matrix = 10 * torch.log10(1/mse)
    return psnr_matrix.cpu()

def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: SSIM result.
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    c1 = (0.01)**2
    c2 = (0.03)**2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid mode for window size 11
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return (ssim_map).mean()

def get_ssim(src, tar):
    """
    src: [0, 255] uint8
    tar: [0, 255] uint8
    """
    ssims = []
    for i in range(tar.shape[2]):
        ssims.append(_ssim(tar[..., i], src[..., i]))
    return np.mean(ssims)

lpips_vgg = lpips.LPIPS(net='vgg')
def get_lpips(src, tar, device='cpu'):
    global lpips_vgg
    src = (src.unsqueeze(0) / 255.0).to(device)
    tar = (tar.unsqueeze(0) / 255.0).to(device)
    # normalization [-1, 1]
    src = src * 2 - 1
    tar = tar * 2 - 1

    lpips_vgg = lpips_vgg.to(device)
    lpips = lpips_vgg(src, tar)
    return lpips.cpu().item()
    
def calculate_dE(img1, img2):
    return np.array(deltaE_cie76(rgb2lab(img1),rgb2lab(img2))).mean()

def get_final_scores(src_img, tar_img, size=(64, 64), device='cpu', p=True):
    np_src, np_tar = pil_to_np(src_img, size), pil_to_np(tar_img, size)
    tensor_src, tensor_tar = pil_to_tensor(src_img, size), pil_to_tensor(tar_img, size)

    psnr = get_psnr_matrix([src_img], [tar_img], size, device).item()
    psnr = round(psnr, 2)

    ssim = get_ssim(np_src / 255.0, np_tar / 255.0)
    ssim = round(ssim, 3)

    lpips = get_lpips(tensor_src, tensor_tar, device)
    lpips = round(lpips, 3)

    delta_e = calculate_dE(np_src / 255.0, np_tar / 255.0)
    delta_e = round(delta_e, 2)

    if p:
        print(f"PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips}, Delta E: {delta_e}")
    return (psnr, ssim, lpips, delta_e)
