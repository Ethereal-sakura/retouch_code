import numpy as np
import cv2
from PIL import Image


def load_img(img_path):
    """Load image from path and return as numpy array in [0, 1] range."""
    img = Image.open(img_path).convert('RGB')
    return np.array(img).astype(np.float32) / 255.0


def convert_to_np(img):
    if isinstance(img, Image.Image):
        img = np.array(img).astype(np.float32) / 255.0
    else:
        raise ValueError("Unsupported image type")
    return img


def convert_to_pil(img):
    """Convert numpy array [0, 1] to PIL Image."""
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        if np.max(img) <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        return Image.fromarray(img)
    raise ValueError("Unsupported image type")

def load_metadata(img):
    return img.info

class AdjustmentFilter:
    def __init__(self, ori_img, gt_img=None):
        """
        Initilize unclip_image with the original raw image.
        self.unclip_img will be updated with the adjustment filters
        """
        self.unclip_img = convert_to_np(ori_img)  
        self.clip_img = convert_to_np(ori_img)
        if gt_img is not None:
            self.metadata = load_metadata(gt_img)
        else:
            self.metadata = {}
    
    def save_img(self, img, save_pth):
        assert ".jpg" not in save_pth
        if isinstance(img, Image.Image):
            img.save(save_pth, **self.metadata)
        else:
            if np.max(img) <= 1.0:
                img = (img * 255)
            img = Image.fromarray(img.astype(np.uint8))
            img.save(save_pth, **self.metadata)
        
    def refresh(self, img):
        self.unclip_img = convert_to_np(img)  
        self.clip_img = convert_to_np(img)

    def clip(self, img=None):
        if img is not None:
            self.clip_img = img
        else:
            self.unclip_img = self.clip_img.copy()
    
    def white_balance(self, f_r, f_g, f_b):
        """
        Manually adjusts the white balance of an image using RGB scaling parameters.
        - f_r (float): Scaling factor for the red channel (range: 0.1 to 5.0).
        - f_g (float): Scaling factor for the green channel (range: 0.1 to 5.0).
        - f_b (float): Scaling factor for the blue channel (range: 0.1 to 5.0).
        """
        adjusted_img = self.clip_img.copy()
        adjusted_img[:, :, 0] = np.clip(self.clip_img[:, :, 0] * float(f_r), 0, 1)  # Red channel scaling
        adjusted_img[:, :, 1] = np.clip(self.clip_img[:, :, 1] * float(f_g), 0, 1)  # Green channel scaling
        adjusted_img[:, :, 2] = np.clip(self.clip_img[:, :, 2] * float(f_b), 0, 1)  # Blue channel scaling
    
        self.clip_img = adjusted_img
        self.unclip_img = adjusted_img
        return self.clip_img

    def exposure(self, f_exp=0.0):
        """
        Adjusts the exposure of an image by scaling its pixel values. (overall brightness)        
        - f_exp: [-1, 1]
        """
        f_exp = float(f_exp) + 1  # scale f_exp from [-1, 1] to [0, 2]
        self.unclip_img *= f_exp
        self.clip_img = np.clip(self.unclip_img, 0, 1)
        return self.clip_img
    
    def gamma(self, f_gam=1.0):
        """
        Adjusts the gamma of an image by applying a nonlinear transformation to its pixel values.
        - f_gam: A gamma compression factor. A value of 1.0 means no change.
            Values greater than 1.0 brighten the image's shadows and reduce its highlights.
            Values less than 1.0 darken the image's shadows and brighten its highlights.
            Range: Typically [0.1, 5.0], but can be adjusted as needed.
        """
        self.unclip_img = np.power(np.maximum(self.unclip_img, 0.001), 1/f_gam)
        self.clip_img = np.clip(self.unclip_img, 0, 1)
        return self.clip_img
       
    def contrast(self, f_cont=0.0):
        """
        Adjusts the contrast of an image by modifying its pixel values based on a scaling factor.    
        - f_cont: [-1, 1]
        """
        f_cont = 1 + float(f_cont) * 0.5
        self.unclip_img = f_cont * self.unclip_img + (1 - f_cont) * np.mean(self.unclip_img)
        self.clip_img = np.clip(self.unclip_img, 0, 1)
        return self.clip_img
              
    def texture(self, f_text=0.0):
        """
        Adjusts the texture of an image by modifying high-frequency details using Gaussian blur. 
        This process can either enhance or soften the texture. 
        - f_text: [-1, 1]
        """
        blurred = cv2.GaussianBlur(self.unclip_img, (7, 7), 0)
        if f_text < 0: # (softening)
            self.unclip_img = cv2.addWeighted(self.unclip_img, 1 + f_text, blurred, -f_text, 0)
        else: # (sharpening)
            high_freq = cv2.subtract(self.unclip_img, blurred)  # Extract high-frequency details
            self.unclip_img = cv2.addWeighted(self.unclip_img, 1, high_freq, f_text, 0)
        self.clip_img = np.clip(self.unclip_img, 0, 1)
        return self.clip_img
    
    def temperature(self, f_temp=0.0):
        """
        Adjusts the color temperature of an image by modifying the balance between warm and cool tones.
        Positive values shift colors towards warmer tones (more red), while negative values shift towards cooler tones (more blue).
        - f_temp: [-1, 1]
        """
        f_r = 1.0 + 0.2 * float(f_temp)
        f_b = 1.0 - 0.2 * float(f_temp)
        f_g = 1.0 - 0.05 * float(f_temp)
        self.unclip_img[..., 0] *= f_r
        self.unclip_img[..., 2] *= f_b
        self.unclip_img[..., 1] *= f_g
        self.clip_img = np.clip(self.unclip_img, 0, 1)
        return self.clip_img

    def saturation(self, f_sat=0.0):
        """
        Adjusts the saturation of an image by scaling the intensity of its colors. 
        A higher value increases color vibrancy, while a lower value desaturates the image towards grayscale.
        - f_sat: [-1, 1]
        """
        f_sat = 0.5 * float(f_sat) + 1  # scale f_sat from [-1, 1] to [0.5, 1.5]
        hsv = cv2.cvtColor(self.unclip_img, cv2.COLOR_RGB2HSV_FULL)
        hsv[..., 1] *= f_sat
        self.unclip_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)

        hsv[..., 1] = np.clip(hsv[..., 1], 0, 1)
        hsv[..., 2] = np.clip(hsv[..., 2], 0, 1)
        self.clip_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
        return self.clip_img
    
    def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))
    
    def highlight(self, f_high=0.0):
        """
        Adjusts the brightness of the bright areas of an image by modifying the V (value) channel in HSV color space,
        using a highlight mask derived from a sigmoid function and a scaling factor.
        - f_high: [-1, 1]
        """
        f_high = 10 * float(f_high)
        hsv = cv2.cvtColor(self.unclip_img, cv2.COLOR_RGB2HSV_FULL)
        v = hsv[...,2]
        highlight_mask = self.sigmoid((v - 1) * 13)

        hsv[...,2] = 1 - (1 - v) * (1 - highlight_mask * f_high)
        self.unclip_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
        
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 1)
        hsv[..., 2] = np.clip(hsv[..., 2], 0, 1)
        self.clip_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
        return self.clip_img

    def shadow(self, f_shad=0.0):
        """
        Adjusts the brightness of the dark areas of an image by modifying the V (value) channel in HSV color space,
        using a shadow mask derived from a sigmoid function and a scaling factor.
        - f_high: [-1, 1]
        """
        f_shad = 5 * float(f_shad)
        hsv = cv2.cvtColor(self.unclip_img, cv2.COLOR_RGB2HSV_FULL)
        v = hsv[...,2]
        shadow_mask = 1 - self.sigmoid((v - 0) * 12)

        hsv[...,2] = v * (1 + shadow_mask * f_shad)
        self.unclip_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 1)
        hsv[..., 2] = np.clip(hsv[..., 2], 0, 1)
        self.clip_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
        return self.clip_img
    
