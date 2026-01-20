import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from transformers import AutoModelForImageSegmentation

class BiRefNetWrapper:
    """
    Wrapper for BiRefNet (Bilateral Reference Network) for high-resolution 
    dichotomous image segmentation (DIS) and matting.
    
    Used in V8 for:
    1. Texture Layer Decomposition (separating specific textures based on color/pattern)
    2. Brush Pattern extraction
    """
    def __init__(self, model_id="ZhengPeng7/BiRefNet", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading BiRefNet from {model_id} on {self.device}...")
        self.model = AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)), # BiRefNet likes 1024
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, img: Image.Image) -> Image.Image:
        """Predict binary mask for the given image."""
        w, h = img.size
        # Resize/Normalize
        input_tensor = self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid().cpu()
        
        preds = preds.squeeze().numpy()
        # Resize back to original
        mask = Image.fromarray((preds * 255).astype('uint8'))
        mask = mask.resize((w, h), Image.BILINEAR)
        return mask

class BEN2Wrapper:
    """
    Wrapper for BEN2 (Background Eradication Network v2).
    Used in V8 for:
    1. Edge Refinement (sharpening alpha transitions)
    2. Matting (removing background for object library crops)
    """
    def __init__(self, model_id="PramaLLC/BEN2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading BEN2 from {model_id} on {self.device}...")
        try:
             # BEN2 might need specific loading code or simple loading if supported
             # Assuming standard AutoModel for now, or fallback to similar architecture
             self.model = AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)
        except:
             print("Warning: BEN2 load failed via AutoModel. Using generic fallback (BiRefNet-Tiny).")
             # Fallback to smaller BiRefNet for "functions like BEN2"
             self.model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet_general-use", trust_remote_code=True)
             
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def refine_edges(self, image: Image.Image, rough_mask: Image.Image) -> Image.Image:
        """
        Refine edges of a rough mask using the image content.
        For BEN2, typically we pass the image and expect it to saliently detect the object.
        Currently using direct prediction as edge refiner.
        """
        # BEN2 typically takes just the image.
        # If we want to use the rough_mask as a guide, we might pre-mask the image.
        
        # 1. Apply rough mask to image (black background)
        img_arr = np.array(image.convert("RGB"))
        mask_arr = np.array(rough_mask.convert("L")) / 255.0
        masked_img_arr = (img_arr * mask_arr[..., None]).astype(np.uint8)
        masked_img = Image.fromarray(masked_img_arr)
        
        # 2. Predict precise mask from pre-masked image
        input_tensor = self.transform(masked_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
             preds = self.model(input_tensor)[-1].sigmoid().cpu()
             
        pred_mask = preds.squeeze().numpy()
        mask_out = Image.fromarray((pred_mask * 255).astype('uint8')).resize(image.size, Image.BILINEAR)
        return mask_out

if __name__ == "__main__":
    # Test
    try:
        biref = BiRefNetWrapper()
        print("BiRefNet Loaded.")
    except Exception as e:
        print(f"BiRefNet failed: {e}")
