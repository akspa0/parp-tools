
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms.functional as TF
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# --- Configuration ---
# Reuse V7.6 Cache (Compatible tensors)
CACHED_DIR = Path("cached_v7_6") 
OUTPUT_DIR = Path("output_v8_light")
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
PREVIEW_DIR = OUTPUT_DIR / "previews"
PREVIEW_DIR.mkdir(exist_ok=True)

# V8 Settings
BATCH_SIZE = 8  # ResNet18 is lighter, we can bump this up from 4 to 8 safely
LEARNING_RATE = 1e-4
EPOCHS = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dataset ---
class V8Dataset(Dataset):
    def __init__(self, cached_dir):
        self.files = list(cached_dir.glob("input_*.pt"))
        print(f"Loaded {len(self.files)} samples from {cached_dir}.")
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        inp_path = self.files[idx]
        # Parse coordinates
        parts = inp_path.stem.split('_')
        x, y = parts[1], parts[2]
        
        tgt_h_path = self.files[idx].parent / f"target_height_{x}_{y}.pt"
        tgt_a_path = self.files[idx].parent / f"target_albedo_{x}_{y}.pt"
        
        # Load
        inp = torch.load(inp_path).float()
        tgt_h = torch.load(tgt_h_path).float()
        tgt_a = torch.load(tgt_a_path).float()
        
        # --- V8 CLEANING ---
        # 1. Enforce Shape
        if inp.shape[1] != 512 or inp.shape[2] != 512:
             inp = TF.resize(inp, (512, 512))
        
        # 2. Slice RGB Only (Discard 4th column/mask)
        # Input tensor is likely (C, H, W). We want first 3 channels.
        if inp.shape[0] > 3:
            inp = inp[:3, :, :]
             
        # Targets
        if tgt_h.shape[1] != 512: tgt_h = TF.resize(tgt_h, (512, 512))
        if tgt_a.shape[1] != 512: tgt_a = TF.resize(tgt_a, (512, 512))
        
        # Normalize Height (0-65535 -> 0-1)
        tgt_h = tgt_h / 65535.0
        
        return inp, tgt_h, tgt_a

# --- Model (ResNet18 Light) ---
class MultiHeadUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder (ResNet18) - Lighter than ResNet34
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # -> 256x256 (64 ch)
        self.pool = resnet.maxpool # -> 128x128
        self.enc1 = resnet.layer1  # -> 128x128 (64 ch)
        self.enc2 = resnet.layer2  # -> 64x64   (128 ch)
        self.enc3 = resnet.layer3  # -> 32x32   (256 ch)
        self.enc4 = resnet.layer4  # -> 16x16   (512 ch)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), # Reduced bottleneck channels from 1024 to 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Headers (Height + Albedo)
        # Shared UP-Block def
        def up_block(in_c, skip_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_c + skip_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
            
        # Height Path
        self.h_up4 = up_block(512, 256, 256) # 16->32
        self.h_up3 = up_block(256, 128, 128) # 32->64
        self.h_up2 = up_block(128, 64, 64)   # 64->128
        self.h_up1 = up_block(64, 64, 64)    # 128->256
        self.h_up0 = nn.Sequential(          # 512->512 (Refine)
            # h_up1 already reached 512x512 because X0 (256) was processed by up_block (256->512)
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

        # Albedo Path
        self.a_up4 = up_block(512, 256, 256)
        self.a_up3 = up_block(256, 128, 128)
        self.a_up2 = up_block(128, 64, 64)
        self.a_up1 = up_block(64, 64, 64)
        self.a_up0 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1), # RGB
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        x0 = self.enc0(x)      # 256 (64)
        x_p = self.pool(x0)    # 128 (64)
        x1 = self.enc1(x_p)    # 128 (64)
        x2 = self.enc2(x1)     # 64  (128)
        x3 = self.enc3(x2)     # 32  (256)
        x4 = self.enc4(x3)     # 16  (512)
        
        b = self.bottleneck(x4) # 16 (512)
        
        # Height Decoder
        # cat(upsample(b), x3) -> 512 + 256 = 768 in? 
        # My up_block def handles (in_c + skip_c).
        # h_up4 in_c=512. skip_c=256.
        h = self.h_up4(torch.cat([nn.functional.interpolate(b, size=x3.shape[2:]), x3], dim=1)) 
        h = self.h_up3(torch.cat([nn.functional.interpolate(h, size=x2.shape[2:]), x2], dim=1))
        h = self.h_up2(torch.cat([nn.functional.interpolate(h, size=x1.shape[2:]), x1], dim=1))
        # Last skip is x0? Or x1?
        # Typically indices: 4->3, 3->2, 2->1, 1->0
        # h_up1 uses x0 (256 res). Enc0 output.
        # But x0 is 256x256. h_up2 output is 128x128. Correct.
        h = self.h_up1(torch.cat([nn.functional.interpolate(h, size=x0.shape[2:]), x0], dim=1))
        h_out = self.h_up0(h) # Upsample to 512
        
        # Albedo Decoder
        a = self.a_up4(torch.cat([nn.functional.interpolate(b, size=x3.shape[2:]), x3], dim=1))
        a = self.a_up3(torch.cat([nn.functional.interpolate(a, size=x2.shape[2:]), x2], dim=1))
        a = self.a_up2(torch.cat([nn.functional.interpolate(a, size=x1.shape[2:]), x1], dim=1))
        a = self.a_up1(torch.cat([nn.functional.interpolate(a, size=x0.shape[2:]), x0], dim=1))
        a_out = self.a_up0(a)
        
        return h_out, a_out

# --- Perceptual Loss ---
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        for x in range(4): self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9): self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16): self.slice3.add_module(str(x), vgg[x])
        for param in self.parameters(): param.requires_grad = False
        
    def forward(self, input, target):
        input = (input - 0.5) / 0.5
        target = (target - 0.5) / 0.5
        h_relu1_2 = self.slice1(input)
        h_relu2_2 = self.slice2(h_relu1_2)
        h_relu3_3 = self.slice3(h_relu2_2)
        h_relu1_2_t = self.slice1(target)
        h_relu2_2_t = self.slice2(h_relu1_2_t)
        h_relu3_3_t = self.slice3(h_relu2_2_t)
        return nn.functional.l1_loss(h_relu1_2, h_relu1_2_t) + \
               nn.functional.l1_loss(h_relu2_2, h_relu2_2_t) + \
               nn.functional.l1_loss(h_relu3_3, h_relu3_3_t)

# --- Training Loop ---
def save_preview(inputs, pred_h, gt_h, pred_a, gt_a, epoch):
    img = inputs[0].cpu().detach()
    ph = pred_h[0].cpu().detach().repeat(3,1,1)
    gh = gt_h[0].cpu().detach().repeat(3,1,1)
    pa = pred_a[0].cpu().detach()
    ga = gt_a[0].cpu().detach()
    row = torch.cat([img, ph, gh, pa, ga], dim=2)
    TF.to_pil_image(row).save(PREVIEW_DIR / f"preview_epoch_{epoch}.png")

def train():
    if not CACHED_DIR.exists():
        print(f"Error: {CACHED_DIR} not found. Please run the cache script or rename the folder.")
        return

    ds = V8Dataset(CACHED_DIR)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    model = MultiHeadUNet().to(DEVICE)
    scaler = torch.amp.GradScaler('cuda')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    criterion_L1 = nn.L1Loss()
    criterion_VGG = VGGPerceptualLoss().to(DEVICE)
    
    print(f"Starting V8 Light Training ({EPOCHS} Epochs, ResNet18)...")
    
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(dl, desc=f"Epoch {epoch}")
        total_loss = 0
        
        for inputs, gt_h, gt_a in loop:
            inputs, gt_h, gt_a = inputs.to(DEVICE), gt_h.to(DEVICE), gt_a.to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                pred_h, pred_a = model(inputs)
                
                if True: # DEBUG SHAPES ONCE
                    tqdm.write(f"Pred H: {pred_h.shape}, GT H: {gt_h.shape}")
                    tqdm.write(f"Pred A: {pred_a.shape}, GT A: {gt_a.shape}")
                
                loss_h = criterion_L1(pred_h, gt_h)
                loss_a_l1 = criterion_L1(pred_a, gt_a)
                loss_a_vgg = criterion_VGG(pred_a, gt_a)
                
                # Weighted Loss
                loss = (loss_h * 2.0) + loss_a_l1 + (0.1 * loss_a_vgg)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            loop.set_postfix(l=loss.item(), h=loss_h.item(), a=loss_a_l1.item())
            
        print(f"Epoch {epoch} Avg Loss: {total_loss/len(dl):.4f}")
        
        if epoch % 1 == 0:
             save_preview(inputs, pred_h, gt_h, pred_a, gt_a, epoch)
             torch.save(model.state_dict(), CHECKPOINT_DIR / "latest.pth")
             
        if epoch % 50 == 0 and epoch > 0:
             torch.save(model.state_dict(), CHECKPOINT_DIR / f"epoch_{epoch}.pth")

if __name__ == "__main__":
    train()
