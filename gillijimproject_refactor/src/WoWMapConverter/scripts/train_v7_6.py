
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
CACHED_DIR = Path("cached_v7_6")
OUTPUT_DIR = Path("output_v7_6")
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 4 # Adjust for VRAM (512x512 float16 tensors)
LEARNING_RATE = 1e-4
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dataset ---
class V7Dataset(Dataset):
    def __init__(self, cached_dir):
        self.files = list(cached_dir.glob("input_*.pt"))
        print(f"Loaded {len(self.files)} samples.")
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        inp_path = self.files[idx]
        # input_X_Y.pt -> target_height_X_Y.pt 
        #               -> target_albedo_X_Y.pt
        
        # Parse coordinates
        # inp_path.name example: input_32_48.pt
        parts = inp_path.stem.split('_')
        x, y = parts[1], parts[2]
        
        tgt_h_path = self.files[idx].parent / f"target_height_{x}_{y}.pt"
        tgt_a_path = self.files[idx].parent / f"target_albedo_{x}_{y}.pt"
        
        # Load (Already tensors, float16)
        # Convert to float32 for training stability, mixed precision handles the rest
        inp = torch.load(inp_path).float()
        tgt_h = torch.load(tgt_h_path).float()
        tgt_a = torch.load(tgt_a_path).float()
        
        # Enforce strict input shape (3, 512, 512)
        if inp.shape[0] == 4:
            inp = inp[:3]
        if inp.shape[1] != 512 or inp.shape[2] != 512:
             inp = TF.resize(inp, (512, 512))
             
        if inp.shape[1] != 512 or inp.shape[2] != 512:
             inp = TF.resize(inp, (512, 512))
             
        # Enforce target shapes
        if tgt_h.shape[1] != 512: tgt_h = TF.resize(tgt_h, (512, 512))
        if tgt_a.shape[1] != 512: tgt_a = TF.resize(tgt_a, (512, 512))
        
        # Normalize Height (cached as raw I16) to 0-1
        tgt_h = tgt_h / 65535.0
        
        return inp, tgt_h, tgt_a

# --- Model Components ---
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class MultiHeadUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- Shared Encoder (ResNet34) ---
        # We can use torchvision resnet, removing the fc/avgpool
        # Features: 
        #   x: (3, 512, 512)
        #   layer0: (64, 256, 256) (conv1+bn+relu) -> maxpool -> (64, 128, 128)
        #   layer1: (64, 128, 128)
        #   layer2: (128, 64, 64)
        #   layer3: (256, 32, 32)
        #   layer4: (512, 16, 16)
        
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # -> 256x256
        self.pool = resnet.maxpool # -> 128x128
        self.enc1 = resnet.layer1  # -> 128x128 (64ch)
        self.enc2 = resnet.layer2  # -> 64x64   (128ch)
        self.enc3 = resnet.layer3  # -> 32x32   (256ch)
        self.enc4 = resnet.layer4  # -> 16x16   (512ch)
        
        # Top-level (Bottleneck)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # --- Header A: Height Decoder ---
        # Upsamples: 16->32->64->128->256->512
        # ResNet34 Channels: enc4=512, enc3=256, enc2=128, enc1=64, enc0=64
        self.h_up4 = self._up_block(1024, 256, 256) # 16->32, cat 256 (enc3) -> 1024+256=1280 in -> 256 out
        self.h_up3 = self._up_block(256, 128, 128)  # 32->64, cat 128 (enc2) -> 256+128=384 in -> 128 out
        self.h_up2 = self._up_block(128, 64, 64)    # 64->128, cat 64 (enc1) -> 128+64=192 in -> 64 out
        self.h_up1 = self._up_block(64, 64, 64)     # 128->256, cat 64 (enc0) -> 64+64=128 in -> 64 out
        self.h_up0 = nn.Sequential(                 # 512 Refinement
            # No Upsample here, h_up1 already reached 512
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1), # Output 1 channel
            nn.Sigmoid()         # 0-1 Range
        )

        # --- Header B: Albedo Decoder ---
        # Similar structure, separate weights
        self.a_up4 = self._up_block(1024, 256, 256)
        self.a_up3 = self._up_block(256, 128, 128)
        self.a_up2 = self._up_block(128, 64, 64)
        self.a_up1 = self._up_block(64, 64, 64)
        self.a_up0 = nn.Sequential(
            # No Upsample
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1), # Output 3 channels (RGB)
            nn.Sigmoid()         # 0-1 Range
        )
        
    def _up_block(self, in_ch, skip_ch, out_ch):
        # In: in_ch. Upsample -> in_ch. Cat skip_ch -> in_ch + skip_ch. Conv -> out_ch.
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # After cat, channels = in_ch + skip_ch
            # We use a helper GenericBlock for the conv part?
            # Or just standard conv block
            ConvBlock(in_ch + skip_ch, out_ch)
        )

    def forward(self, x):
        # Encoder
        x0 = self.enc0(x)      # 256 (64)
        x_p = self.pool(x0)    # 128 (64)
        x1 = self.enc1(x_p)    # 128 (64)
        x2 = self.enc2(x1)     # 64 (128)
        x3 = self.enc3(x2)     # 32 (256)
        x4 = self.enc4(x3)     # 16 (512)
        
        b = self.bottleneck(x4) # 16 (1024)
        
        # Height Head
        h = self.h_up4(torch.cat([nn.functional.interpolate(b, size=x3.shape[2:]), x3], dim=1))
        h = self.h_up3(torch.cat([nn.functional.interpolate(h, size=x2.shape[2:]), x2], dim=1))
        h = self.h_up2(torch.cat([nn.functional.interpolate(h, size=x1.shape[2:]), x1], dim=1))
        h = self.h_up1(torch.cat([nn.functional.interpolate(h, size=x0.shape[2:]), x0], dim=1))
        h_out = self.h_up0(h)
        
        # Albedo Head
        a = self.a_up4(torch.cat([nn.functional.interpolate(b, size=x3.shape[2:]), x3], dim=1))
        a = self.a_up3(torch.cat([nn.functional.interpolate(a, size=x2.shape[2:]), x2], dim=1))
        a = self.a_up2(torch.cat([nn.functional.interpolate(a, size=x1.shape[2:]), x1], dim=1))
        a = self.a_up1(torch.cat([nn.functional.interpolate(a, size=x0.shape[2:]), x0], dim=1))
        a_out = self.a_up0(a)
        
        return h_out, a_out

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# --- Perceptual Loss Utility ---
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
        input = (input - 0.5) / 0.5 # Simple norm assumption
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
    # Inputs: (B, 3, H, W)
    # H: (B, 1, H, W)
    # A: (B, 3, H, W)
    
    # Take first sample
    img = inputs[0].cpu().detach()
    ph = pred_h[0].cpu().detach().repeat(3,1,1) # 1->3 ch
    gh = gt_h[0].cpu().detach().repeat(3,1,1)
    pa = pred_a[0].cpu().detach()
    ga = gt_a[0].cpu().detach()
    
    row = torch.cat([img, ph, gh, pa, ga], dim=2)
    TF.to_pil_image(row).save(OUTPUT_DIR / f"preview_epoch_{epoch}.png")

def train():
    ds = V7Dataset(CACHED_DIR)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    model = MultiHeadUNet().to(DEVICE)
    scaler = torch.amp.GradScaler('cuda')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    criterion_L1 = nn.L1Loss()
    criterion_VGG = VGGPerceptualLoss().to(DEVICE)
    
    print("Starting Training V7.6...")
    
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(dl, desc=f"Epoch {epoch}")
        total_loss = 0
        
        for inputs, gt_h, gt_a in loop:
            inputs, gt_h, gt_a = inputs.to(DEVICE), gt_h.to(DEVICE), gt_a.to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                pred_h, pred_a = model(inputs)
                
                loss_h = criterion_L1(pred_h, gt_h)
                loss_a_l1 = criterion_L1(pred_a, gt_a)
                loss_a_vgg = criterion_VGG(pred_a, gt_a)
                
                loss = loss_h + loss_a_l1 + (0.1 * loss_a_vgg)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item(), h=loss_h.item(), a=loss_a_l1.item())
            
        print(f"Epoch {epoch} Loss: {total_loss/len(dl):.4f}")
        
        # Checkpoint & Preview
        if epoch % 1 == 0:
             save_preview(inputs, pred_h, gt_h, pred_a, gt_a, epoch)
             torch.save(model.state_dict(), CHECKPOINT_DIR / "latest.pth")

if __name__ == "__main__":
    train()
