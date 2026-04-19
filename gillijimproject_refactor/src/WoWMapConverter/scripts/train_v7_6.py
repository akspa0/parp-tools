from contextlib import nullcontext

import argparse
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

# ---------------------------------------------------------------------------
# Defaults (overridable via CLI)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _SCRIPT_DIR.parents[4]

DEFAULT_CACHED_DIR = Path("cached_v7_6")
DEFAULT_OUTPUT_DIR = _WORKSPACE_ROOT / "output" / "ml-training" / "v7_6"
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_EPOCHS = 100

# --- Dataset ---
class V7Dataset(Dataset):
    def __init__(self, cached_dir):
        self.files = sorted(Path(cached_dir).glob("input_*.pt"))
        print(f"Loaded {len(self.files)} samples.")
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        inp_path = self.files[idx]
        suffix = inp_path.stem[len("input_"):]

        tgt_h_path = self.files[idx].parent / f"target_height_{suffix}.pt"
        tgt_a_path = self.files[idx].parent / f"target_albedo_{suffix}.pt"
        
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
def save_preview(inputs, pred_h, gt_h, pred_a, gt_a, epoch, output_dir):
    img = inputs[0].cpu().detach()
    ph = pred_h[0].cpu().detach().repeat(3,1,1)
    gh = gt_h[0].cpu().detach().repeat(3,1,1)
    pa = pred_a[0].cpu().detach()
    ga = gt_a[0].cpu().detach()
    row = torch.cat([img, ph, gh, pa, ga], dim=2)
    TF.to_pil_image(row).save(Path(output_dir) / f"preview_epoch_{epoch}.png")

def parse_args():
    p = argparse.ArgumentParser(description="V7.6 dual-head minimap→height+albedo trainer.")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHED_DIR),
                   help=f"Cached dataset directory (default: {DEFAULT_CACHED_DIR})")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                   help=f"Output directory for checkpoints and previews.")
    p.add_argument("--resume", default="",
                   help="Path to checkpoint .pth to resume from.")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    return p.parse_args()

def train():
    args = parse_args()

    cached_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = V7Dataset(cached_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = MultiHeadUNet().to(device)
    use_cuda_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda_amp)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_loss = float("inf")
    if args.resume and Path(args.resume).exists():
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", {}))
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_loss = checkpoint.get("best_loss", float("inf"))
            print(f"Resumed from {args.resume} (epoch {start_epoch}, best_loss={best_loss:.4f})")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded weights from {args.resume}")

    criterion_L1 = nn.L1Loss()
    criterion_VGG = VGGPerceptualLoss().to(device)

    print(f"Starting V7.6 training: epochs={args.epochs} batch={args.batch_size} device={device} "
          f"cache={cached_dir} output={output_dir}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        loop = tqdm(dl, desc=f"Epoch {epoch}")
        total_loss = 0

        for inputs, gt_h, gt_a in loop:
            inputs, gt_h, gt_a = inputs.to(device), gt_h.to(device), gt_a.to(device)

            with (torch.amp.autocast('cuda') if use_cuda_amp else nullcontext()):
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

        avg_loss = total_loss / max(len(dl), 1)
        print(f"Epoch {epoch} avg_loss={avg_loss:.4f}")

        save_preview(inputs, pred_h, gt_h, pred_a, gt_a, epoch, output_dir)

        # Save latest
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
        }
        torch.save(ckpt, checkpoint_dir / "latest.pth")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, checkpoint_dir / "best.pth")
            print(f"  New best: {best_loss:.4f} → {checkpoint_dir / 'best.pth'}")

if __name__ == "__main__":
    train()
