import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchdiffeq import odeint
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split


from paper_unet import UNet

# =============================
# SEED
# =============================
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# LOAD CHECKPOINTS
# =============================
ot_ckpt = torch.load("ot_ckpt.pt", map_location=DEVICE)
diff_ckpt = torch.load("diff_ckpt.pt", map_location=DEVICE)

ot_model = UNet().to(DEVICE)
ot_model.load_state_dict(ot_ckpt.get("ema", ot_ckpt["model"]))
ot_model.eval()

diff_model = UNet().to(DEVICE)
diff_model.load_state_dict(diff_ckpt.get("ema", diff_ckpt["model"]))
diff_model.eval()

# =============================
# LOSS PLOT
# =============================
plt.figure(figsize=(8, 5))
plt.plot(ot_ckpt["loss_history"], label="OT-CFM")
plt.plot(diff_ckpt["loss_history"], label="Diffusion-CFM")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.grid()
plt.savefig("loss_comparison.png")
plt.show()

print("Saved loss plot")

# =============================
# ADAPTIVE ODE SAMPLER
# =============================
def sample_ode(model, init_noise):
    nfe = 0

    def ode_func(t, x):
        nonlocal nfe
        nfe += 1
        t_batch = torch.ones(x.size(0), device=x.device) * t
        return model(x, t_batch)

    t_span = torch.tensor([0.0, 1.0], device=DEVICE)

    with torch.no_grad():
        x = odeint(
            ode_func,
            init_noise,
            t_span,
            method="dopri5",
            rtol=1e-5,
            atol=1e-5,
        )[-1]

    return x.clamp(-1, 1), nfe

# =============================
# GENERATE SAMPLES
# =============================
batch_size = 64
init_noise = torch.randn(batch_size, 3, 32, 32).to(DEVICE)

ot_samples, ot_nfe = sample_ode(ot_model, init_noise.clone())
diff_samples, diff_nfe = sample_ode(diff_model, init_noise.clone())

# Visualization only
ot_vis = F.interpolate(ot_samples, size=128, mode='bilinear', align_corners=False)
diff_vis = F.interpolate(diff_samples, size=128, mode='bilinear', align_corners=False)

save_image((ot_vis + 1) / 2, "ot_samples.png", nrow=8)
save_image((diff_vis + 1) / 2, "diff_samples.png", nrow=8)

print(f"OT NFE: {ot_nfe}")
print(f"Diffusion NFE: {diff_nfe}")

# =============================
# FID SETUP
# =============================

transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)
])

dataset = ImageFolder(
    root='/home/nksingh9/CFM/data',
    transform=transform
)

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

_, test_dataset = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

real_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=torch.cuda.is_available()
)

# =============================
# FID PREPROCESS
# =============================
def preprocess_for_fid(x):
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=299, mode='bilinear', align_corners=False)
    x = (x + 1) / 2
    x = (x * 255).clamp(0, 255)
    return x.to(torch.uint8)

# =============================
# FID COMPUTATION
# =============================
def compute_fid(model):
    fid = FrechetInceptionDistance(feature=2048).to(DEVICE)
    fid.eval()

    with torch.no_grad():
        for real, _ in real_loader:
            real = preprocess_for_fid(real.to(DEVICE))
            fid.update(real, real=True)

        total = 0
        while total < 50000:
            bs = min(64, 50000 - total)
            z = torch.randn(bs, 3, 32, 32).to(DEVICE)
            gen, _ = sample_ode(model, z)
            gen = preprocess_for_fid(gen)
            fid.update(gen, real=False)
            total += bs

    return fid.compute().item()

print("Computing FID...")

ot_fid = compute_fid(ot_model)
diff_fid = compute_fid(diff_model)

# =============================
# FINAL RESULTS
# =============================
print("\n===== FINAL COMPARISON =====")
print(f"OT-CFM     → FID: {ot_fid:.3f}, NFE: {ot_nfe}")
print(f"Diffusion  → FID: {diff_fid:.3f}, NFE: {diff_nfe}")

print("\nSaved outputs:")
print("- loss_comparison.png")
print("- ot_samples.png")
print("- diff_samples.png")