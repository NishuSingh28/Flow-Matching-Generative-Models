import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from paper_unet import UNet


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return {k: v.detach().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.detach().clone() for k, v in state_dict.items()}


# =============================
# SEED
# =============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# =============================
# DATA
# =============================
transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(32),
    transforms.ToTensor(),
    lambda x: x * 2 - 1
])

dataset = ImageFolder(
    root='/home/nksingh9/CFM/data',
    transform=transform
)

# Limit to 50k (important)
dataset = Subset(dataset, list(range(min(50000, len(dataset)))))

# Split
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, _ = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
)

# =============================
# MODEL
# =============================
model = UNet().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
ema = EMA(model, decay=0.999)

# =============================
# CHECKPOINT
# =============================
ckpt_path = 'ot_ckpt.pt'
save_path = 'ot_model.pt'
loss_history = []
start = 0
NUM_EPOCHS = 200

if os.path.exists(ckpt_path):
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt.get('ema', ckpt['model'])) 
        opt.load_state_dict(ckpt['opt'])
        loss_history = ckpt.get('loss_history', [])
        start = ckpt.get('epoch', -1) + 1
        if 'ema' in ckpt:
            ema.load_state_dict(ckpt['ema'])
        print(f"Resumed from epoch {start}")
    except Exception as e:
        print(f"Could not resume OT checkpoint ({e}); starting fresh.")
        loss_history = []
        start = 0

# =============================
# TRAIN
# =============================
for epoch in range(start, NUM_EPOCHS):
    model.train()
    total = 0.0

    for x, _ in loader:
        x = x.to(DEVICE, non_blocking=True)

        x0 = torch.randn_like(x)
        t = torch.rand(x.size(0), device=DEVICE)
        t_img = t[:, None, None, None]

        xt = (1 - (1 - 1e-4) * t_img) * x0 + t_img * x
        target = x - (1 - 1e-4) * x0

        pred = model(xt, t)
        loss = F.mse_loss(pred, target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)

        total += loss.item()

    sched.step()
    avg = total / len(loader)
    loss_history.append(avg)
    print(epoch, avg)

    torch.save(
        {
            'model': model.state_dict(),
            'ema': ema.state_dict(),
            'opt': opt.state_dict(),
            'epoch': epoch,
            'loss_history': loss_history,
        },
        ckpt_path,
    )

# =============================
# SAVE FINAL MODEL
# =============================
torch.save(ema.state_dict(), save_path)