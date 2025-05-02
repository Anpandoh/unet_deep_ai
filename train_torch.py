# ====================================================
# PyTorch re‑implementation of the original Keras U‑Net
# (Ronneberger et al., 2015) membrane segmentation demo
# ====================================================
# Tested with:
#   torch==2.2.1
#   torchvision==0.17.1
#   albumentations==1.4.0
#   scikit-image==0.23.3
#   pillow==10.3.0
# ----------------------------------------------------
# Project layout keeps the spirit of the original:
#   data_torch.py   – dataset, augmentation, helpers
#   model_torch.py  – U‑Net architecture
#   train_torch.py  – training / inference script
# ----------------------------------------------------
# >>> python train_torch.py
# ====================================================
# ------------------- data_torch.py ------------------
import os, glob, numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.utils import save_image

# RGB palette identical to original code
Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]
COLOR_DICT = np.array(
    [
        Sky,
        Building,
        Pole,
        Road,
        Pavement,
        Tree,
        SignSymbol,
        Fence,
        Car,
        Pedestrian,
        Bicyclist,
        Unlabelled,
    ],
    dtype=np.uint8,
)

# ---- Helper that preserves original adjustData semantics ----
# (Normalise image, one‑hot or binarise mask)


def adjust_data(
    img: np.ndarray, mask: np.ndarray, flag_multi_class: bool, num_class: int
):
    if flag_multi_class:
        img = img / 255.0
        new_mask = np.zeros(mask.shape + (num_class,), dtype=np.float32)
        for i in range(num_class):
            new_mask[mask == i, i] = 1
        mask = new_mask.reshape(-1, num_class)
    else:
        img = img / 255.0
        mask = mask / 255.0
        mask = (mask > 0.5).astype(np.float32)
    return img.astype(np.float32), mask.astype(np.float32)


# ---- Albumentations transform mirroring the old ImageDataGenerator ----


def get_training_transform():
    return A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=18,
                shear_limit=0.05,
                p=0.7,
                border_mode=0,
            ),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2(),
        ]
    )


def get_simple_transform():
    return A.Compose([A.Normalize(mean=0.0, std=1.0), ToTensorV2()])


# ----------------  custom Dataset -------------------
class MembraneDataset(Dataset):
    """Pairs image and mask from directory tree identical to Keras demo."""

    def __init__(
        self,
        root_dir: str,
        image_folder="image",
        mask_folder="label",
        transform=None,
        flag_multi_class=False,
        num_class=2,
        as_gray=True,
    ):
        self.image_paths = sorted(
            glob.glob(os.path.join(root_dir, image_folder, "*.png"))
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {root_dir}/{image_folder}")
        self.mask_paths = [
            p.replace(image_folder, mask_folder) for p in self.image_paths
        ]
        self.transform = transform
        self.flag_multi_class = flag_multi_class
        self.num_class = num_class
        self.as_gray = as_gray

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = np.array(
            Image.open(self.image_paths[idx]).convert("L" if self.as_gray else "RGB")
        )
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]
        else:
            img, mask = adjust_data(img, mask, self.flag_multi_class, self.num_class)
            img = torch.from_numpy(img).unsqueeze(0)  # C=1
            mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask


# -------------- test generator (lazy) ---------------


def test_generator(
    test_path: str, num_image: int = 30, target_size=(256, 256), as_gray: bool = True
):
    from torchvision.transforms.functional import resize

    for i in range(num_image):
        img = Image.open(os.path.join(test_path, f"{i}.png")).convert(
            "L" if as_gray else "RGB"
        )
        img = resize(img, target_size)
        img = (
            torch.tensor(np.array(img) / 255.0, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        yield img


# -------------- colourise prediction ----------------


def label_visualize(num_class: int, color_dict: np.ndarray, img: np.ndarray):
    if img.ndim == 3:
        img = img[:, :, 0]
    img_out = np.zeros(img.shape + (3,), dtype=np.uint8)
    for i in range(num_class):
        img_out[img == i] = color_dict[i]
    return img_out


def save_result(
    save_path: str, npyfile, flag_multi_class: bool = False, num_class: int = 2
):
    os.makedirs(save_path, exist_ok=True)
    for i, item in enumerate(npyfile):
        if flag_multi_class:
            img = label_visualize(num_class, COLOR_DICT, item)
        else:
            img = (item.squeeze() * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(save_path, f"{i}_predict.png"))


# ----------------- model_torch.py -------------------
import torch.nn as nn, torch


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.c1 = ConvBlock(in_ch, 64)
        self.p1 = nn.MaxPool2d(2)
        self.c2 = ConvBlock(64, 128)
        self.p2 = nn.MaxPool2d(2)
        self.c3 = ConvBlock(128, 256)
        self.p3 = nn.MaxPool2d(2)
        self.c4 = ConvBlock(256, 512)
        self.dropout4 = nn.Dropout(0.5)
        self.p4 = nn.MaxPool2d(2)
        self.c5 = ConvBlock(512, 1024)
        self.dropout5 = nn.Dropout(0.5)
        # Expansive path
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.c6 = ConvBlock(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.c7 = ConvBlock(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.c8 = ConvBlock(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.c9 = ConvBlock(128, 64)
        self.c9_out = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(2, out_ch, kernel_size=1)

    def forward(self, x):
        c1 = self.c1(x)
        p1 = self.p1(c1)
        c2 = self.c2(p1)
        p2 = self.p2(c2)
        c3 = self.c3(p2)
        p3 = self.p3(c3)
        c4 = self.c4(p3)
        c4d = self.dropout4(c4)
        p4 = self.p4(c4d)
        c5 = self.dropout5(self.c5(p4))
        up6 = torch.relu(self.up6(c5))
        merge6 = torch.cat([c4d, up6], dim=1)
        c6 = self.c6(merge6)
        up7 = torch.relu(self.up7(c6))
        merge7 = torch.cat([c3, up7], dim=1)
        c7 = self.c7(merge7)
        up8 = torch.relu(self.up8(c7))
        merge8 = torch.cat([c2, up8], dim=1)
        c8 = self.c8(merge8)
        up9 = torch.relu(self.up9(c8))
        merge9 = torch.cat([c1, up9], dim=1)
        c9 = self.c9(merge9)
        c9_out = self.c9_out(c9)
        return self.out_conv(c9_out)


# ----------------- train_torch.py -------------------
import torch, os, numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_torch import (
    MembraneDataset,
    get_training_transform,
    get_simple_transform,
    save_result,
)
from model_torch import UNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Training Loader
    train_ds = MembraneDataset(
        "data/membrane/train", transform=get_training_transform()
    )
    train_loader = DataLoader(
        train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True
    )
    # Model
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    epochs = 1  # to mirror original example
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, masks = imgs.to(device), masks.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running += loss.item()
        avg_loss = running / len(train_loader)
        print(f"Epoch {epoch+1}: loss = {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "unet_membrane.pt")
    # Inference – replicate testGenerator/predict_generator
    model.eval()
    preds = []
    test_iter = MembraneDataset(
        "data/membrane/test",
        image_folder="",
        mask_folder="",
        transform=get_simple_transform(),
    )
    test_loader = DataLoader(test_iter, batch_size=1, shuffle=False)
    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc="Infer"):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            preds.append((probs > 0.5).astype(np.uint8))
    save_result("data/membrane/test", preds)


if __name__ == "__main__":
    main()

