# ====================================================
# PyTorch re‑implementation of the original Keras U‑Net
# (Ronneberger et al., 2015) membrane segmentation demo
# ====================================================
# Only the *data_torch.py* section has changed in this revision.
#   • Fixed Albumentations warning (Affine `mode` parameter)
#   • Added robust path resolution so relative paths work no
#     matter where `python train_torch.py` is launched.
# ====================================================
# ------------------- data_torch.py ------------------
import os, glob, numpy as np, torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

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


# ---- Albumentations transforms (warning‑free) ----


def get_training_transform():
    return A.Compose(
        [
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(0.0, 0.05),
                rotate=(-18, 18),
                shear=(-5, 5),
                mode="constant",
                p=0.7,
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
    """Pairs image and mask.
    *Path handling now checks both CWD and the directory that
    contains this file, so relative paths always work.*"""

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
        # Resolve root path robustly
        root_path = Path(root_dir).expanduser()
        if not root_path.is_absolute():
            script_dir = Path(__file__).resolve().parent
            root_path = (script_dir / root_path).resolve()
        self.root_dir = root_path
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform
        self.flag_multi_class = flag_multi_class
        self.num_class = num_class
        self.as_gray = as_gray

        if image_folder:
            image_pattern = self.root_dir / image_folder / "*.png"
        else:
            image_pattern = self.root_dir / "*.png"
        self.image_paths = sorted(glob.glob(str(image_pattern)))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found with pattern: {image_pattern}")

        # build mask paths only if mask_folder provided
        if mask_folder:
            self.mask_paths = [
                str(
                    Path(p).with_name(f"{Path(p).stem}.png").parent.parent
                    / mask_folder
                    / f"{Path(p).stem}.png"
                )
                for p in self.image_paths
            ]
        else:
            self.mask_paths = [None] * len(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = np.array(Image.open(img_path).convert("L" if self.as_gray else "RGB"))
        if self.mask_paths[idx]:
            mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        else:
            mask = np.zeros_like(img)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]
        else:
            img, mask = adjust_data(img, mask, self.flag_multi_class, self.num_class)
            img = torch.from_numpy(img).unsqueeze(0)
            mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask


# -------------- test generator (lazy) ---------------


def test_generator(
    test_path: str, num_image: int = 30, target_size=(256, 256), as_gray: bool = True
):
    from torchvision.transforms.functional import resize

    test_root = Path(test_path).expanduser()
    if not test_root.is_absolute():
        test_root = (Path(__file__).resolve().parent / test_root).resolve()
    for i in range(num_image):
        img_path = test_root / f"{i}.png"
        if not img_path.exists():
            raise FileNotFoundError(img_path)
        img = Image.open(img_path).convert("L" if as_gray else "RGB")
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
    save_root = Path(save_path).expanduser()
    if not save_root.is_absolute():
        save_root = (Path(__file__).resolve().parent / save_root).resolve()
    save_root.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(npyfile):
        if flag_multi_class:
            img = label_visualize(num_class, COLOR_DICT, item)
        else:
            img = (item.squeeze() * 255).astype(np.uint8)
        Image.fromarray(img).save(save_root / f"{i}_predict.png")
