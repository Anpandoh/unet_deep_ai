# ====================================================
# train_torch.py — fixed for channel‑mismatch error
# ====================================================
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from data_torch import (
    MembraneDataset,
    get_training_transform,
    get_simple_transform,
    save_result,
)
from model_torch import UNet


def pixel_accuracy(logits, masks):
    """Return accuracy for a batch (0‑1 float)."""
    preds = (torch.sigmoid(logits) > 0.5).float()
    correct = (preds == masks).sum()
    total = masks.numel()
    return correct / total


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, running_acc = 0.0, 0.0

    for imgs, masks in tqdm(loader, desc="Training", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        masks = masks.float()

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += pixel_accuracy(logits, masks).item()

    n_batches = len(loader)
    return running_loss / n_batches, running_acc / n_batches


def predict(model, loader, device):
    """Yield binary numpy predictions batch‑wise."""
    model.eval()
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Infer", leave=False):
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs)).squeeze().cpu().numpy()
            yield (probs > 0.5).astype("uint8")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------- DATA ----------------------
    train_ds = MembraneDataset(
        "data/membrane/train", transform=get_training_transform()
    )
    test_ds = MembraneDataset(
        "data/membrane/test",
        image_folder="",
        mask_folder="",
        transform=get_simple_transform(),
    )

    train_loader = DataLoader(
        train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=True
    )  # 2 workers to satisfy Colab warning
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # --------------------- MODEL ----------------------
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    # --------------------- TRAIN ----------------------
    best_loss = float("inf")
    for epoch in range(5):  # mirrors original demo
        avg_loss, avg_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}  pixel‑acc={avg_acc:.4f}")

        print(f"Epoch {epoch+1}: mean loss = {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "unet_membrane.pt")
            print("  ↳ saved checkpoint: unet_membrane.pt")

    # -------------------- INFER ----------------------
    preds = list(predict(model, test_loader, device))
    save_result("data/membrane/test", preds)
    print("Inference PNGs written to data/membrane/test/")


if __name__ == "__main__":
    main()
