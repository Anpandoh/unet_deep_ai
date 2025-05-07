import os

import numpy as np
from skimage import io
from tensorflow.keras.models import load_model

from data import saveResult, testGenerator


def calculate_iou(pred_mask, true_mask):
    """
    Calculate Intersection over Union (IoU) between prediction and ground truth masks.
    Both masks should be binary (0 or 1).
    """
    # Ensure masks are binary
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    true_mask = (true_mask > 0.5).astype(np.uint8)

    # Calculate intersection and union
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()

    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    return iou


def evaluate_model():
    # Load the trained model
    model = load_model("unet_membrane.keras")
    print("Model loaded successfully!")

    # Get test data
    test_path = "data/membrane/test"
    testGene = testGenerator(test_path)

    # Load ground truth masks
    gt_path = "data/membrane/test/label"  # Adjust this path to your ground truth masks location

    # Predict and evaluate
    num_test_images = 30
    ious = []

    print("Evaluating model performance...")
    for i in range(num_test_images):
        # Get prediction
        img = next(testGene)
        pred = model.predict(img, verbose=0)[
            0, :, :, 0
        ]  # Get first image, remove batch and channel dimensions

        # Load ground truth mask
        gt_mask = io.imread(os.path.join(gt_path, f"{i}.png"), as_gray=True)
        gt_mask = gt_mask / 255.0  # Normalize to [0,1]

        # Calculate IoU
        iou = calculate_iou(pred, gt_mask)
        ious.append(iou)
        print(f"Image {i+1}/{num_test_images} - IoU: {iou:.4f}")

    # Calculate average IoU
    mean_iou = np.mean(ious)
    print(f"\nAverage IoU across all images: {mean_iou:.4f}")
    print(f"Best IoU: {np.max(ious):.4f}")
    print(f"Worst IoU: {np.min(ious):.4f}")


if __name__ == "__main__":
    evaluate_model()
