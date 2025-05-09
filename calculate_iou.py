import os
import numpy as np
from skimage import io
from skimage.transform import resize

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

def evaluate_iou_for_last10():
    # Paths
    predictions_path = "data/membrane/train_last10/results"
    ground_truth_path = "data/membrane/train_last10/label"
    
    # Mapping between prediction and ground truth files
    # Since files are named differently (0_predict.png -> 20.png, etc.)
    mapping = {
        '0_predict.png': '20.png',
        '1_predict.png': '21.png',
        '2_predict.png': '22.png',
        '3_predict.png': '23.png',
        '4_predict.png': '24.png',
        '5_predict.png': '25.png',
        '6_predict.png': '26.png',
        '7_predict.png': '27.png',
        '8_predict.png': '28.png',
        '9_predict.png': '29.png'
    }
    
    ious = []
    print("Calculating IoU for last 10 training images...")
    
    for pred_file, gt_file in mapping.items():
        # Load prediction mask
        pred_path = os.path.join(predictions_path, pred_file)
        pred = io.imread(pred_path, as_gray=True)
        
        # Normalize prediction to [0,1] if needed
        if pred.max() > 1.0:
            pred = pred / 255.0
        
        # Load ground truth mask
        gt_path = os.path.join(ground_truth_path, gt_file)
        gt = io.imread(gt_path, as_gray=True)
        
        # Normalize ground truth to [0,1] if needed
        if gt.max() > 1.0:
            gt = gt / 255.0
        
        # Print shapes for debugging
        print(f"Shape of prediction {pred_file}: {pred.shape}")
        print(f"Shape of ground truth {gt_file}: {gt.shape}")
        
        # Resize if necessary
        if pred.shape != gt.shape:
            print(f"Resizing prediction to match ground truth shape")
            pred = resize(pred, gt.shape, order=0, preserve_range=True)
        
        # Calculate IoU
        iou = calculate_iou(pred, gt)
        ious.append(iou)
        print(f"IoU for {pred_file} and {gt_file}: {iou:.4f}")
    
    # Calculate average IoU
    mean_iou = np.mean(ious)
    print(f"\nAverage IoU across all 10 images: {mean_iou:.4f}")
    print(f"Best IoU: {np.max(ious):.4f}")
    print(f"Worst IoU: {np.min(ious):.4f}")
    
    return ious, mean_iou

if __name__ == "__main__":
    evaluate_iou_for_last10() 