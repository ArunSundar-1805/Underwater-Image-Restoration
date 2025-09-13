import os
import cv2
import numpy as np
import torch
from CCCBLSHL import CCCBLSHL
import pyiqa

# --------------------------
# Custom UCIQE and UIQM
# --------------------------
def UCIQE(img):
    img = img.astype(np.float32) / 255.0
    I = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, A, B = I[:, :, 0], I[:, :, 1], I[:, :, 2]
    stdL = np.std(L)
    S = np.sqrt(A**2 + B**2)
    meanS = np.mean(S)
    H = np.arctan2(B, A)
    stdH = np.std(H)
    return 0.4680 * stdL + 0.2745 * meanS + 0.2576 * stdH

def UIQM(img):
    img = img.astype(np.float32)
    rg = img[:, :, 0] - img[:, :, 1]
    yb = 0.5 * (img[:, :, 0] + img[:, :, 1]) - img[:, :, 2]
    stdRoot = np.sqrt(np.std(rg)**2 + np.std(yb)**2)
    meanRoot = np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    colorfulness = stdRoot + 0.3 * meanRoot
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = np.std(gray)
    return 0.0282 * colorfulness + 0.2953 * sharpness + 3.5753 * contrast

# --------------------------
# Initialize pyiqa models
# --------------------------
brisque_model = pyiqa.create_metric('brisque', as_loss=False)
niqe_model = pyiqa.create_metric('niqe', as_loss=False)

# --------------------------
# Evaluate metrics
# --------------------------
def evaluate_metrics(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_tensor = torch.from_numpy(gray.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0)

    metrics = {
        'UCIQE': UCIQE(img_rgb),
        'UIQM': UIQM(img_rgb),
        'BRISQUE': brisque_model(img_tensor).item(),
        'NIQE': niqe_model(img_tensor).item()
    }
    return metrics

# --------------------------
# Main function
# --------------------------
def main():
    images_dir = 'image/11_Images'
    supported_extensions = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp')

    # Ask user for output directory
    save_dir = 'Results'
    os.makedirs(save_dir, exist_ok=True)

    try:
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(supported_extensions)]
    except FileNotFoundError:
        print(f"Error: Input directory not found at '{os.path.abspath(images_dir)}'")
        return

    if not image_files:
        print("No supported images found in the directory.")
        return

    # Store metrics for computing overall improvement
    total_orig = {'UCIQE':0, 'UIQM':0, 'BRISQUE':0, 'NIQE':0}
    total_restored = {'UCIQE':0, 'UIQM':0, 'BRISQUE':0, 'NIQE':0}
    num_images = 0

    # Print table header
    print(f"{'Image':30} {'UCIQE(Orig)':>12} {'UCIQE(Rest)':>12} {'UIQM(Orig)':>12} {'UIQM(Rest)':>12} {'BRISQUE(Orig)':>14} {'BRISQUE(Rest)':>14} {'NIQE(Orig)':>12} {'NIQE(Rest)':>12}")
    print("-"*120)

    for filename in image_files:
        input_path = os.path.join(images_dir, filename)
        Input = cv2.imread(input_path)

        if Input is None:
            print(f"Warning: Could not read image {filename}. Skipping.")
            continue

        # Evaluate original image
        orig_metrics = evaluate_metrics(Input)

        # Apply underwater image enhancement
        output = CCCBLSHL(Input)

        # Save restored image
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, output)

        # Evaluate restored image
        restored_metrics = evaluate_metrics(output)

        # Print per-image metrics
        print(f"{filename:30} "
              f"{orig_metrics['UCIQE']:12.4f} {restored_metrics['UCIQE']:12.4f} "
              f"{orig_metrics['UIQM']:12.4f} {restored_metrics['UIQM']:12.4f} "
              f"{orig_metrics['BRISQUE']:14.4f} {restored_metrics['BRISQUE']:14.4f} "
              f"{orig_metrics['NIQE']:12.4f} {restored_metrics['NIQE']:12.4f}")

        # Accumulate for overall improvement
        for key in total_orig:
            total_orig[key] += orig_metrics[key]
            total_restored[key] += restored_metrics[key]
        num_images += 1

    # Compute average metrics
    print("\nOverall average metrics:")
    print(f"{'Metric':>10} {'Original Avg':>14} {'Restored Avg':>14} {'Improvement':>14}")
    print("-"*60)
    for key in total_orig:
        orig_avg = total_orig[key]/num_images
        rest_avg = total_restored[key]/num_images
        # Improvement = Restored - Original (for UCIQE/UIQM, higher is better; for BRISQUE/NIQE, lower is better)
        if key in ['BRISQUE', 'NIQE']:
            improvement = orig_avg - rest_avg
        else:
            improvement = rest_avg - orig_avg
        print(f"{key:>10} {orig_avg:14.4f} {rest_avg:14.4f} {improvement:14.4f}")

    print(f"\nAll restored images saved in '{os.path.abspath(save_dir)}'")

if __name__ == '__main__':
    main()
