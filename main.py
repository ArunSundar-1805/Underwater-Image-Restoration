import os
import cv2
import matplotlib.pyplot as plt
from CCCBLSHL import CCCBLSHL

def main():
    """
    Main function to run the underwater image restoration process.
    """
    images_dir = 'data/ruie/UCCS/blue'

    supported_extensions = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp')

    try:
        image_files = [
            f for f in os.listdir(images_dir) 
            if f.lower().endswith(supported_extensions)
        ]
    except FileNotFoundError:
        print(f"Error: Input directory not found at '{os.path.abspath(images_dir)}'")
        return

    for filename in image_files:
        input_path = os.path.join(images_dir, filename)
        Input = cv2.imread(input_path)

        if Input is None:
            print(f"Warning: Could not read image {filename}. Skipping.")
            continue

        print(f"Processing '{filename}'...")

        # Apply the underwater image restoration algorithm
        output = CCCBLSHL(Input)

        # Display the original and enhanced images
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(Input, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.title('Enhanced Image')
        plt.axis('off')

        print("Close the image window to continue to the next image...")
        plt.show()

if __name__ == '__main__':
    main()
