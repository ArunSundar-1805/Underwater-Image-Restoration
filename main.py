import os
import cv2
import matplotlib.pyplot as plt

# Assume the CCCBLSHL function is in a file named CCCBLSHL.py
from CCCBLSHL import CCCBLSHL

def main():
    """
    Main function to run the underwater image restoration process.
    """
    # Set the directory where the images are stored
    # Note: Using forward slashes is recommended for cross-platform compatibility
    images_dir = 'image/11_images'

    # Get a list of all bmp images in the directory
    # You can add other extensions like '.png', '.jpg' if needed
    supported_extensions = ('.bmp',)
    try:
        image_files = [
            f for f in os.listdir(images_dir) 
            if f.lower().endswith(supported_extensions)
        ]
    except FileNotFoundError:
        print(f"Error: Input directory not found at '{os.path.abspath(images_dir)}'")
        return

    # Loop over each image in the directory
    for filename in image_files:
        
        # Construct the full path to the input image
        input_path = os.path.join(images_dir, filename)
        
        # Read the input image
        # OpenCV reads images in BGR format by default
        Input = cv2.imread(input_path)
        
        if Input is None:
            print(f"Warning: Could not read image {filename}. Skipping.")
            continue
            
        print(f"Processing '{filename}'...")

        # Apply the underwater image restoration algorithm
        # This function is expected to take a BGR image and return a BGR image
        output = CCCBLSHL(Input)
        
        # Create a new figure to display the original and enhanced images
        plt.figure(figsize=(12, 6))
        
        # Display the original image in the first subplot
        plt.subplot(1, 2, 1)
        # Convert BGR (OpenCV) to RGB (Matplotlib) for correct color display
        plt.imshow(cv2.cvtColor(Input, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off') # Hide axes

        # Display the enhanced image in the second subplot
        plt.subplot(1, 2, 2)
        # Convert the BGR output to RGB for display
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.title('Enhanced Image')
        plt.axis('off') # Hide axes
        
        # In Python, plt.show() displays the figure and pauses execution
        # until the figure window is closed. This is the equivalent of 'pause'.
        print("Close the image window to continue to the next image...")
        plt.show()

if __name__ == '__main__':
    main()