import numpy as np

def Grayscale_remapping(image):
    """
    Grayscale World White Balance and Histogram Auto-Contrast Adjustment.
    A literal Python translation of the MATLAB script, using explicit loops.
    
    Args:
        image (np.array): The input image as a NumPy array (H x W x C).
                          
    Returns:
        np.array: The processed output image.
    """
    # Ensure the image is in float format for calculations
    image_float = image.astype(np.float64)

    # Find the maximum values of each RGB channel
    V = np.max(image_float, axis=(0, 1))
    
    # Find the overall maximum value
    max_1 = np.max(V)

    # Define the ratio for the saturation level
    ratio = np.array([1, 1, 1])

    # Set the saturation level for quantile adjustment
    satLevel = 0.001 * ratio

    # Get the size of the image
    m, n, p = image_float.shape
    
    # Reshape each RGB channel into a row vector using a for loop, like MATLAB
    imgRGB_orig = np.zeros((p, m * n), dtype=np.float64)
    for i in range(p):
       imgRGB_orig[i, :] = image_float[:, :, i].flatten()

    imRGB = np.zeros_like(imgRGB_orig)

    # Histogram contrast adjustment for each channel
    for ch in range(p):
        # Define quantile range for contrast adjustment
        q = np.array([satLevel[ch], 1 - satLevel[ch]])
        
        # Calculate quantiles for the current channel
        tiles = np.quantile(imgRGB_orig[ch, :], q)
        
        # Make a copy to modify
        temp = imgRGB_orig[ch, :].copy()
        
        # Clip pixel values using boolean indexing, like MATLAB
        temp[temp < tiles[0]] = tiles[0]
        temp[temp > tiles[1]] = tiles[1]
        
        # Store adjusted values
        imRGB[ch, :] = temp
        
        # Normalize the pixel values to the range [0, max_1]
        pmin = np.min(imRGB[ch, :])
        pmax = np.max(imRGB[ch, :])

        # Perform the division directly, without a safety check, to match MATLAB's behavior.
        # This will produce NaN/inf if pmax == pmin, just like MATLAB.
        imRGB[ch, :] = (imRGB[ch, :] - pmin) / (pmax - pmin) * max_1

    # Initialize the output image
    output = np.zeros_like(image_float)

    # Reshape and assign the adjusted RGB values back to the output image using a for loop
    for i in range(p):
        output[:, :, i] = imRGB[i, :].reshape(m, n)

    return output