import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

def _rgb2gray_matlab(rgb_image):
    """ A manual implementation of rgb2gray using MATLAB's specific coefficients. """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel RGB image.")
    coeffs = np.array([0.2989, 0.5870, 0.1140])
    return np.dot(rgb_image[..., :3], coeffs)

def wls_optimization(in_image, data_weight, guidance, lambda_val=None):
    """ Weighted Least Squares optimization solver. """
    small_num = 1e-5
    if lambda_val is None:
        lambda_val = 0.05

    in_image = in_image.astype(np.float64)
    data_weight = data_weight.astype(np.float64)
    guidance = guidance.astype(np.float64)

    h, w = in_image.shape
    k = h * w
    
    if guidance.ndim == 3:
        guidance_gray = _rgb2gray_matlab(guidance)
    else:
        guidance_gray = guidance

    dy = np.diff(guidance_gray, 1, axis=0)
    dy = -lambda_val / (np.abs(dy)**2 + small_num)
    dy = np.pad(dy, ((0, 1), (0, 0)), 'constant')
    dy = dy.flatten(order='F')

    dx = np.diff(guidance_gray, 1, axis=1)
    dx = -lambda_val / (np.abs(dx)**2 + small_num)
    dx = np.pad(dx, ((0, 0), (0, 1)), 'constant')
    dx = dx.flatten(order='F')
    
    B = np.vstack((dx, dy)).T
    d = np.array([-h, -1])
    tmp = spdiags(B.T, d, k, k)
    
    ea = dx
    we = np.pad(dx, (h, 0), 'constant')[:-h]
    so = dy
    no = np.pad(dy, (1, 0), 'constant')[:-1]
    
    D = -(ea + we + so + no)
    Asmoothness = tmp + tmp.T + spdiags(D, 0, k, k)
    
    data_weight = data_weight - np.min(data_weight)
    data_weight = data_weight / (np.max(data_weight) + small_num)
    
    reliability_mask = data_weight[0, :] < 0.6
    in_row1 = np.min(in_image, axis=0)
    data_weight[0, reliability_mask] = 0.8
    in_image[0, reliability_mask] = in_row1[reliability_mask]
    
    Adata = spdiags(data_weight.flatten(order='F'), 0, k, k)
    
    A = Adata + Asmoothness
    b = Adata @ in_image.flatten(order='F')
    
    # --- CORRECTED SECTION ---
    # Convert matrix to CSC format to resolve the efficiency warning
    A = A.tocsc()
    # --- END CORRECTED SECTION ---
    
    out_flat = spsolve(A, b)
    out = out_flat.reshape((h, w), order='F')

    return out