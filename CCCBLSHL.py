import numpy as np
import cv2
from scipy import ndimage
from scipy.signal import convolve2d
from skimage.color import rgb2lab, lab2rgb
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from sklearn.cluster import KMeans

# Import the previously translated custom functions
from Grayscale_remapping import Grayscale_remapping
from estimate_airlight import estimate_airlight
from wls_optimization import wls_optimization

def _robust_accumarray_op(data, labels, index, func, fill_value=0):
    """ Helper function to safely perform accumarray-like operations. """
    labels_flat = labels.ravel()
    if not np.all(index == np.arange(1, len(index) + 1)):
        raise ValueError("Index must be a contiguous array starting from 1.")

    counts = np.bincount(labels_flat, minlength=len(index) + 1)[index]
    valid_mask = counts > 0
    result = np.full(len(index), fill_value, dtype=float)

    if func == 'mean':
        sums = ndimage.sum_labels(data, labels=labels_flat, index=index)
        result[valid_mask] = sums[valid_mask] / counts[valid_mask]
    elif func == 'std':
        sums = ndimage.sum_labels(data, labels=labels_flat, index=index)
        mean_val = np.zeros_like(sums, dtype=float)
        mean_val[valid_mask] = sums[valid_mask] / counts[valid_mask]
        sum_sq = ndimage.sum_labels(data**2, labels=labels_flat, index=index)
        variance = np.zeros_like(sums, dtype=float)
        variance[valid_mask] = (sum_sq[valid_mask] / counts[valid_mask]) - (mean_val[valid_mask]**2)
        result = np.sqrt(np.maximum(variance, 0))
    elif func == 'max':
        result = ndimage.maximum(data, labels=labels_flat, index=index)
    return result

def CCCBLSHL(Input):
    im_c = Input.astype(np.float64) / 255.0
    m, n, c = im_c.shape
    im_c_remap = Grayscale_remapping(im_c)

    sum_r, sum_g, sum_b = np.sum(im_c, axis=(0, 1))

    if sum_r > max(sum_g, sum_b):
        im_c_remap[:, :, 1:3] = im_c[:, :, 1:3]
    elif sum_g > max(sum_r, sum_b):
        im_c_remap[:, :, [0, 2]] = im_c[:, :, [0, 2]]
    else:
        im_c_remap[:, :, 0:2] = im_c[:, :, 0:2]

    yy, xx = np.mgrid[0:m, 0:n]
    coords_flat_y = yy.flatten()
    coords_flat_x = xx.flatten()
    remap_flat = im_c_remap.reshape(-1, 3)
    orig_flat = im_c.reshape(-1, 3)

    im_c_r_x = np.column_stack((remap_flat[:, 0], coords_flat_y, coords_flat_x, orig_flat[:, 0]))
    im_c_g_x = np.column_stack((remap_flat[:, 1], coords_flat_y, coords_flat_x, orig_flat[:, 1]))
    im_c_b_x = np.column_stack((remap_flat[:, 2], coords_flat_y, coords_flat_x, orig_flat[:, 2]))

    im_c_r_x = im_c_r_x[im_c_r_x[:, 3].argsort()]
    im_c_g_x = im_c_g_x[im_c_g_x[:, 3].argsort()]
    im_c_b_x = im_c_b_x[im_c_b_x[:, 3].argsort()]

    with np.errstate(divide='ignore', invalid='ignore'):
        if sum_r > max(sum_g, sum_b):
            gain = im_c_r_x[:, 0] / im_c_r_x[:, 3]
            gain[np.isnan(gain)] = 0
            a1 = np.max(im_c_r_x[:, 0]) / np.max(im_c_g_x[:, 3])
            a2 = np.max(im_c_r_x[:, 0]) / np.max(im_c_b_x[:, 3])
            for i in range(m * n):
                row_g, col_g = int(im_c_g_x[i, 1]), int(im_c_g_x[i, 2])
                im_c_remap[row_g, col_g, 1] = im_c_g_x[i, 0] * gain[i] * a1
                row_b, col_b = int(im_c_b_x[i, 1]), int(im_c_b_x[i, 2])
                im_c_remap[row_b, col_b, 2] = im_c_b_x[i, 0] * gain[i] * a2
        elif sum_g > max(sum_r, sum_b):
            gain = im_c_g_x[:, 0] / im_c_g_x[:, 3]
            gain[np.isnan(gain)] = 0
            a1 = np.max(im_c_g_x[:, 0]) / np.max(im_c_r_x[:, 3])
            a2 = np.max(im_c_g_x[:, 0]) / np.max(im_c_b_x[:, 3])
            for i in range(m * n):
                row_r, col_r = int(im_c_r_x[i, 1]), int(im_c_r_x[i, 2])
                im_c_remap[row_r, col_r, 0] = im_c_r_x[i, 0] * gain[i] * a1
                row_b, col_b = int(im_c_b_x[i, 1]), int(im_c_b_x[i, 2])
                im_c_remap[row_b, col_b, 2] = im_c_b_x[i, 0] * gain[i] * a2
        else: # sum_b is dominant
            gain = im_c_b_x[:, 0] / im_c_b_x[:, 3]
            gain[np.isnan(gain)] = 0
            a1 = np.max(im_c_b_x[:, 0]) / np.max(im_c_r_x[:, 3])
            a2 = np.max(im_c_b_x[:, 0]) / np.max(im_c_g_x[:, 3])
            for i in range(m * n):
                row_r, col_r = int(im_c_r_x[i, 1]), int(im_c_r_x[i, 2])
                im_c_remap[row_r, col_r, 0] = im_c_r_x[i, 0] * gain[i] * a1
                row_g, col_g = int(im_c_g_x[i, 1]), int(im_c_g_x[i, 2])
                im_c_remap[row_g, col_g, 1] = im_c_g_x[i, 0] * gain[i] * a2
    
    im_c = im_c_remap
    sum_r, sum_g, sum_b = np.sum(im_c, axis=(0, 1))
    Thr, alpha, beta = 10, 0.8, 0.2

    if (sum_g > sum_r) and (sum_g > sum_b):
        g_r = min(sum_g / (sum_r + 1e-9), Thr)
        g_b = min(sum_g / (sum_b + 1e-9), Thr)
        im_c[:, :, 0] = im_c[:, :, 0] * alpha + (g_r - alpha - beta) * sum_r * im_c[:, :, 1] / (sum_g + 1e-9) + beta * sum_r / (m * n)
        im_c[:, :, 2] = im_c[:, :, 2] * alpha + (g_b - alpha - beta) * sum_b * im_c[:, :, 1] / (sum_g + 1e-9) + beta * sum_b / (m * n)
    elif (sum_r > sum_g) and (sum_r > sum_b):
        r_g = min(sum_r / (sum_g + 1e-9), Thr)
        r_b = min(sum_r / (sum_b + 1e-9), Thr)
        im_c[:, :, 1] = im_c[:, :, 1] * alpha + (r_g - alpha - beta) * sum_g * im_c[:, :, 0] / (sum_r + 1e-9) + beta * sum_g / (m * n)
        im_c[:, :, 2] = im_c[:, :, 2] * alpha + (r_b - alpha - beta) * sum_b * im_c[:, :, 0] / (sum_r + 1e-9) + beta * sum_b / (m * n)
    else: # sum_b is dominant or equal
        b_r = min(sum_b / (sum_r + 1e-9), Thr)
        b_g = min(sum_b / (sum_g + 1e-9), Thr)
        im_c[:, :, 0] = im_c[:, :, 0] * alpha + (b_r - alpha - beta) * sum_r * im_c[:, :, 2] / (sum_b + 1e-9) + beta * sum_r / (m * n)
        im_c[:, :, 1] = im_c[:, :, 1] * alpha + (b_g - alpha - beta) * sum_g * im_c[:, :, 2] / (sum_b + 1e-9) + beta * sum_g / (m * n)

    im_c = im_c / (np.max(im_c) + 1e-9)
    im_c = np.clip(im_c, 0, 1)

    A = estimate_airlight(im_c).reshape(1, 1, 3)
    
    delt_I_reshape = im_c.reshape(m * n, c)
    delt_I_reshape_H = np.hstack((delt_I_reshape, np.ones((m * n, 1))))
    A_reshape = A.reshape(3, 1)

    a, b = 90, -180
    Rx = np.array([[1, 0, 0], [0, np.cos(np.deg2rad(a)), -np.sin(np.deg2rad(a))], [0, np.sin(np.deg2rad(a)), np.cos(np.deg2rad(a))]])
    Ry = np.array([[np.cos(np.deg2rad(b)), 0, np.sin(np.deg2rad(b))], [0, 1, 0], [-np.sin(np.deg2rad(b)), 0, np.cos(np.deg2rad(b))]])
    RT = np.hstack((Ry @ Rx, A_reshape))

    delt_I_Rota = np.abs((RT @ delt_I_reshape_H.T).T)
    I_Rota = delt_I_Rota.reshape(m, n, c)

    PixelNum = 50000
    Label = slic(I_Rota, n_segments=PixelNum, compactness=10, sigma=1, start_label=1)
    n_points = np.max(Label)
    indices = np.arange(1, n_points + 1)
    
    radius = np.sqrt(np.sum(delt_I_Rota**2, axis=1))

    red_mean = _robust_accumarray_op(delt_I_Rota[:, 0], Label, indices, 'mean')
    green_mean = _robust_accumarray_op(delt_I_Rota[:, 1], Label, indices, 'mean')
    blue_mean = _robust_accumarray_op(delt_I_Rota[:, 2], Label, indices, 'mean')
    vec = np.column_stack((red_mean, green_mean, blue_mean))
    
    transform_lab = rgb2lab(np.clip(vec, 0, 1))
    ab = transform_lab[:, 1:3]
    kmeans = KMeans(n_clusters=min(2000, len(ab)), n_init='auto', random_state=0).fit(ab)
    T = kmeans.labels_ + 1 
    
    cutoff = np.max(T)
    cluster_indices = np.arange(1, cutoff + 1)
    T_superpixels = T[Label.ravel() - 1]

    radius_max_per_cluster = _robust_accumarray_op(radius, T_superpixels, cluster_indices, 'max')
    radius_std_per_cluster = _robust_accumarray_op(radius, T_superpixels, cluster_indices, 'std')

    radius_max_reshape = radius_max_per_cluster[T_superpixels - 1].reshape(m, n)
    radius_std_reshape = radius_std_per_cluster[T_superpixels - 1].reshape(m, n)
    radius_reshape = radius.reshape(m, n)
    
    radius_std_weight = radius_std_reshape / (np.max(radius_std_reshape) + 1e-9)

    trans_lower_bound = 1 - np.min(im_c / A, axis=2)
    transmission_estimation = radius_reshape / (radius_max_reshape + 1e-9)
    transmission_estimation[np.isnan(transmission_estimation)] = 0
    
    transmission = np.maximum(transmission_estimation, trans_lower_bound)

    transmission = wls_optimization(transmission, radius_std_weight, im_c, lambda_val=0.1)

    # Dehazing in LAB color space
    transform_lab = rgb2lab(im_c)
    L = transform_lab[:, :, 0]
    air_Lab = rgb2lab(A)
    air_l = air_Lab[0, 0, 0]

    transform_lab[:, :, 0] = (L - (1 - transmission) * air_l) / np.maximum(transmission, 0.2)
    
    Lab_l = transform_lab[:, :, 0]

    # --- FINAL CORRECTED imadjust LOGIC ---
    # 1. Normalize the L-channel to the [0, 1] range to prepare for imadjust
    L_min, L_max = np.min(Lab_l), np.max(Lab_l)
    if L_max > L_min:
        Lab_l_norm = (Lab_l - L_min) / (L_max - L_min)
    else:
        Lab_l_norm = np.zeros_like(Lab_l)

    # 2. Find percentiles on the NORMALIZED image and rescale
    p_low, p_high = np.percentile(Lab_l_norm, (1, 99))
    Lab_l_adj = rescale_intensity(Lab_l_norm, in_range=(p_low, p_high))
    
    # 3. Scale the final [0, 1] result to the LAB standard [0, 100] range
    Lab_l = Lab_l_adj * 100
    # --- END CORRECTION ---
    
    H1 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    H2 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    dx = convolve2d(Lab_l, H1, mode='same', boundary='symm')
    dy = convolve2d(Lab_l, H2, mode='same', boundary='symm')
    
    Lab_l = Lab_l + np.sqrt(dx**2 + dy**2) * 0.1
    
    # Clip the L-channel to its valid [0, 100] range before converting back
    transform_lab[:, :, 0] = np.clip(Lab_l, 0, 100)

    img_dehazed = lab2rgb(transform_lab)
    img_Restored = np.clip(img_dehazed ** 1.05, 0, 1)

    return (img_Restored * 255).astype(np.uint8)