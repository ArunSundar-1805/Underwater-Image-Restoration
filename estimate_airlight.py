import numpy as np

def _rgb2ind_python(img, n_colors):
    """
    A Python implementation of Minimum Variance Quantization to replicate MATLAB's
    rgb2ind function, ensuring deterministic output.
    """
    # Reshape image into a list of pixels
    pixels = img.reshape(-1, 3)
    
    # Get unique colors and their counts
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    
    # Initialize the first box to contain all colors
    boxes = [{
        'colors': unique_colors,
        'counts': counts,
        'box': np.array([np.min(unique_colors, axis=0), np.max(unique_colors, axis=0)])
    }]

    # Iteratively split boxes until we have the desired number of colors
    while len(boxes) < n_colors:
        # Find the box with the largest variance to split
        max_var = -1
        box_to_split_idx = -1
        split_axis = -1

        for i, box in enumerate(boxes):
            if box['colors'].shape[0] > 1:
                box_dims = box['box'][1] - box['box'][0]
                axis = np.argmax(box_dims)
                if box_dims[axis] > max_var:
                    max_var = box_dims[axis]
                    box_to_split_idx = i
                    split_axis = axis

        if box_to_split_idx == -1:
            break # No more boxes to split

        # Perform the split
        box_to_split = boxes.pop(box_to_split_idx)
        colors_in_box = box_to_split['colors']
        counts_in_box = box_to_split['counts']
        
        # Find the median to split the box into two
        # A weighted median is more accurate
        sorted_indices = np.argsort(colors_in_box[:, split_axis])
        sorted_colors = colors_in_box[sorted_indices]
        sorted_counts = counts_in_box[sorted_indices]
        
        cumulative_counts = np.cumsum(sorted_counts)
        median_idx = np.where(cumulative_counts >= cumulative_counts[-1] / 2)[0][0]
        
        split_val = sorted_colors[median_idx, split_axis]
        
        # Create two new boxes
        mask1 = colors_in_box[:, split_axis] <= split_val
        # Ensure the second box gets at least one color
        if np.all(mask1):
            mask1[-1] = False
        
        box1_colors, box1_counts = colors_in_box[mask1], counts_in_box[mask1]
        box2_colors, box2_counts = colors_in_box[~mask1], counts_in_box[~mask1]

        for b_colors, b_counts in [(box1_colors, box1_counts), (box2_colors, box2_counts)]:
            if b_colors.shape[0] > 0:
                boxes.append({
                    'colors': b_colors,
                    'counts': b_counts,
                    'box': np.array([np.min(b_colors, axis=0), np.max(b_colors, axis=0)])
                })

    # Create the colormap (points) by finding the weighted average of colors in each box
    colormap = np.zeros((len(boxes), 3))
    for i, box in enumerate(boxes):
        colormap[i] = np.average(box['colors'], weights=box['counts'], axis=0)
        
    # Create the indexed image
    # Build a lookup dictionary from original color to new colormap index
    color_to_index = {}
    for i, box in enumerate(boxes):
        for color in box['colors']:
            color_to_index[tuple(color)] = i
            
    # Map each pixel to its new index
    img_ind = np.array([color_to_index[tuple(p)] for p in pixels]).reshape(img.shape[:2])
    
    return img_ind, colormap


def estimate_airlight(img, Amin=None, Amax=None, N=None, spacing=None, K=None, thres=None):
    """
    Estimates the airlight of an image using a 3x2D Hough transform.
    (Main function with K-Means replaced by the deterministic version)
    """
    # 1. Verify input params, set defaults when necessary
    if thres is None: thres = 0.01
    if spacing is None: spacing = 0.02
    if N is None: N = 1000
    if K is None: K = 40
    if Amin is None: Amin = [0, 0.05, 0.1]
    if Amax is None: Amax = 1.0

    if isinstance(Amin, (int, float)): Amin = [Amin] * 3
    if isinstance(Amax, (int, float)): Amax = [Amax] * 3
    Amin = np.array(Amin)
    Amax = np.array(Amax)

    # 2. Convert input image to an indexed image using the deterministic algorithm
    h, w, _ = img.shape
    img_ind, points = _rgb2ind_python(img, N)
    
    # Remove empty clusters and create a sequential index map
    idx_in_use = np.unique(img_ind)
    points = points[idx_in_use]
    
    map_idx = {old: new for new, old in enumerate(idx_in_use)}
    vectorized_map = np.vectorize(map_idx.get)
    img_ind_sequential = vectorized_map(img_ind)

    # Count the occurrences of each index - this is the clusters' weight
    points_weight = np.bincount(img_ind_sequential.ravel(), minlength=points.shape[0])
    points_weight = points_weight / (h * w)

    # 3. Define arrays of candidate air-light values and angles
    angle_list = np.linspace(0, np.pi, K).reshape(-1, 1)
    directions_all = np.hstack([np.sin(angle_list[:-1]), np.cos(angle_list[:-1])])
    ArangeR = np.arange(Amin[0], Amax[0] + spacing, spacing)
    ArangeG = np.arange(Amin[1], Amax[1] + spacing, spacing)
    ArangeB = np.arange(Amin[2], Amax[2] + spacing, spacing)

    # 4. Estimate air-light in each pair of color channels
    Aall_rg = generate_Avals(ArangeR, ArangeG)
    _, AvoteRG = vote_2D(points[:, 0:2], points_weight, directions_all, Aall_rg, thres)
    Aall_gb = generate_Avals(ArangeG, ArangeB)
    _, AvoteGB = vote_2D(points[:, 1:3], points_weight, directions_all, Aall_gb, thres)
    Aall_rb = generate_Avals(ArangeR, ArangeB)
    _, AvoteRB = vote_2D(points[:, [0, 2]], points_weight, directions_all, Aall_rb, thres)

    # 5. Find most probable airlight from marginal probabilities
    max_val = max(AvoteRB.max(), AvoteRG.max(), AvoteGB.max()) if max(AvoteRB.max(), AvoteRG.max(), AvoteGB.max()) > 0 else 1
    AvoteRG2 = AvoteRG / max_val
    AvoteGB2 = AvoteGB / max_val
    AvoteRB2 = AvoteRB / max_val

    A11 = AvoteRG2.reshape(len(ArangeR), len(ArangeG))[:, :, np.newaxis]
    A22 = AvoteRB2.reshape(len(ArangeR), len(ArangeB))[:, np.newaxis, :]
    A33 = AvoteGB2.reshape(len(ArangeG), len(ArangeB))[np.newaxis, :, :]
    AvoteAll = A11 * A22 * A33
    
    idx = np.argmax(AvoteAll)
    idx_r, idx_g, idx_b = np.unravel_index(idx, AvoteAll.shape)
    
    Aout = np.array([ArangeR[idx_r], ArangeG[idx_g], ArangeB[idx_b]])
    return Aout

# --- Sub-functions (Unchanged) ---
def generate_Avals(Avals1, Avals2):
    Avals1 = Avals1.reshape(-1, 1)
    Avals2 = Avals2.reshape(-1, 1)
    A1 = np.kron(Avals1, np.ones((len(Avals2), 1)))
    A2 = np.kron(np.ones((len(Avals1), 1)), Avals2)
    return np.hstack([A1, A2])

def vote_2D(points, points_weight, directions_all, Aall, thres):
    n_directions = directions_all.shape[0]
    accumulator_votes_idx = np.zeros((Aall.shape[0], points.shape[0], n_directions), dtype=bool)
    for i_point in range(points.shape[0]):
        for i_direction in range(n_directions):
            idx_to_use = np.where((Aall[:, 0] > points[i_point, 0]) & (Aall[:, 1] > points[i_point, 1]))[0]
            if idx_to_use.size == 0: continue
            A_candidates = Aall[idx_to_use, :]
            dist1 = np.sqrt(np.sum((A_candidates - points[i_point, :])**2, axis=1)) / np.sqrt(2) + 1
            dist = -points[i_point, 0] * directions_all[i_direction, 1] + points[i_point, 1] * directions_all[i_direction, 0] + A_candidates[:, 0] * directions_all[i_direction, 1] - A_candidates[:, 1] * directions_all[i_direction, 0]
            idx = np.abs(dist) < 2 * thres * dist1
            if not np.any(idx): continue
            accumulator_votes_idx[idx_to_use[idx], i_point, i_direction] = True
    accumulator_votes_idx2 = (accumulator_votes_idx.sum(axis=1)) >= 2
    accumulator_votes_idx &= accumulator_votes_idx2[:, np.newaxis, :]
    accumulator_unique = np.zeros(Aall.shape[0])
    for iA in range(Aall.shape[0]):
        idx_to_use = np.where((Aall[iA, 0] > points[:, 0]) & (Aall[iA, 1] > points[:, 1]))[0]
        if idx_to_use.size == 0: continue
        points_dist = np.sqrt(np.sum((Aall[iA, :] - points[idx_to_use, :])**2, axis=1))
        points_weight_dist = points_weight[idx_to_use] * (5. * np.exp(-points_dist) + 1)
        voted_points_mask = np.any(accumulator_votes_idx[iA, idx_to_use, :], axis=1)
        accumulator_unique[iA] = np.sum(points_weight_dist[voted_points_mask])
    Aestimate_idx = np.argmax(accumulator_unique)
    Aout = Aall[Aestimate_idx, :]
    Avote2 = accumulator_unique
    return Aout, Avote2