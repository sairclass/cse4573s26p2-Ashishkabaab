'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''
import torch
import kornia as K
from typing import Dict
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

# ------------------------------------ Helper Functions ------------------------------------ #

def to_float(img: torch.Tensor) -> torch.Tensor:
    """ Convert uint8 CHW [0,255] to float32 CHW [0,1] """
    return img.float() / 255.0

def to_uint8(img: torch.Tensor) -> torch.Tensor:
    """ Convert float32 CHW [0,1] tensor to uint8 CHW [0,255] """
    return (img.clamp(0, 1) * 255).byte()

def extract_and_match_features(img1: torch.Tensor, img2: torch.Tensor, num_features: int = 2000):
    """ Extract keypoints + HardNet descriptors from two float CHW images and 
        match them using a second nearest neighbor matcher.
        Args: img1, img2: float32 CHW [0,1] tensors. num_features: max keypoints to detect per image
        Returns: pts1, pts2: (M, 2) float tensors of matched (x, y) pixel coordinates
    """

    b1 = img1.unsqueeze(0)  # (1, C, H, W) 
    b2 = img2.unsqueeze(0)  # (1, C, H, W)

    # Convert to grayscale for feature detection
    gray1 = K.color.rgb_to_grayscale(b1)  # (1, 1, H, W)
    gray2 = K.color.rgb_to_grayscale(b2)  # (1, 1, H, W)

    sift    = K.feature.SIFTFeature(num_features=num_features, upright=False)
    matcher = K.feature.DescriptorMatcher('smnn', 0.85)

    with torch.no_grad():
        lafs1, resps1, descs1 = sift(gray1)
        lafs2, resps2, descs2 = sift(gray2)

        print(f"Keypoints detected: img1={lafs1.shape[1]}, img2={lafs2.shape[1]}")

        if lafs1.shape[1] == 0 or lafs2.shape[1] == 0:
            empty = torch.zeros(0, 2, dtype=torch.float32)
            return empty, empty

        _, idxs = matcher(descs1[0], descs2[0])
        print(f"Matches after SMNN: {idxs.shape[0]}")
    
    kpts1 = K.feature.get_laf_center(lafs1)[0]  # (N1, 2) float tensor of (x, y) keypoint centers in img1
    kpts2 = K.feature.get_laf_center(lafs2)[0]  # (N2, 2) float tensor of (x, y) keypoint centers in img2

    pts1 = kpts1[idxs[:, 0]]  # (M, 2) matched keypoints in img1
    pts2 = kpts2[idxs[:, 1]]  # (M, 2) matched keypoints in img2

    return pts1, pts2

def count_inliers(pts1: torch.Tensor, pts2: torch.Tensor, H: torch.Tensor, threshold: float = 3.0):
    """ Return the inlier mask and count for the homography H mapping pts1 to pts2.
    """
    N = pts1.shape[0]
    ones = torch.ones(N, 1, dtype=pts1.dtype, device=pts1.device)
    p1h = torch.cat([pts1, ones], dim=1)  # (N, 3)
    proj = (H @ p1h.T).T  # (N, 3)
    proj = proj / (proj[:, 2:3] + 1e-8)  # normalize homogeneous coordinates
    errs = torch.norm(proj[:, :2] - pts2, dim=1)  # (N,) reprojection errors
    mask = errs < threshold
    return mask, mask.sum().item()

def compute_homography_ransac(pts1: torch.Tensor, pts2: torch.Tensor, max_iter: int = 2000, threshold: float = 3.0):
    """ Estimate homography H such that pts2 ~ H @ pts1 using RANSAC,
        then refine on all inliers with DLT.

        Args: pts1, pts2: (N, 2) matched point tensors in (x, y) order
        max_iter: number of RANSAC iterations
        threshold: reprojection error threshold in pixels

        Returns: H: (3,3) homography tensor, or None if < 4 matches
    """
    N = pts1.shape[0]
    if N < 4:
        return None
    
    best_H = None
    best_inliers = -1
    best_mask = None

    for _ in range(max_iter):
        idx = torch.randperm(N)[:4]  # random sample of 4 matches
        s1 = pts1[idx].unsqueeze(0)  # (1, 4, 2)
        s2 = pts2[idx].unsqueeze(0)  # (1, 4, 2)

        try:
            H_cand = K.geometry.homography.find_homography_dlt(s1, s2)[0]  # (3, 3)
        except Exception:
            continue  # skip degenerate configurations

        mask, n_inliers = count_inliers(pts1, pts2, H_cand, threshold)

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_H = H_cand
            best_mask = mask
    
    #Refine on inliers
    if best_H is not None and best_inliers >= 4:
        try:
            best_H = K.geometry.find_homography_dlt(pts1[best_mask].unsqueeze(0), pts2[best_mask].unsqueeze(0))[0]
        except Exception:
            pass  # if refinement fails, keep the best from RANSAC
    return best_H

def warp_and_composite(img_new: torch.Tensor, canvas: torch.Tensor, H: torch.Tensor, foreground_elimination: bool = False):
    """ Warp img_new onto canvas using homography H and composite both onto
        an output canvas large enough to hold both.
        Args: 
        img_new: (C, H, W) tensor of the new image to warp
        canvas: (C, Hc, Wc) tensor of the existing canvas
        H: (3, 3) homography mapping img_new to canvas
        foreground_elimination: if True, average-blend the overlap region to suppress moving foreground
        Returns:
        composite: float32 CHcWc merged image
    """
    C, Hn, Wn = img_new.shape
    C, Hc, Wc = canvas.shape

    # compute bounding box of warped img_new in canvas coordinates to determine output size
    corners = torch.tensor([[0, 0, 1], [Wn, 0, 1], [0, Hn, 1], [Wn, Hn, 1]], dtype=torch.float32).T  # (3, 4)
    pc = H @ corners  # (3, 4)
    pc = pc / (pc[2:3] + 1e-8)  # normalize homogeneous coordinates

    all_x = torch.cat([pc[0], torch.tensor([0.0, float(Wc)])])
    all_y = torch.cat([pc[1], torch.tensor([0.0, float(Hc)])])

    min_x = int(all_x.min().floor().item())
    max_x = int(all_x.max().ceil().item())
    min_y = int(all_y.min().floor().item())
    max_y = int(all_y.max().ceil().item())

    out_W = max_x - min_x
    out_H = max_y - min_y

    #Translation that shifts both into non-negative coordinates
    T = torch.eye(3, dtype=torch.float32)
    T[0, 2] = -min_x
    T[1, 2] = -min_y

    H_total = T @ H  # warp img_new into output canvas

    # Warp img_new
    warped_new = K.geometry.warp_perspective(img_new.unsqueeze(0), H_total.unsqueeze(0), dsize=(out_H, out_W), mode='bilinear', padding_mode='zeros')[0]  # (C, out_H, out_W)

    # Place canvas into output
    out_canvas = torch.zeros(C, out_H, out_W, dtype=torch.float32)
    ox = -min_x
    oy = -min_y
    x0 = max(0, ox)
    y0 = max(0, oy)
    x1 = min(Wc + ox, out_W)
    y1 = min(Hc + oy, out_H)
    sx0 = x0 - ox
    sy0 = y0 - oy
    out_canvas[:, y0:y1, x0:x1] = canvas[:, sy0:sy0+(y1-y0), sx0:sx0+(x1-x0)]

    #Masks
    mask_new = (warped_new.sum(0, keepdim=True) > 0).float()  # (1, out_H, out_W) mask of warped new image
    mask_canvas = (out_canvas.sum(0, keepdim=True) > 0).float()  # (1, out_H, out_W) mask of existing canvas
    overlap = mask_new * mask_canvas  # (1, out_H, out_W) mask of overlap region

    #Composite
    if foreground_elimination:
        # Average in overlap to cancel out moving foreground objects
        blended = (warped_new + out_canvas) / 2.0
        composite = (warped_new * mask_new * (1 - overlap) + out_canvas * mask_canvas * (1 - overlap) + blended * overlap)
    else:
        #Standard mosaic: prefer the existing canvas in overlap
        composite = (warped_new * mask_new * (1 - overlap) + out_canvas * mask_canvas * (1 - overlap) + out_canvas * overlap)

    return composite

# ------------------------------------ Task 1 ------------------------------------ #
def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image.
    """
    #TODO: Add your code here. Do not modify the return and input arguments.

    keys = list(imgs.keys())
    img1 = to_float(imgs[keys[0]])  # (C, H, W) float tensor
    img2 = to_float(imgs[keys[1]])  # (C, H, W) float tensor

    pts1, pts2 = extract_and_match_features(img1, img2, num_features=2000)

    H = compute_homography_ransac(pts1, pts2, max_iter=2000, threshold=3.0)

    if H is None:
        print("Not enough matches to compute homography.")
        return imgs[keys[1]]  # return img2 as fallback
    
    #Warp img1 into img2's frame with foreground blending
    result = warp_and_composite(img1, img2, H, foreground_elimination=True)
    
    return to_uint8(result)

# ------------------------------------ Task 2 ------------------------------------ #

def compute_overlap_matrix(imgs_float: Dict[str, torch.Tensor], keys: list, min_inliers: int = 20, overlap_threshold: float = 0.2):
    """ Build symmetric NxN overlap matrix. Entry (i, j) is 1 if imgs_float[keys[i]] and imgs_float[keys[j]] have sufficient overlap based on RANSAC inliers and estimated overlap area.
        Args:
        imgs_float: dict of float CHW [0,1] tensors
        keys: ordered key list
        min_inliers: minimum number of RANSAC inliers to consider a valid overlap
        overlap_threshold: minimum fraction of image area that must be in overlap to consider valid
        Returns:
        overlap_matrix: (N, N) int32 Torch.tensor
    """
    N = len(keys)
    overlap = torch.zeros((N, N), dtype=torch.int32)

    for i in range(N):
        overlap[i, i] = 1  # self-overlap
    
    for i in range(N):
        for j in range(i+1, N):
            img_i = imgs_float[keys[i]]
            img_j = imgs_float[keys[j]]

            pts_i, pts_j = extract_and_match_features(img_i, img_j, num_features=2000)

            if pts_i.shape[0] < 4:
                continue  # not enough matches to compute homography

            H = compute_homography_ransac(pts_i, pts_j, max_iter=1000, threshold=3.0)
            if H is None:
                continue  # homography estimation failed

            _, n_inliers = count_inliers(pts_i, pts_j, H, threshold=3.0)
            if n_inliers < min_inliers:
                continue  # not enough inliers to consider overlap

            #Estimate overlap fraction via projected bounding box
            _, Hi, Wi = img_i.shape
            _, Hj, Wj = img_j.shape

            corners = torch.tensor([[0, 0, 1], [Wi, 0, 1], [0, Hi, 1], [Wi, Hi, 1]], dtype=torch.float32).T  # (3, 4)
            pc = H @ corners  # (3, 4)
            pc = pc / (pc[2:3] + 1e-8)

            ix0 = max(pc[0].min().item(), 0)
            iy0 = max(pc[1].min().item(), 0)
            ix1 = min(pc[0].max().item(), Wj)
            iy1 = min(pc[1].max().item(), Hj)

            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            smaller_area = min(Hi * Wi, Hj * Wj)
            frac = inter / (smaller_area + 1e-8)

            if frac >= overlap_threshold:
                overlap[i, j] = 1
                overlap[j, i] = 1

    return overlap

def build_panorama(imgs_float: Dict[str, torch.Tensor], keys: list, overlap: torch.Tensor):
    """ Incrementally stitch all connected images into one panorama using BFS
    The largest image (by pixel area) is chosen as the reference/root. 
    Args:
        imgs_float: dict of float CHW [0,1] tensors
        keys: ordered list matching rows/cols of overlap 
        overlap: (N, N) overlap matrix
    Returns:
        canvas: float CHW panorama image
    """
    N = len(keys)

    #Pick the largest image as the BFS root
    areas = [imgs_float[k].shape[1] * imgs_float[k].shape[2] for k in keys]
    root = int(torch.tensor(areas, dtype=torch.int32).argmax().item())

    #BFS to get placement order
    visited = [False] * N
    queue = [root]
    order = []
    visited[root] = True
    while queue:
        node = queue.pop(0)
        order.append(node)
        for nb in range(N):
            if not visited[nb] and overlap[node, nb].item() == 1:
                visited[nb] = True
                queue.append(nb)
    
    #Initialize canvas with root image
    canvas = imgs_float[keys[root]]
    # H_to_canvas[i] = homography from image i's original coordinates to current canvas coordinates
    H_to_canvas = {}
    H_to_canvas[root] = torch.eye(3, dtype=torch.float32)

    for idx in order[1:]: # skip root
        img_new = imgs_float[keys[idx]]

        #Find the best already placed neighbor to stitch with
        best_nb = None
        best_H_nb = None
        best_inliers = -1

        for nb in order:
            if nb == idx or nb not in H_to_canvas:
                continue
            if overlap[idx, nb].item() != 1:
                continue

            pts_new, pts_nb = extract_and_match_features(img_new, imgs_float[keys[nb]], num_features=2000)
            if pts_new.shape[0] < 4:
                continue

            H_new_nb = compute_homography_ransac(pts_new, pts_nb, max_iter=1000, threshold=3.0)
            if H_new_nb is None:
                continue

            _, n_inliers = count_inliers(pts_new, pts_nb, H_new_nb)
            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_nb = nb
                best_H_nb = H_new_nb
        
        if best_nb is None:
            continue # could not find a valid neighbor to stitch with, skip this image

        # Chain: img_new -> best_nb -> canvas
        H_new_canvas = H_to_canvas[best_nb] @ best_H_nb  # map img_new to canvas coordinates

        #Compute the translation offset that warp_and_composite will apply
        C_, Hn, Wn = img_new.shape
        C_, Hc, Wc = canvas.shape

        corners = torch.tensor([[0, 0, 1], [Wn, 0, 1], [0, Hn, 1], [Wn, Hn, 1]], dtype=torch.float32).T  # (3, 4)
        pc = H_new_canvas @ corners  # (3, 4)
        pc = pc / (pc[2:3] + 1e-8)

        all_x = torch.cat([pc[0], torch.tensor([0.0, float(Wc)])])
        all_y = torch.cat([pc[1], torch.tensor([0.0, float(Hc)])])

        min_x = int(all_x.min().floor().item())
        min_y = int(all_y.min().floor().item())

        T_offset = torch.eye(3, dtype=torch.float32)
        T_offset[0, 2] = -min_x
        T_offset[1, 2] = -min_y

        # Update all existing homographies to canvas with the new offset
        for k in list(H_to_canvas.keys()):
            H_to_canvas[k] = T_offset @ H_to_canvas[k]

        H_to_canvas[idx] = T_offset @ H_new_canvas

        #Composite img_new onto growing canvas
        canvas = warp_and_composite(img_new, canvas, H_new_canvas, foreground_elimination=False)

    return canvas

def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Stitch multiple images into a panorama.
    Args:
        imgs: dict {filename: CxHxW tensor} for task-2.
    Returns:
        img: panorama, 
        overlap: torch.Tensor of the output image. 
    """

    #TODO: Add your code here. Do not modify the return and input arguments.
    imgs_float = {k: to_float(v) for k, v in imgs.items()}
    keys = list(imgs_float.keys())

    overlap_matrix = compute_overlap_matrix(imgs_float, keys, min_inliers=20, overlap_threshold=0.2)

    result = build_panorama(imgs_float, keys, overlap_matrix)

    return to_uint8(result), overlap_matrix
