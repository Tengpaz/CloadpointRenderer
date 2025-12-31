#!/usr/bin/env python3
# depth_warp_points_reproject.py
import json
import imageio
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import os, sys

# -------------------------
# camera loaders (same format as yours)
# -------------------------
def load_split_info(scene_dir: Path):
    with open(scene_dir / "split_info.json", "r", encoding="utf-8") as f:
        return json.load(f)

def load_camera_poses(scene_dir: Path, idx: int):
    split_info = load_split_info(scene_dir)

    split_idx = 0
    while split_idx < len(split_info["split"]) and idx not in split_info["split"][split_idx]:
        split_idx += 1
    # if idx is not in all splits
    if split_idx >= len(split_info["split"]):
        # 不存在该帧：返回 None，外层会处理为全黑 mask
        return None, None

    start_id = split_info["split"][split_idx][0]
    frame_count = len(split_info["split"][split_idx])
    cam_file = scene_dir / "camera" / f"split_{split_idx}.json"
    with open(cam_file, "r", encoding="utf-8") as f:
        cam = json.load(f)

    intrinsics = np.repeat(np.eye(3)[None, ...], frame_count, axis=0).astype(np.float32)
    intrinsics[:, 0, 0] = cam["focals"]
    intrinsics[:, 1, 1] = cam["focals"]
    intrinsics[:, 0, 2] = cam["cx"]
    intrinsics[:, 1, 2] = cam["cy"]

    extrinsics = np.repeat(np.eye(4)[None, ...], frame_count, axis=0).astype(np.float32)
    quat_wxyz = np.array(cam["quats"])           # (S,4) (w,x,y,z)
    quat_xyzw = np.concatenate([quat_wxyz[:, 1:], quat_wxyz[:, :1]], axis=1)  # (x,y,z,w)
    rotations = R.from_quat(quat_xyzw).as_matrix().astype(np.float32)
    translations = np.array(cam["trans"]).astype(np.float32)

    extrinsics[:, :3, :3] = rotations
    extrinsics[:, :3, 3] = translations

    # extrinsics are world-to-camera (w2c)
    K_A = intrinsics[idx - start_id]
    w2c_A = extrinsics[idx - start_id]
    return K_A, w2c_A

# -------------------------
# depth IO helpers
# -------------------------
def load_depth_16bit(path):
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"cannot read depth: {path}")
    return d.astype(np.float32)

def load_depth(depthpath):
    """
    Returns
    -------
    depthmap : (H, W) float32
    valid   : (H, W) bool      True for reliable pixels
    """

    depthmap = imageio.v2.imread(depthpath).astype(np.float32) / 65535.0
    near_mask = depthmap < 0.0015   # 1. too close
    far_mask = depthmap > (65500.0 / 65535.0) # 2. filter sky
    # far_mask = depthmap > np.percentile(depthmap[~far_mask], 95) # 3. filter far area (optional)
    near, far = 1., 1000.
    depthmap = depthmap / (far - depthmap * (far - near)) / 0.004

    valid = ~(near_mask | far_mask)
    depthmap[~valid] = -1

    return depthmap, valid

def save_depth16(path, arr, scale_inv=1000.0):
    zn = arr*0.004*1000/(1+arr*0.004*999)
    out = (zn * 65535).astype(np.int64)
    out = np.clip(out, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    cv2.imwrite(str(path), out)

# -------------------------
# Backproject depth -> world points (numpy)
# -------------------------
def backproject_depth_to_world(depth_np, K, w2c, depth_scale=1.0):
    """
    depth_np: H x W (raw values already converted to meters by caller)
    K: 3x3
    w2c: 4x4
    returns:
        pts_world: (N,3) float32 world coordinates (only for valid depth>0)
        src_indices: (N,) int   indices in 0..H*W-1 corresponding to pixel order (v*W+u)
        H, W: ints
    """
    H, W = depth_np.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))  # shape HxW
    z = depth_np.reshape(-1)       # Npix
    valid = z == z
    if valid.sum() == 0:
        return np.zeros((0,3), dtype=np.float32), np.zeros((0,), dtype=np.int32), H, W

    u_f = (u.reshape(-1)[valid].astype(np.float32) + 0.5)  # center of pixel
    v_f = (v.reshape(-1)[valid].astype(np.float32) + 0.5)
    z_f = z[valid].astype(np.float32)

    fx = float(K[0,0]); fy=float(K[1,1]); cx=float(K[0,2]); cy=float(K[1,2])
    x_cam = (u_f - cx) * z_f / fx
    y_cam = (v_f - cy) * z_f / fy
    pts_cam = np.stack([x_cam, y_cam, z_f], axis=1)  # (M,3)

    # cam -> world: need c2w = inv(w2c)
    c2w = np.linalg.inv(w2c)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0],1), dtype=np.float32)], axis=1)  # Mx4
    pts_world_h = (c2w @ pts_h.T).T  # Mx4
    pts_world = pts_world_h[:, :3]

    # compute src_indices (original pixel flat index)
    all_indices = np.arange(H*W, dtype=np.int32)
    src_indices = all_indices[valid]  # map each point to flat pixel index

    return pts_world.astype(np.float32), src_indices.astype(np.int32), H, W

# -------------------------
# Render points (world) into a depth image with z-buffer and index map
# -------------------------
def render_points_with_index(pts_world, src_indices, K, w2c, H, W):
    """
    pts_world: (M,3)
    src_indices: (M,)  original point indices (pixel indices from source)
    K: 3x3
    w2c: 4x4
    returns:
        depth_img: HxW float32 (meters), 0 for empty
        idx_map: HxW int32  (source point index that was nearest), -1 for empty
    """
    if pts_world.shape[0] == 0:
        return np.zeros((H,W), dtype=np.float32), -np.ones((H,W), dtype=np.int32)

    # world -> camera
    pts_h = np.concatenate([pts_world, np.ones((pts_world.shape[0],1), dtype=np.float32)], axis=1)  # Mx4
    pts_cam_h = (w2c @ pts_h.T).T  # Mx4
    X = pts_cam_h[:,0]; Y = pts_cam_h[:,1]; Z = pts_cam_h[:,2]

    # valid: Z>0
    valid = Z == Z
    if valid.sum() == 0:
        return np.zeros((H,W), dtype=np.float32), -np.ones((H,W), dtype=np.int32)

    X = X[valid]; Y = Y[valid]; Z = Z[valid]
    idx_valid = src_indices[valid]

    fx = float(K[0,0]); fy = float(K[1,1]); cx = float(K[0,2]); cy = float(K[1,2])
    u = (fx * (X / Z) + cx)
    v = (fy * (Y / Z) + cy)

    # integer pixel coords
    u_i = np.round(u).astype(np.int64)
    v_i = np.round(v).astype(np.int64)

    # keep only inside image
    inside = (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
    if inside.sum() == 0:
        return np.zeros((H,W), dtype=np.float32), -np.ones((H,W), dtype=np.int32)

    u_i = u_i[inside]; v_i = v_i[inside]; Z = Z[inside]; idx_valid = idx_valid[inside]

    # flatten pixel index
    pix_flat = v_i * W + u_i  # shape P

    # lexsort by pixel then depth (we want smallest depth per pixel)
    order = np.lexsort((Z, pix_flat))  # sorts by pix_flat asc, then Z asc
    pix_sorted = pix_flat[order]
    depths_sorted = Z[order]
    idxs_sorted = idx_valid[order]

    # find first occurrence per unique pixel => nearest
    unique_pix, first_idx = np.unique(pix_sorted, return_index=True)
    chosen_depths = depths_sorted[first_idx]
    chosen_src_idx = idxs_sorted[first_idx]

    # fill depth flat and idx flat
    depth_flat = np.full(H*W, np.inf, dtype=np.float32)
    idx_flat = -np.ones(H*W, dtype=np.int32)
    depth_flat[unique_pix] = chosen_depths
    idx_flat[unique_pix] = chosen_src_idx

    depth_img = depth_flat.reshape(H, W)
    idx_map = idx_flat.reshape(H, W)

    # convert inf to 0 for empty pixels
    depth_img[np.isinf(depth_img)] = 0.0

    return depth_img.astype(np.float32), idx_map.astype(np.int32)

# -------------------------
# main pipeline: use point-projection (方案 A)
# -------------------------
def compute_visibility_backward(scene_dir, split_idx, idA, idB, depth_scale=1/1000.0, vis_thresh=0.02, out_dir="./warp_points_out"):
    scene = Path(scene_dir)

    depthA_path = scene / "depth" / f"{idA:06d}.png"
    depthB_path = scene / "depth" / f"{idB:06d}.png"
    if not depthA_path.exists() or not depthB_path.exists():
        raise FileNotFoundError("depth files not found for A or B")

    rawA, rawAvalid = load_depth(depthA_path)
    rawB, rawBvalid = load_depth(depthB_path)
    
    K_A, w2c_A = load_camera_poses(scene, idA)
    K_B, w2c_B = load_camera_poses(scene, idB)

    H, W = rawA.shape
    print(f"Loaded depths: H={H}, W={W}, depth_scale={depth_scale}")

    # if camera info missing -> directly save full-black mask and return
    if K_A is None or K_B is None:
        print(f"[Warning] missing camera for idA={idA} or idB={idB}. Save full-black mask.")
        os.makedirs(out_dir, exist_ok=True)
        mask_black = np.zeros((H,W), dtype=np.uint8)
        cv2.imwrite(str(Path(out_dir)/f"{idA:06d}.png"), mask_black)  # 0=not visible
        return {
            "A_in_B": np.zeros((H,W), np.float32),
            "B_in_A": np.zeros((H,W), np.float32),
            "A_roundtrip": np.zeros((H,W), np.float32),
            "mask_visible": np.zeros((H,W), dtype=bool),
            "mask_invisible": np.zeros((H,W), dtype=bool),
        }

    # to meters (caller chooses depth_scale)
    dA = (rawA * depth_scale).astype(np.float32)
    dB = (rawB * depth_scale).astype(np.float32)

    # 1) A depth -> world points
    ptsA_world, src_idxA, H_a, W_a = backproject_depth_to_world(dA, K_A, w2c_A, depth_scale)
    # 2) Render A points into B (z-buffer) -> depth image + idx_map (point index from A)
    A_in_B, idx_map_B = render_points_with_index(ptsA_world, src_idxA, K_B, w2c_B, H, W)

    # 3) B depth -> world points and render to A (optional)
    ptsB_world, src_idxB, _, _ = backproject_depth_to_world(dB, K_B, w2c_B, depth_scale)
    B_in_A, idx_map_A = render_points_with_index(ptsB_world, src_idxB, K_A, w2c_A, H, W)

    # 4) Determine visible pixels of A when viewed from B:
    # For every source pixel p in A (flat index i), project it into B: we have idx_map_B at projected pixel,
    # if idx_map_B[...] == i => that A pixel is the nearest point at that B pixel → visible from B.
    # We need to compute for each A pixel its projection pixel in B (int coords).
    # Instead of recomputing u,v for all points, we can derive them from ptsA_world -> project to B camera:
    if ptsA_world.shape[0] > 0:
        # project ptsA_world into B camera to get pixel coords
        pts_h = np.concatenate([ptsA_world, np.ones((ptsA_world.shape[0],1), dtype=np.float32)], axis=1)  # Mx4
        pts_cam_B = (w2c_B @ pts_h.T).T  # Mx4
        Xb = pts_cam_B[:,0]; Yb = pts_cam_B[:,1]; Zb = pts_cam_B[:,2]
        fx_b = float(K_B[0,0]); fy_b = float(K_B[1,1]); cx_b=float(K_B[0,2]); cy_b=float(K_B[1,2])
        u_b = np.round(fx_b * (Xb / Zb) + cx_b).astype(np.int64)
        v_b = np.round(fy_b * (Yb / Zb) + cy_b).astype(np.int64)
        # inside check
        inside = (Zb > 1e-6) & (u_b >= 0) & (u_b < W) & (v_b >= 0) & (v_b < H)
        # build flat indices for A points (same ordering as src_idxA)
        # src_idxA gives flat index in image (v*W+u)
        # For each valid point, check idx_map_B at (v_b,u_b)
        flat_uv_b = v_b * W + u_b
        # initialize A visibility to False
        mask_visible_flat = np.zeros(H*W, dtype=bool)
        if inside.sum() > 0:
            # index into idx_map_B flattened
            idx_map_B_flat = idx_map_B.reshape(-1)
            # For valid points, if idx_map_B_flat[flat_uv_b[k]] == src_idxA[k] -> visible
            valid_positions = np.where(inside)[0]
            check_flat = flat_uv_b[valid_positions]
            check_src_indices = src_idxA[valid_positions]
            matched = idx_map_B_flat[check_flat] == check_src_indices
            mask_visible_flat[check_src_indices[matched]] = True
        mask_visible = mask_visible_flat.reshape(H, W)
    else:
        mask_visible = np.zeros((H,W), dtype=bool)

    mask_invisible = (dA > 0) & (~mask_visible)

    # Roundtrip A->B->A: approximate by checking whether A point survives reprojection (mask_visible),
    # and produce A_roundtrip depth image where visible points keep their original depth, others 0.
    A_roundtrip = np.zeros((H,W), dtype=np.float32)
    if ptsA_world.shape[0] > 0:
        # src_idxA are flat indices for ptsA_world
        flat_idx = src_idxA  # flat index for each point
        # convert to v,u
        v_src = flat_idx // W
        u_src = flat_idx % W
        # assign original depth dA_flat to A_roundtrip where mask_visible
        dA_flat = dA.reshape(-1)
        visible_positions = mask_visible.reshape(-1)
        A_roundtrip.reshape(-1)[visible_positions] = dA_flat[visible_positions]

    # save outputs
    os.makedirs(out_dir, exist_ok=True)
    save_depth16(Path(out_dir)/"A_depth_original.png", dA, scale_inv=1.0/depth_scale)
    save_depth16(Path(out_dir)/"B_depth_original.png", dB, scale_inv=1.0/depth_scale)
    # save warped depths (A_in_B, B_in_A, A_roundtrip) as 16-bit (convert back by scale_inv)
    save_depth16(Path(out_dir)/"A_in_B.png", A_in_B, scale_inv=1.0/depth_scale)
    save_depth16(Path(out_dir)/"B_in_A.png", B_in_A, scale_inv=1.0/depth_scale)
    save_depth16(Path(out_dir)/"A_roundtrip.png", A_roundtrip, scale_inv=1.0/depth_scale)

    # save masks: visible / invisible (0/255)
    cv2.imwrite(str(Path(out_dir)/"mask_visible.png"), (mask_visible.astype(np.uint8) * 255))
    cv2.imwrite(str(Path(out_dir)/"mask_invisible.png"), (mask_invisible.astype(np.uint8) * 255))

    print("Saved outputs to", out_dir)
    return {
        "A_in_B": A_in_B, "B_in_A": B_in_A, "A_roundtrip": A_roundtrip,
        "mask_visible": mask_visible, "mask_invisible": mask_invisible
    }

# -------------------------
# run example - edit these
# -------------------------
if __name__ == "__main__":
    # === EDIT THESE PATHS / IDS ===
    scene_dir = "/cos_zw1/share_304064442/hunyuan/suanhuang/4d_world_videos/OmniWorld-Game/0c3cefcf3a15"
    split_idx = 0
    idA = 1252
    idB = 1302
    # idA = 53
    # idB = 83
    # depth_scale: if 16-bit stores mm, use 1/1000.0; else 1.0
    depth_scale = 1.0
    vis_thresh = 0.02
    out_dir = "./warp_points_out"
    res = compute_visibility_backward(scene_dir, split_idx, idA, idB, depth_scale=depth_scale, vis_thresh=vis_thresh, out_dir=out_dir)