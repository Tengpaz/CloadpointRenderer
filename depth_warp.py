#!/usr/bin/env python3
# depth_warp_backward.py
import json
from pathlib import Path
import numpy as np
import cv2
import imageio
import torch
import torch.nn.functional as F
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
    # print(frame_count, split_info["split"][split_idx])
    while split_idx < len(split_info["split"]) and idx not in split_info["split"][split_idx]:
        split_idx += 1
    # if idx is not in all splits
    if split_idx >= len(split_info["split"]):
        print(f"[Warning] Frame {idx} not found in any split. Using identity camera and empty mask.")
        # 返回默认的相机参数（不会影响 warp，外层可生成全黑 mask）
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

def save_depth16(path, arr, scale_inv=1000.0):
    zn = arr*0.004*1000/(1+arr*0.004*999)
    out = (zn * 65535).astype(np.int64)
    out = np.clip(out, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    cv2.imwrite(str(path), out)

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
    depthmap[~valid] = 0

    return depthmap, valid
    
# -------------------------
# backward warp implemented with PyTorch grid_sample
# For each pixel in target (H,W) we:
#  1) backproject using target intrinsics + depth_target -> P_world
#  2) transform P_world to source camera coords using world->camera_src (w2c_src)
#  3) project to source pixels (u_src, v_src) and sample depth_src with grid_sample
# -------------------------
def backward_warp_depth_torch(depth_src_np, depth_tgt_np, K_src, K_tgt, w2c_src, w2c_tgt, depth_scale=1.0, device='cuda'):
    """
    depth_src_np, depth_tgt_np: HxW numpy (raw 16-bit values already converted by depth_scale -> meters)
    K_src, K_tgt: 3x3 numpy (intrinsics)
    w2c_src, w2c_tgt: 4x4 numpy (world->camera)
    Returns:
        depth_src_sampled_in_tgt: HxW numpy (meters)  -- values from depth_src sampled at source coords for each target pixel
        valid_mask: HxW bool (True where sampled value valid and projected z>0)
    """
    assert depth_src_np.shape == depth_tgt_np.shape
    H, W = depth_src_np.shape

    # device
    use_cuda = torch.cuda.is_available() and (device == 'cuda')
    dev = torch.device('cuda' if use_cuda else 'cpu')

    # convert to torch tensors
    depth_src = torch.from_numpy(depth_src_np.astype(np.float32)).to(dev)  # HxW (meters)
    depth_tgt = torch.from_numpy(depth_tgt_np.astype(np.float32)).to(dev)

    Kt = torch.from_numpy(K_tgt.astype(np.float32)).to(dev)
    Ks = torch.from_numpy(K_src.astype(np.float32)).to(dev)
    w2c_s = torch.from_numpy(w2c_src.astype(np.float32)).to(dev)
    w2c_t = torch.from_numpy(w2c_tgt.astype(np.float32)).to(dev)

    # meshgrid of target pixel coordinates
    ys, xs = torch.meshgrid(
        torch.arange(H, device=dev, dtype=torch.float32),
        torch.arange(W, device=dev, dtype=torch.float32),
        indexing='ij'
    )  # HxW
    # backproject target pixels into target camera coords
    fx_t, fy_t = Kt[0,0], Kt[1,1]
    cx_t, cy_t = Kt[0,2], Kt[1,2]

    Zt = depth_tgt  # HxW, meters
    valid_tgt = Zt > 0
    if valid_tgt.sum() == 0:
        # nothing to warp
        return np.zeros((H,W), np.float32), np.zeros((H,W), dtype=bool)

    X_t = (xs - cx_t) * Zt / fx_t
    Y_t = (ys - cy_t) * Zt / fy_t
    ones = torch.ones_like(X_t)

    pts_t_cam = torch.stack([X_t, Y_t, Zt, ones], dim=-1).view(-1, 4).T  # 4x(H*W)

    # tgt cam -> world: P_w = c2w_tgt @ P_t_cam  (note we have w2c, so invert)
    c2w_t = torch.inverse(w2c_t)
    pts_world = (c2w_t @ pts_t_cam)  # 4xN

    # world -> src cam: P_s_cam = w2c_src @ P_w
    pts_s_cam = (w2c_s @ pts_world)  # 4xN

    Xs = pts_s_cam[0, :].view(H, W)
    Ys = pts_s_cam[1, :].view(H, W)
    Zs = pts_s_cam[2, :].view(H, W)

    # where Zs <= 0, mark invalid (behind camera)
    valid_proj = Zs > 1e-6

    # project to source pixel coords
    fx_s, fy_s = Ks[0,0], Ks[1,1]
    cx_s, cy_s = Ks[0,2], Ks[1,2]

    u_s = fx_s * (Xs / Zs) + cx_s
    v_s = fy_s * (Ys / Zs) + cy_s

    # normalize to grid_sample coords [-1,1]
    u_norm = (u_s / (W - 1)) * 2.0 - 1.0
    v_norm = (v_s / (H - 1)) * 2.0 - 1.0
    grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)  # 1,H,W,2

    # prepare source depth for sampling: shape (1,1,H,W)
    depth_src_t = depth_src.unsqueeze(0).unsqueeze(0)  # 1,1,H,W

    # sample (bilinear) — yields 1,1,H,W
    sampled = F.grid_sample(depth_src_t, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    sampled = sampled[0,0]  # HxW

    # occlusion check: tgt point depth in src cam must match src depth map
    # 如果 Zs 大于 src 深度（采样后的 sampled），说明被遮挡，不可见
    eps = 1e-3  # 2cm tolerance
    visible_mask = (Zs - sampled) <= eps

    # valid = in-front + inside-image + tgt valid + visible
    inside_mask = (u_norm >= -1.0) & (u_norm <= 1.0) & (v_norm >= -1.0) & (v_norm <= 1.0)
    valid_mask = (valid_proj & inside_mask & valid_tgt & visible_mask).cpu().numpy()

    sampled_np = sampled.detach().cpu().numpy()
    sampled_np[~valid_mask] = 0.0

    return sampled_np, valid_mask

# -------------------------
# high-level driver that uses backward warp + round-trip visibility
# -------------------------
def compute_visibility_backward(scene_dir, idA, idB, depth_scale=1/1000.0, vis_thresh=0.02, out_dir="./warp_backward_out"):
    scene = Path(scene_dir)

    depthA_path = scene / "depth" / f"{idA:06d}.png"
    depthB_path = scene / "depth" / f"{idB:06d}.png"
    rawA, rawA_valid = load_depth(depthA_path)
    rawB, reaB_valid = load_depth(depthB_path)
    
    K_A, w2c_A = load_camera_poses(scene, idA)
    K_B, w2c_B = load_camera_poses(scene, idB)

    H, W = rawA.shape
    print(f"Loaded depths: H={H}, W={W}, depth_scale={depth_scale}")
    
    # ------------------- load RGBs -------------------
    imageA_path = scene / "color" / f"{idA:06d}.png"
    imageB_path = scene / "color" / f"{idB:06d}.png"

    imgA = cv2.imread(str(imageA_path))[:, :, ::-1]   # BGR->RGB
    imgB = cv2.imread(str(imageB_path))[:, :, ::-1]

    # float32 normalized RGB for multiplication
    imgA_f = imgA.astype(np.float32) / 255.0
    imgB_f = imgB.astype(np.float32) / 255.0

    # -------------------------------------------------

    # to meters
    dA = (rawA * depth_scale).astype(np.float32)
    dB = (rawB * depth_scale).astype(np.float32)

    # ---------------- A -> B  (backward using depth_B) ----------------
    print("Warping A -> B using depth_B (backward sampling)...")
    A_in_B, validAinB = backward_warp_depth_torch(dA, dB, K_A, K_B, w2c_A, w2c_B)
    print("A_in_B valid:", int(validAinB.sum()), "/", H*W)

    # ---------------- B -> A (backward using depth_A) ----------------
    print("Warping B -> A using depth_A (backward sampling)...")
    B_in_A, validBinA = backward_warp_depth_torch(dB, dA, K_B, K_A, w2c_B, w2c_A)
    print("B_in_A valid:", int(validBinA.sum()), "/", H*W)

    # ---------------- roundtrip A->B->A (sample A_in_B back to A using depth_A) ---------------
    # For roundtrip we want depth values in A after warping A->B then back to A.
    # We'll use backward sampling again: treat A_in_B as "source depth" and dA as "target depth"
    # so we can sample A_in_B into A pixels using dA as driving depth (B->A style).
    print("Computing A -> B -> A roundtrip (backward)...")
    A_rt, valid_rt = backward_warp_depth_torch(A_in_B, dA, K_B, K_A, w2c_B, w2c_A)
    print("A roundtrip valid:", int(valid_rt.sum()), "/", H*W)

    # ---------------- visibility mask ----------------
    diff = np.abs(dA - A_rt)
    mask_A_rt_visible = (A_rt > 0) & (diff < vis_thresh)
    mask_A_in_B_visible = validAinB
    mask_B_in_A_visible = validBinA

    os.makedirs(out_dir, exist_ok=True)
    # save originals (as 16-bit, back to original unit)
    save_depth16(Path(out_dir)/"A_depth_original.png", dA, scale_inv=1.0/depth_scale)
    save_depth16(Path(out_dir)/"B_depth_original.png", dB, scale_inv=1.0/depth_scale)
    # ---------------- save RGB originals ----------------
    cv2.imwrite(str(Path(out_dir)/"A_rgb.png"), imgA[:, :, ::-1])  # RGB->BGR
    cv2.imwrite(str(Path(out_dir)/"B_rgb.png"), imgB[:, :, ::-1])
    # save warped results
    save_depth16(Path(out_dir)/"A_in_B.png", A_in_B, scale_inv=1.0/depth_scale)
    save_depth16(Path(out_dir)/"B_in_A.png", B_in_A, scale_inv=1.0/depth_scale)
    save_depth16(Path(out_dir)/"A_roundtrip.png", A_rt, scale_inv=1.0/depth_scale)
    # save masks as png
    cv2.imwrite(str(Path(out_dir)/"mask_A_rt_visible.png"), (mask_A_rt_visible.astype(np.uint8)*255))
    cv2.imwrite(str(Path(out_dir)/"mask_A_in_B_visible.png"), (mask_A_in_B_visible.astype(np.uint8)*255))
    cv2.imwrite(str(Path(out_dir)/"mask_B_in_A_visible.png"), (mask_B_in_A_visible.astype(np.uint8)*255))

    # ---------------- multiply RGB * mask ----------------
    # mask is HxW → need expand to HxWx3
    def mask_rgb(rgb, mask, name):
        mask3 = np.repeat(mask[:, :, None], 3, axis=2)
        out = (rgb * mask3 * 255.0).astype(np.uint8)
        cv2.imwrite(str(Path(out_dir)/name), out[:, :, ::-1])  # RGB->BGR

    # A_rgb * mask_A_rt_visible
    mask_rgb(imgA_f, mask_A_rt_visible, "A_rgb_masked_rt.png")

    # A_rgb * mask_A_in_B
    mask_rgb(imgA_f, mask_B_in_A_visible, "A_rgb_masked_in_B.png")

    # B_rgb * mask_B_in_A
    mask_rgb(imgB_f, mask_A_in_B_visible, "B_rgb_masked_in_A.png")

    print("Saved outputs to", out_dir)
    return {
        "A_in_B": A_in_B, "B_in_A": B_in_A, "A_roundtrip": A_rt,
        "mask_A_rt_visible": mask_A_rt_visible, 
    }


if __name__ == "__main__":
    scene_dir = "/vePFS-MLP/buaa/OmniWorld-Game/0c3cefcf3a15"
    split_idx = 0
    # idA = 1252
    # idB = 1302
    idA = 53
    idB = 63

    depth_scale = 1.0
    vis_thresh = 0.02
    out_dir = "./warp_backward_out"
    res = compute_visibility_backward(scene_dir, idA, idB, depth_scale=depth_scale, vis_thresh=vis_thresh, out_dir=out_dir)