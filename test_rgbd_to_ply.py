#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import numpy as np
import imageio
import cv2

# Use depth and camera loaders from depth_warp.py
from depth_warp import load_depth, load_camera_poses


def project_rgbd_to_pointcloud(depth, valid_mask, K, w2c, rgb):
    """
    depth: (H,W) float32 depth in meters (from load_depth)
    valid_mask: (H,W) bool mask for reliable pixels
    K: (3,3) intrinsics
    w2c: (4,4) world-to-camera
    rgb: (H,W,3) uint8 RGB image

    Returns: points (N,3) float32 in world coords, colors (N,3) uint8
    """
    H, W = depth.shape

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Grid of pixel coordinates
    ys, xs = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing='ij')

    Z = depth
    valid = (Z > 0) & valid_mask
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy

    # Camera-space points (OpenCV convention: x right, y down, z forward)
    pts_cam = np.stack([X, Y, Z, np.ones_like(Z)], axis=-1)  # (H,W,4)

    # Flatten valid
    valid_flat = valid.reshape(-1)
    pts_cam_flat = pts_cam.reshape(-1, 4)[valid_flat]

    # Transform to world: c2w = inv(w2c)
    c2w = np.linalg.inv(w2c).astype(np.float32)
    pts_world_h = (c2w @ pts_cam_flat.T).T  # (N,4)
    pts_world = pts_world_h[:, :3].astype(np.float32)

    # Colors
    colors = rgb.reshape(-1, 3)[valid_flat]

    return pts_world, colors


def save_ply_ascii(output_path: Path, points: np.ndarray, colors: np.ndarray):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def main():
    parser = argparse.ArgumentParser(description='Project RGB-D to PLY using depth_warp loaders')
    parser.add_argument('--scene', required=True, help='Scene directory containing color/, depth/, camera/')
    parser.add_argument('--id', type=int, required=True, help='Frame id (e.g., 53)')
    parser.add_argument('--output', default=None, help='Output .ply path')
    args = parser.parse_args()

    scene = Path(args.scene)
    frame_id = args.id

    # Paths
    depth_path = scene / 'depth' / f'{frame_id:06d}.png'
    color_path = scene / 'color' / f'{frame_id:06d}.png'

    # Load depth (already converted to meters by load_depth) and valid mask
    depth, valid_mask = load_depth(str(depth_path))

    # Load camera intrinsics and extrinsics (world-to-camera)
    K, w2c = load_camera_poses(scene, frame_id)
    if K is None or w2c is None:
        raise RuntimeError(f'Camera data not found for frame {frame_id} in {scene}')

    # Load RGB as uint8
    img_bgr = cv2.imread(str(color_path))
    if img_bgr is None:
        raise FileNotFoundError(f'RGB image not found: {color_path}')
    rgb = img_bgr[:, :, ::-1]  # BGR->RGB

    # Sanity check resolution
    H, W = depth.shape
    if rgb.shape[0] != H or rgb.shape[1] != W:
        raise ValueError(f'Resolution mismatch: RGB {rgb.shape[:2]} vs Depth {(H, W)}')

    # Project to world
    points, colors = project_rgbd_to_pointcloud(depth, valid_mask, K, w2c, rgb)

    if args.output is None:
        out_dir = scene / 'pointcloud_out'
        output_path = out_dir / f'{frame_id:06d}.ply'
    else:
        output_path = Path(args.output)

    # Save PLY
    save_ply_ascii(output_path, points, colors)

    print(f'Saved point cloud: {output_path}')
    print(f'Total points: {len(points)}')


if __name__ == '__main__':
    main()
