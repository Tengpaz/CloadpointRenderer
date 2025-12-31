import numpy as np
import imageio
import argparse
import os
from PIL import Image

def load_camera_intrinsics(path):
    return np.loadtxt(path) if path.endswith('.txt') else np.load(path)

def load_camera_extrinsics(path):
    return np.loadtxt(path) if path.endswith('.txt') else np.load(path)

def rgb_depth_to_pointcloud(rgb_path, depth_path, K, Rt, output_path):
    """
    rgb_path: RGB图像路径
    depth_path: 深度图路径
    K: 相机内参矩阵3x3
    Rt: 相机外参矩阵4x4（世界到相机坐标系）
    """
    depth = imageio.v2.imread(depth_path)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth / 65535 * 30
    rgb = np.array(Image.open(rgb_path))
    depth_height, depth_width = depth.shape
    rgb_height, rgb_width = rgb.shape[:2]
    if (depth_height != rgb_height) or (depth_width != rgb_width):
        raise ValueError("RGB图像和深度图分辨率不匹配！")
    height, width = depth.shape
    
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    points = []
    colors = []

    bg_val = None
    if depth.dtype == np.float32 or depth.dtype == np.float64:
        bg_val = np.max(depth)
    else:
        if depth.max() > 10000:
            bg_val = 65535
        else:
            bg_val = 255
    print("depth background value:", bg_val)
    print("depth min max:", depth.min(), depth.max())

    R = Rt[:3, :3]
    T = Rt[:3, 3]
    R = np.linalg.inv(R)
    T = -T

    for v in range(height):
        for u in range(width):
            Z = depth[v, u].astype(np.float32)
            if Z == 0 or Z >= bg_val * 0.99:
                continue  # 跳过无效深度
            
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy

            P_cam = np.array([X, Y, Z]).reshape(3, 1)
            P_cam = P_cam + T.reshape(3, 1)
            P_world = R @ P_cam
            points.append(P_world.flatten())
            colors.append(rgb[v, u])

    points = np.array(points)
    colors = np.array(colors)

    # 保存为ply格式
    with open(output_path, 'w') as f:
        f.write('ply\nformat ascii 1.0\nelement vertex {}\n'.format(len(points)))
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
    print(f"Saved point cloud: {output_path}, points: {len(points)} (depth assumed mm, converted to meters)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb', required=True, help='RGB图像路径')
    parser.add_argument('--depth', required=True, help='深度图路径')
    parser.add_argument('--intrinsics', required=True, help='相机内参路径')
    parser.add_argument('--extrinsics', help='相机外参路径（可选）')
    parser.add_argument('--output', required=True, help='输出ply路径')
    args = parser.parse_args()
    K = load_camera_intrinsics(args.intrinsics)
    Rt = load_camera_extrinsics(args.extrinsics)
    rgb_depth_to_pointcloud(args.rgb, args.depth, K, Rt, args.output)