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
    depth_path: 深度图路径（PNG/TIFF）或 npy文件路径（numpy数组）
    K: 相机内参矩阵3x3
    Rt: 相机外参矩阵4x4（世界到相机坐标系）
    
    深度值说明：
    - npy文件：直接从export_depth.py输出的浮点深度值（单位：Blender单位，通常为米）
    - 图像文件：16位PNG，范围0-65535，需要转换为实际深度值
    """
    if depth_path.endswith('.npy'):
        # 直接加载numpy数组，深度值为浮点数（米）
        depth = np.load(depth_path)
        depth = np.clip(depth, 0, None)  # 确保深度值非负
        print(f"Loaded depth from npy: shape={depth.shape}, dtype={depth.dtype}")
    else:
        # 加载深度图像
        depth = imageio.v2.imread(depth_path)
        if depth.ndim == 3:
            depth = depth[..., 0]
        # 将16位PNG深度图转换为实际深度值（假设最大深度30米）
        if depth.dtype == np.uint16:
            # depth = depth.astype(np.float32) / 65535.0 * 30.0
            depth = depth.astype(np.float32) / 1000.0  # 假设深度以毫米为单位存储，转换为米
        else:
            depth = depth.astype(np.float32)
        print(f"Loaded depth from image: shape={depth.shape}, dtype={depth.dtype}")
    
    # 加载RGB图像
    rgb = np.array(Image.open(rgb_path))
    
    # 检查分辨率匹配
    depth_height, depth_width = depth.shape
    rgb_height, rgb_width = rgb.shape[:2]
    if (depth_height != rgb_height) or (depth_width != rgb_width):
        raise ValueError(f"RGB图像和深度图分辨率不匹配！RGB: {rgb_height}x{rgb_width}, Depth: {depth_height}x{depth_width}")
    
    height, width = depth.shape
    
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    points = []
    colors = []

    bg_val = None
    if depth.dtype == np.float32 or depth.dtype == np.float64:
        # 对于浮点深度（npy文件），使用最大值作为背景
        bg_val = np.max(depth)
    else:
        # 对于整数深度图
        if depth.max() > 10000:
            bg_val = 65535
        else:
            bg_val = 255
    
    print(f"Depth background value: {bg_val}")
    print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")

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

            P_cam = np.array([X, -Y, -Z]).reshape(3, 1) # 
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
    
    print(f"Saved point cloud: {output_path}")
    print(f"Total points: {len(points)}")
    print(f"Depth unit: Blender units (typically meters)")

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