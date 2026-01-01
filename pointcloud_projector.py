import numpy as np
import imageio
import argparse
from PIL import Image
import os

def load_camera_intrinsics(path):
    return np.loadtxt(path) if path.endswith('.txt') else np.load(path)

def load_camera_extrinsics(path):
    return np.loadtxt(path) if path.endswith('.txt') else np.load(path)

def project_pointcloud_to_depth(points, colors, depth_path, K, Rt, out_h, out_w):
    # points: (N, 3) colors: (N, 3) depth_path: 新视角深度图路径 K: 3x3 Rt: 4x4
    depth = imageio.v2.imread(depth_path)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth / 65535 * 30
    R = Rt[:3, :3]
    T = Rt[:3, 3]
    points = R @ points.T
    points = points.T + T.reshape(1, 3)
    cam_points = points  # (N,3)
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    zs = -cam_points[:,2] # 深度    
    valid = zs > 0  # 仅保留正深度点
    for i in range(len(valid)):
        if not valid[i]:
            continue
        y_index = int((fy * (-cam_points[i,1] / zs[i]) + cy))
        x_index = int((fx * (cam_points[i,0] / zs[i]) + cx))
        if y_index < 0 or y_index >= depth.shape[0] or x_index < 0 or x_index >= depth.shape[1]:
            continue
        z_depth = depth[y_index, x_index]
        if zs[i] > z_depth * 1.1: # 被遮挡
            valid[i] = False
            continue

    cam_points = cam_points[valid]
    zs = zs[valid]
    colors = colors[valid]
    xs = cam_points[:,0] / zs # 归一化坐标
    ys = -cam_points[:,1] / zs # 归一化坐标
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    u = fx * xs + cx
    v = fy * ys + cy
    u = u.astype(int) # 投影到像素坐标
    v = v.astype(int) # 投影到像素坐标
    # 创建输出图像和mask
    img = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    # 调试：输出投影坐标和有效点数
    print(f"u range: {u.min()} ~ {u.max()}, v range: {v.min()} ~ {v.max()}")
    in_img = np.logical_and.reduce((u >= 0, u < out_w, v >= 0, v < out_h))
    print(f"投影在图像内的点数: {np.sum(in_img)} / {len(u)}")
    for idx in range(len(u)):
        if 0 <= u[idx] < out_w and 0 <= v[idx] < out_h:
            img[v[idx], u[idx]] = colors[idx]
            mask[v[idx], u[idx]] = 255
    return img, mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply', required=True, help='输入点云ply')
    parser.add_argument('--depth', required=False, help='新视角深度图路径（可选）')
    parser.add_argument('--intrinsics', required=True, help='新视角相机内参')
    parser.add_argument('--extrinsics', required=True, help='新视角相机外参4x4')
    parser.add_argument('--out_h', type=int, required=True, help='输出图像高')
    parser.add_argument('--out_w', type=int, required=True, help='输出图像宽')
    parser.add_argument('--rgb_out', required=True, help='输出RGB图像路径')
    parser.add_argument('--mask_out', required=True, help='输出mask路径')
    args = parser.parse_args()
    # 读取点云
    points, colors = [], []
    with open(args.ply) as f:
        for line in f:
            if line.startswith('end_header'):
                break
        for line in f:
            vals = line.strip().split()
            if len(vals) == 6:
                points.append([float(vals[0]), float(vals[1]), float(vals[2])])
                colors.append([int(vals[3]), int(vals[4]), int(vals[5])])
    points = np.array(points)
    colors = np.array(colors)
    K = load_camera_intrinsics(args.intrinsics)
    # 要求K矩阵与投影分辨率完全一致，否则报错
    cx, cy = K[0,2], K[1,2]
    expected_cx, expected_cy = args.out_w / 2.0, args.out_h / 2.0
    if abs(cx - expected_cx) > 1e-2 or abs(cy - expected_cy) > 1e-2:
        raise ValueError(f"K矩阵中心点(cx,cy)={cx},{cy} 与输出分辨率中心({expected_cx},{expected_cy})不一致，请重新导出K矩阵！")
    Rt = load_camera_extrinsics(args.extrinsics)
    img, mask = project_pointcloud_to_depth(points, colors, args.depth, K, Rt, args.out_h, args.out_w)
    Image.fromarray(img).save(args.rgb_out)
    Image.fromarray(mask).save(args.mask_out)
    print(f"Saved: {args.rgb_out}, {args.mask_out}")

if __name__ == '__main__':
    main()
