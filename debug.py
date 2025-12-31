import numpy as np
import argparse

def load_camera_intrinsics(path):
    K = np.loadtxt(path)
    print(f"Loaded K matrix from {path}:\n{K}\n")
    return K

def load_camera_extrinsics(path):
    RT = np.loadtxt(path)
    print(f"Loaded RT matrix from {path}:\n{RT}\n")
    return RT

def load_ply_points(ply_path):
    points = []
    with open(ply_path, 'r') as f:
        for line in f:
            if line.strip() == 'end_header':
                break
        for line in f:
            vals = line.strip().split()
            if len(vals) >= 3:
                points.append([float(vals[0]), float(vals[1]), float(vals[2])])
    points = np.array(points)
    print(f"Loaded {len(points)} points from {ply_path}\n")
    return points

def project_points_debug(points, K, RT, out_w, out_h, sample_num=10):
    points_h = np.concatenate([points, np.ones((points.shape[0],1))], axis=1)  # (N,4)
    cam_points = (RT @ points_h.T).T[:, :3]  # (N,3)

    z = cam_points[:,2]
    print(f"Depth (z) range: min={z.min()}, max={z.max()}")
    print(f"Points with positive depth: {(z>0).sum()} / {len(z)}\n")

    valid = z > 0
    cam_points_valid = cam_points[valid]
    points_valid = points[valid]

    xs = cam_points_valid[:,0] / cam_points_valid[:,2]
    ys = cam_points_valid[:,1] / cam_points_valid[:,2]

    u = K[0,0] * xs + K[0,2]
    v = K[1,1] * ys + K[1,2]

    print(f"u range: {u.min()} ~ {u.max()}")
    print(f"v range: {v.min()} ~ {v.max()}")
    in_img = (u >= 0) & (u < out_w) & (v >= 0) & (v < out_h)
    print(f"Points projected inside image: {np.sum(in_img)} / {len(u)}\n")

    # 采样打印点（相机坐标和对应u,v）
    step = max(1, len(cam_points_valid) // sample_num)
    print(f"Sample {sample_num} points camera coords and projection (u,v):\n")
    print(f"{'Index':>5} | {'X_c':>10} {'Y_c':>10} {'Z_c':>10} | {'u':>10} {'v':>10} {'InImg':>5}")
    print("-" * 60)
    for i in range(0, len(cam_points_valid), step):
        print(f"{i:5d} | "
              f"{cam_points_valid[i,0]:10.4f} {cam_points_valid[i,1]:10.4f} {cam_points_valid[i,2]:10.4f} | "
              f"{u[i]:10.2f} {v[i]:10.2f} {str(in_img[i]):>5}")
    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--K_path', required=True, help='相机内参矩阵文件路径')
    parser.add_argument('--RT_path', required=True, help='相机外参矩阵文件路径')
    parser.add_argument('--points_path', required=True, help='点云文件路径(ply格式)')
    parser.add_argument('--out_w', type=int, required=True, help='输出图像宽度')
    parser.add_argument('--out_h', type=int, required=True, help='输出图像高度')
    parser.add_argument('--sample_num', type=int, default=10, help='打印采样点数量')
    args = parser.parse_args()

    K = load_camera_intrinsics(args.K_path)
    RT = load_camera_extrinsics(args.RT_path)
    points = load_ply_points(args.points_path)
    project_points_debug(points, K, RT, args.out_w, args.out_h, args.sample_num)

if __name__ == '__main__':
    main()
