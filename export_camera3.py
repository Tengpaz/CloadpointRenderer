import bpy
import numpy as np
import sys
import argparse
import mathutils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', type=int, default=None, help='Blender帧号（从1开始）')
    parser.add_argument('--K_out', type=str, default='/tmp/camera_K.txt', help='K矩阵输出路径')
    parser.add_argument('--R_out', type=str, default='/tmp/camera_R.txt', help='RT矩阵输出路径')
    parser.add_argument('--T_out', type=str, default='/tmp/camera_T.txt', help='T矩阵输出路径')
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])

    scene = bpy.context.scene
    cam_obj = scene.camera
    cam = cam_obj.data
    render = scene.render

    # 跳转到指定帧
    if args.frame is not None:
        scene.frame_set(args.frame)

    # 分辨率
    w = render.resolution_x
    h = render.resolution_y

    f_in_mm = cam.lens
    sensor_width_in_mm = cam.sensor_width
    sensor_height_in_mm = cam.sensor_height

    s_u = w / sensor_width_in_mm
    s_v = h / sensor_height_in_mm

    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = w / 2.0
    v_0 = h / 2.0

    K = np.array([
        [alpha_u, 0, u_0],
        [0, alpha_v, v_0],
        [0, 0, 1]
    ])

    R = cam_obj.matrix_world.to_3x3()
    T = cam_obj.matrix_world.translation

    RT = cam_obj.matrix_world.inverted()
    RT = np.array(RT)

    # 保存内参和外参
    np.savetxt(args.K_out, K)
    print(f"K (分辨率: {w}x{h}):\n", K)

    np.savetxt(args.R_out, R)
    print("R (相机到世界):\n", R)

    np.savetxt(args.T_out, T)
    print("T (相机到世界):\n", T)

if __name__ == "__main__":
    main()