import bpy
import numpy as np
import sys
import argparse
import mathutils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', type=int, default=None, help='Blender帧号（从1开始）')
    parser.add_argument('--K_out', type=str, default='/tmp/camera_K.txt', help='K矩阵输出路径')
    parser.add_argument('--RT_out', type=str, default='/tmp/camera_RT.txt', help='RT矩阵输出路径')
    parser.add_argument('--width', type=int, default=None, help='渲染宽度')
    parser.add_argument('--height', type=int, default=None, help='渲染高度')
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])

    scene = bpy.context.scene
    cam_obj = scene.camera
    cam = cam_obj.data
    render = scene.render

    # 跳转到指定帧
    if args.frame is not None:
        scene.frame_set(args.frame)

    # 分辨率
    w = args.width if args.width is not None else render.resolution_x
    h = args.height if args.height is not None else render.resolution_y

    f_in_mm = cam.lens
    sensor_width_in_mm = cam.sensor_width
    sensor_height_in_mm = cam.sensor_height
    # pixel_aspect = render.pixel_aspect_y / render.pixel_aspect_x # 计算像素宽高比

    if cam.sensor_fit == 'VERTICAL' or (cam.sensor_fit == 'AUTO' and sensor_height_in_mm * w > sensor_width_in_mm * h):
        s_u = w / sensor_width_in_mm
        s_v = h / sensor_height_in_mm
    else:
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

    RT = cam_obj.matrix_world.inverted()
    RT = np.array(RT)

    # 保存内参和外参
    np.savetxt(args.K_out, K)
    print(f"K (分辨率: {w}x{h}):\n", K)

    np.savetxt(args.RT_out, RT)
    print("RT (世界到相机):\n", RT)

if __name__ == "__main__":
    main()