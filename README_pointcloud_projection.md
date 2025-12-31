# 3D点云投影与视角变换工具

## 0. 导出相机矩阵

```bash
/apdcephfs_cq5/share_300600172/suanhuang/users/wangyuzhen/WorldRenderer/blender/blender-5.0.0-linux-x64/blender -b renderer/town.blend --python renderer/export_camera2.py -- --frame 50 --K_out renderer/camera_K_frame50.txt --RT_out renderer/camera_RT_frame50.txt --width 512 --height 384
```

```bash
/apdcephfs_cq5/share_300600172/suanhuang/users/wangyuzhen/WorldRenderer/blender/blender-5.0.0-linux-x64/blender -b renderer/town.blend --python renderer/export_camera3.py -- --frame 50 --K_out renderer/camera_K3_frame50.txt --R_out renderer/camera_R3_frame50.txt --T_out renderer/camera_T3_frame50.txt
```

调试

```bash
python renderer/debug.py \
--K_path renderer/camera_K_frame50.txt \
--RT_path renderer/camera_RT_frame50.txt \
--points_path town_frame1.ply \
--out_w 512 --out_h 384 \
--sample_num 20
```

## 1. RGBD帧生成世界点云

```bash
python renderer/rgbd_to_pointcloud.py \
  --rgb renderer/frame_00051.png \
  --depth renderer/depth_00050.png \
  --intrinsics renderer/camera_K_frame50.txt \
  --extrinsics renderer/camera_RT_frame50.txt \
  --output town_frame50.ply
```
- `town_rgb_1.png`：第1帧的RGB图片
- `town_depth_1.png`：第1帧的深度图（单位与相机一致）
- `camera_K.txt`：3x3相机内参矩阵（txt或npy）
- `camera_RT.txt`：4x4相机外参矩阵（txt或npy）
- `town_frame1.ply`：输出点云

```bash
python renderer/rgbd_to_pointcloud.py \
  --rgb renderer/frame_00051.png \
  --depth renderer/depth_00050.png \
  --intrinsics renderer/camera_K3_frame50.txt \
  --extrinsics renderer/camera_R3_frame50.txt \
  --output town_frame60.ply
```

## 2. 点云投影到新视角

```bash
python renderer/pointcloud_projector.py \
  --ply town_frame50.ply \
  --depth renderer/depth_00060.png \
  --intrinsics renderer/camera_K_frame60.txt \
  --extrinsics renderer/camera_RT_frame60.txt \
  --out_h 384 --out_w 512 \
  --rgb_out proj_rgb_60.png \
  --mask_out proj_mask_60.png
```
- `town_frame1.ply`：上一步生成的点云
- `depth.png`：新视角深度图
- `camera_K.txt`：新视角相机内参
- `camera_3_Rt.txt`：新视角相机4x4外参（世界到相机）
- `proj_rgb_3.png`：新视角下的RGB图（有纹理为原色，无纹理为黑色）
- `proj_mask_3.png`：新视角下的mask（有纹理为白，无为黑）

## 说明
- 点云生成与投影均支持自定义相机参数。
- 未被点云投影到的像素自动为黑色，mask为黑色。
- 适用于blender导出的深度与相机参数。