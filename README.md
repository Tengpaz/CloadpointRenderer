# 3D点云投影与视角变换工具

## 环境配置

```bash
conda create -n renderer python=3.10
conda activate renderer
pip install -r requirements.txt
```

## 功能

- 指定相机分辨率导出blender相机内外参数（相机最好不要有Scale缩放，会导致最后生成的点云存在畸变）
- 输入RGB帧、深度图（16位图像）、blender相机内外参数、分辨率大小（与相机内外参数匹配），生成对应视角的点云文件（ply格式，坐标系为blender世界坐标系）。
- 输入点云文件（ply格式）和新视角的深度图、相机内外参数，投影生成新视角下的RGB图和mask图。

## 可能存在的问题

- blender相机有Scale缩放时，导出的点云会存在畸变，建议将Scale设置为1。
- 深度图单位不为毫米固定量纲时（blender默认为米），生成的点云会存在畸变，建议保持一致，如果无法保持一致，可以通过修改`rgbd_to_pointcloud.py`中对depth的计算方式进行调整。

## 0. 导出相机矩阵

```bash
blender -b town.blend --python export_camera.py -- --frame 60 --K_out camera_K_frame60.txt --RT_out camera_RT_frame60.txt --width 720 --height 480
```

- blender： blender可执行文件路径
- `town.blend`：包含相机动画的blender文件
- `--frame 50`：导出第50帧的相机参数
- `--K_out camera_K_frame50.txt`：输出相机内参文件路径
- `--RT_out camera_RT_frame50.txt`：输出相机外参文件路径
- `--width 720 --height 480`：导出分辨率，需要与相机参数一致

调试

```bash
python renderer/debug.py \
--K_path renderer/camera_K_frame50.txt \
--RT_path renderer/camera_RT_frame50.txt \
--points_path town_frame1.ply \
--out_w 720 --out_h 480 \
--sample_num 20
```

## 0. 导出depth, normal

```bash
blender -b town.blend -P export_depth_normal.py
```

在export_depth_normal.py中存在以下参数

- OUTPUT_ROOT 输出路径
- DEPTH_DIR 输出路径中存放depth的文件夹
- NORMAL_DIR 输出路径中存放normal的文件夹
- DEPTH_NEAR depth深度值最小值
- DEPTH_FAR depth深度值最大值，当值为65.535时输出的16位深度图是无归一化的
- PNGHEIGHT 输出的图片的高
- PNGWIDTH 输出的图片的宽
- FRAME_START 开始帧数
- FRAME_END 结束帧数

- town.blend: 3D模型路径

## 1. RGBD帧生成世界点云

```bash
python rgbd_to_pointcloud.py \
  --rgb /vePFS-MLP/buaa/wangyuzhen/CloadpointRenderer/outputs/normal/normal_0060.png \
  --depth /vePFS-MLP/buaa/wangyuzhen/CloadpointRenderer/outputs/depth/depth_0060.png \
  --intrinsics camera_K_frame60.txt \
  --extrinsics camera_RT_frame60.txt \
  --output town_frame60.ply
```

- `town_rgb_1.png`：第1帧的RGB图片
- `town_depth_1.png`：第1帧的深度图（单位为mm）
- `camera_K.txt`：3x3相机内参矩阵（txt）
- `camera_RT.txt`：4x4相机外参矩阵（txt，世界到相机）
- `town_frame1.ply`：输出点云

## 2. 点云投影到新视角

```bash
python pointcloud_projector.py \
  --ply town_frame50.ply \
  --depth depth_00060.png \
  --intrinsics camera_K_frame60.txt \
  --extrinsics camera_RT_frame60.txt \
  --out_h 480 --out_w 720 \
  --rgb_out proj_rgb_60.png \
  --mask_out proj_mask_60.png
```

- `town_frame1.ply`：上一步生成的点云
- `depth.png`：新视角深度图
- `camera_K.txt`：新视角相机内参
- `camera_3_Rt.txt`：新视角相机4x4外参（世界到相机）
- `proj_rgb_3.png`：新视角下的RGB图路径（有纹理为原色，无纹理为黑色）
- `proj_mask_3.png`：新视角下的mask路径（有纹理为白，无为黑）

## 说明

- 点云生成与投影均支持自定义相机参数。
- 未被点云投影到的像素自动为黑色，mask为黑色。
- 适用于blender导出的深度与相机参数。