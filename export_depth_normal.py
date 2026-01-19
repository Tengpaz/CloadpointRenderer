import bpy
import os

# ======================
# 参数配置
# ======================
OUTPUT_ROOT = "//outputs"

DEPTH_DIR = "depth"
NORMAL_DIR = "normal"

DEPTH_NEAR = 0.0
DEPTH_FAR  = 65.535   # 根据场景调，越小对比越明显

scene = bpy.context.scene

# ======================
# 渲染设置
# ======================
scene.render.engine = "CYCLES"
scene.cycles.samples = 1
scene.use_nodes = True

scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"

# ======================
# View Layer Pass
# ======================
view_layer = scene.view_layers["ViewLayer"]
view_layer.use_pass_z = True
view_layer.use_pass_normal = True

# ======================
# Compositor
# ======================
tree = scene.node_tree
tree.nodes.clear()

rl = tree.nodes.new("CompositorNodeRLayers")
rl.location = (0, 0)

# ---------- Depth（16-bit PNG） ----------
depth_map = tree.nodes.new("CompositorNodeMapRange")
depth_map.inputs["From Min"].default_value = DEPTH_NEAR
depth_map.inputs["From Max"].default_value = DEPTH_FAR
depth_map.inputs["To Min"].default_value = 0.0
depth_map.inputs["To Max"].default_value = 1.0
# depth_map.clamp = True
depth_map.location = (300, 200)

depth_out = tree.nodes.new("CompositorNodeOutputFile")
depth_out.base_path = os.path.join(OUTPUT_ROOT, DEPTH_DIR)
depth_out.file_slots[0].path = "depth_"
depth_out.format.file_format = "PNG"
depth_out.format.color_mode = "BW"     # 单通道
depth_out.format.color_depth = "16"    # ⭐ 16-bit
depth_out.location = (600, 200)

tree.links.new(rl.outputs["Depth"], depth_map.inputs[0])
tree.links.new(depth_map.outputs[0], depth_out.inputs[0])

# ---------- Normal（RGB 可视化） ----------
normal_map = tree.nodes.new("CompositorNodeMapRange")
normal_map.inputs["From Min"].default_value = -1.0
normal_map.inputs["From Max"].default_value = 1.0
normal_map.inputs["To Min"].default_value = 0.0
normal_map.inputs["To Max"].default_value = 1.0
# normal_map.clamp = True
normal_map.location = (300, -200)

normal_out = tree.nodes.new("CompositorNodeOutputFile")
normal_out.base_path = os.path.join(OUTPUT_ROOT, NORMAL_DIR)
normal_out.file_slots[0].path = "normal_"
normal_out.format.file_format = "PNG"
normal_out.format.color_mode = "RGB"
normal_out.format.color_depth = "8"
normal_out.location = (600, -200)

tree.links.new(rl.outputs["Normal"], normal_map.inputs[0])
tree.links.new(normal_map.outputs[0], normal_out.inputs[0])

# ======================
# 创建输出目录
# ======================
def ensure_dir(p):
    os.makedirs(bpy.path.abspath(p), exist_ok=True)

ensure_dir(os.path.join(OUTPUT_ROOT, DEPTH_DIR))
ensure_dir(os.path.join(OUTPUT_ROOT, NORMAL_DIR))

# ======================
# 逐帧渲染
# ======================
for frame in range(0, 101):
    scene.frame_set(frame)
    print(f"[INFO] Render frame {frame}")
    scene.render.filepath = os.path.join(OUTPUT_ROOT, f"render_{frame:04d}.png")
    bpy.ops.render.render(write_still=True)

print("[DONE] 16-bit depth PNG & normal PNG exported.")
