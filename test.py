import open3d as o3d

pcd1 = o3d.io.read_point_cloud("../town_frame50.ply")
pcd2 = o3d.io.read_point_cloud("../town_frame60.ply")
pcd1.paint_uniform_color([1, 0, 0])  # 红色
pcd2.paint_uniform_color([0, 1, 0])  # 绿色

o3d.visualization.draw_geometries([pcd1, pcd2])