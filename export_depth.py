import bpy
import os
import numpy as np


def export_depth(frame, depth_out, width, height):
    bpy.context.scene.frame_set(frame)

    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height
    bpy.context.scene.render.filepath = os.path.dirname(depth_out)

    # Enable depth pass
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # Clear default nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Create input render layer node
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    # Create output file node for depth (save as EXR to preserve float values)
    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    depth_file_output.base_path = ''
    depth_file_output.format.file_format = 'OPEN_EXR'
    depth_file_output.file_slots[0].path = 'depth_temp_'

    # Link render layer depth output to file output node
    links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])

    # Render the scene
    bpy.ops.render.render(write_still=False)

    # Read the depth data from the temporary EXR file
    depth_image_path = os.path.join(bpy.context.scene.render.filepath, 'depth_temp_00' + str(frame) + '.exr')
    depth_image = bpy.data.images.load(depth_image_path)

    # Convert depth data to numpy array
    depth_pixels = np.array(depth_image.pixels[:])
    depth_pixels = depth_pixels[::4]  # Take only the R channel (depth is stored in R)
    depth_array = depth_pixels.reshape((height, width))
    
    # Flip vertically as Blender images are stored bottom-up
    depth_array = np.flipud(depth_array)

    # Save as numpy array
    np.save(depth_out, depth_array)
    
    print(f"Depth array saved to: {depth_out}")
    print(f"Array shape: {depth_array.shape}, dtype: {depth_array.dtype}")
    print(f"Depth range: [{depth_array.min():.3f}, {depth_array.max():.3f}]")

    # Clean up temporary files
    bpy.data.images.remove(depth_image)
    if os.path.exists(depth_image_path):
        os.remove(depth_image_path)


if __name__ == '__main__':
    import sys
    import argparse
    
    # Parse command line arguments
    argv = sys.argv
    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', type=int, required=True, help='Frame number to render')
    parser.add_argument('--depth_out', type=str, required=True, help='Output depth image path')
    parser.add_argument('--width', type=int, default=512, help='Render width')
    parser.add_argument('--height', type=int, default=384, help='Render height')
    
    args = parser.parse_args(argv)
    
    export_depth(args.frame, args.depth_out, args.width, args.height)