import os
import numpy as np
import open3d as o3d
from cto.ext.o3d.file import get_file_list
from cto.conversion import convert_color_depth_to_rgbd


def parse_o3d_trajectory(o3d_workspace):
    # http://www.open3d.org/docs/release/python_api/open3d.io.read_pinhole_camera_trajectory.html
    return o3d.io.read_pinhole_camera_trajectory(
        o3d_workspace.camera_traj_ifp)


def parse_o3d_data(o3d_workspace):

    depth_image_ifp_list = get_file_list(
        o3d_workspace.depth_image_idp, extension=".png")
    color_image_ifp_list = get_file_list(
        o3d_workspace.color_image_idp, extension=".jpg")
    assert (len(depth_image_ifp_list) == len(color_image_ifp_list))

    # Read RGBD images
    rgbd_images = []

    # Determine value range
    depth_map_min = float('inf')
    depth_map_max = -float('inf')
    for i in range(len(depth_image_ifp_list)):
        depth = o3d.io.read_image(os.path.join(depth_image_ifp_list[i]))
        color = o3d.io.read_image(os.path.join(color_image_ifp_list[i]))

        depth_map_min = min(depth_map_min, np.amin(depth))
        depth_map_max = max(depth_map_max, np.amax(depth))

        rgbd_image = convert_color_depth_to_rgbd(color, depth, depth_scale=1000.0)
        rgbd_images.append(rgbd_image)

    print('Depth Value Range (o3d):', "min:", depth_map_min, "max:", depth_map_max)

    return rgbd_images
