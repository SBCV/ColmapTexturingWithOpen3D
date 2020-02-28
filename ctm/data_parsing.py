import os
import math
import numpy as np
import open3d as o3d
from PIL import Image as PILImage

from ctm.ext.o3d.file import get_file_list
from ctm.ext.colmap.read_dense import read_array as read_colmap_array
from ctm.ext.colmap.read_write_model import read_model
from ctm.conversion import convert_colmap_to_o3d_camera_trajectory


def parse_mesh(workspace):
    return o3d.io.read_triangle_mesh(
         workspace.mesh_ifp)


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


def convert_color_depth_to_rgbd(color,
                                depth,
                                depth_scale=1.0,
                                depth_trunc=float('inf'),
                                convert_rgb_to_intensity=False):
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=convert_rgb_to_intensity)


def parse_colmap_camera_trajectory(colmap_workspace):

    colmap_camera_parameter_dict, colmap_image_parameter_dict, points3D_dict = read_model(
        colmap_workspace.model_idp, ext='.bin')

    camera_trajectory, ordered_image_names = convert_colmap_to_o3d_camera_trajectory(
        colmap_camera_parameter_dict, colmap_image_parameter_dict)
    return camera_trajectory, ordered_image_names


def parse_colmap_rgb_and_depth_data(ordered_image_names, colmap_workspace, lazy=False):
    color_image_ifp_list = []
    depth_array_ifp_list = []
    for image_name in ordered_image_names:
        color_image_ifp_list.append(
            os.path.join(colmap_workspace.color_image_idp, image_name))
        depth_array_ifp_list.append(
            os.path.join(colmap_workspace.depth_image_idp, image_name + colmap_workspace.depth_map_suffix))

    color_image_resized_fp_list = resize_images(
        color_image_ifp_list, depth_array_ifp_list, colmap_workspace.color_image_resized_dp, lazy)

    depth_scale_value = compute_scaling_value(depth_array_ifp_list)

    rgbd_images = []
    for color_image_resized_fp, depth_image_ifp in zip(
            color_image_resized_fp_list, depth_array_ifp_list):

        depth_arr = read_colmap_array(depth_image_ifp)
        # Scale the values, so that the cast to uint16 does not remove accuracy
        depth_arr = depth_arr * depth_scale_value
        depth_arr = np.asarray(depth_arr, dtype=np.uint16)

        color_image = o3d.io.read_image(os.path.join(color_image_resized_fp))
        depth_image = o3d.geometry.Image(depth_arr)

        rgbd_image = convert_color_depth_to_rgbd(
            color_image,
            depth_image,
            depth_scale=depth_scale_value,
            convert_rgb_to_intensity=False)

        rgbd_images.append(rgbd_image)

    return rgbd_images


def compute_scaling_value(depth_array_ifp_list):

    depth_map_min, depth_map_max = compute_depth_min_max(
        depth_array_ifp_list)

    # https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html
    #   An uint16 can store up to 65535 values
    depth_map_max_possible_value = 65535.0
    depth_scale_value = compute_best_scaling_value(
        depth_map_min,
        depth_map_max,
        depth_map_max_possible_value)

    return depth_scale_value


def compute_depth_min_max(depth_array_ifp_list):
    # Determine value range
    depth_map_min = float('inf')
    depth_map_max = -float('inf')
    for depth_image_ifp in depth_array_ifp_list:
        depth_arr = read_colmap_array(
            depth_image_ifp)
        depth_map_min = min(depth_map_min, np.amin(depth_arr))
        depth_map_max = max(depth_map_max, np.amax(depth_arr))

    # > Depth Value Range (colmap): min: 0.0 max: 50.510017
    print('Depth Value Range (colmap):', "min:", depth_map_min, "max:", depth_map_max)
    return depth_map_min, depth_map_max


def compute_best_scaling_value(depth_map_min, depth_map_max, depth_map_max_possible_value):

    assert depth_map_min >= 0.0
    depth_scale_value = depth_map_max_possible_value / depth_map_max
    depth_scale_value = float(math.floor(depth_scale_value))
    print('depth_scale_value:', depth_scale_value)
    assert depth_map_max * depth_scale_value < depth_map_max_possible_value
    return depth_scale_value


def resize_images(color_image_ifp_list, depth_array_ifp_list, color_image_resized_dp, lazy):
    mkdir_safely(color_image_resized_dp)
    color_image_resized_fp_list = []
    for color_image_ifp, depth_image_ifp in zip(color_image_ifp_list, depth_array_ifp_list):
        color_image_resized_fp = os.path.join(
            color_image_resized_dp, os.path.basename(color_image_ifp))
        if not os.path.isfile(color_image_resized_fp) or not lazy:
            depth_arr = read_colmap_array(
                depth_image_ifp)
            height, width = depth_arr.shape
            write_resized_image_to_disc(
                color_image_ifp,
                width,
                height,
                color_image_resized_fp
            )
        color_image_resized_fp_list.append(color_image_resized_fp)
    return color_image_resized_fp_list


def write_resized_image_to_disc(ifp, new_width, new_height, ofp):

    pil_image = PILImage.open(ifp)
    pil_image = pil_image.resize((new_width, new_height), PILImage.BICUBIC)
    pil_image.save(ofp)


def mkdir_safely(odp):
    if not os.path.isdir(odp):
        os.mkdir(odp)