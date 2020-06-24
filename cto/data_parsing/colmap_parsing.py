import os
import numpy as np
import math
from PIL import Image as PILImage
import open3d as o3d

from cto.ext.colmap.read_write_model import read_model
from cto.ext.colmap.read_dense import read_array as read_colmap_array

from cto.conversion import convert_colmap_to_o3d_camera_trajectory
from cto.conversion import convert_color_depth_to_rgbd

from cto.utility.os_extension import get_corresponding_files_in_directories
from cto.utility.os_extension import mkdir_safely

from cto.config_api import get_reconstruction_mode

from cto.depth_map import create_depth_maps_from_mesh
from cto.config_api import depth_maps_from_mesh


def parse_colmap_camera_trajectory(colmap_workspace, config):

    reconstruction_mode = get_reconstruction_mode(config)
    model_ext = config.get_option_value(
        'model_ext', target_type=str, section=reconstruction_mode)

    colmap_camera_parameter_dict, colmap_image_parameter_dict, points3D_dict = read_model(
        colmap_workspace.model_idp, ext=model_ext)
    camera_trajectory, ordered_image_names = convert_colmap_to_o3d_camera_trajectory(
        colmap_camera_parameter_dict, colmap_image_parameter_dict, colmap_workspace)
    return camera_trajectory, ordered_image_names


def get_resized_image_fp_s(ordered_image_names, color_image_resized_dp):
    color_image_resized_fp_list = []
    for image_name in ordered_image_names:
        color_image_resized_fp_list.append(
            os.path.join(color_image_resized_dp, image_name))
    return color_image_resized_fp_list


def get_depth_array_fp_s(ordered_image_names, colmap_workspace, config):
    depth_array_ifp_list = []

    use_depth_maps_from_mesh = depth_maps_from_mesh(config)

    if use_depth_maps_from_mesh:
        suffix = colmap_workspace.depth_from_mesh_suffix
        depth_image_dp = colmap_workspace.depth_image_from_mesh_dp
    else:
        suffix = colmap_workspace.depth_map_suffix
        depth_image_dp = colmap_workspace.depth_image_idp

    for image_name in ordered_image_names:
        depth_array_ifp_list.append(
            os.path.join(depth_image_dp, image_name + suffix))
    return depth_array_ifp_list


def compute_depth_min_max(depth_array_ifp_list, use_depth_map_from_mesh):
    # Determine value range
    depth_map_min = float('inf')
    depth_map_max = -float('inf')
    for depth_image_ifp in depth_array_ifp_list:
        if use_depth_map_from_mesh:
            depth_arr = np.load(depth_image_ifp)
        else:
            depth_arr = read_colmap_array(depth_image_ifp)
        depth_map_min = min(depth_map_min, np.amin(depth_arr))
        depth_map_max = max(depth_map_max, np.amax(depth_arr))

    print('Depth Value Range (colmap):', "min:", depth_map_min, "max:", depth_map_max)
    return depth_map_min, depth_map_max


def write_resized_image_to_disc(ifp, new_width, new_height, ofp):

    pil_image = PILImage.open(ifp)
    pil_image = pil_image.resize((new_width, new_height), PILImage.BICUBIC)
    pil_image.save(ofp)


def compute_resized_images(colmap_workspace, lazy):

    color_image_ifp_list, depth_array_ifp_list = get_corresponding_files_in_directories(
        colmap_workspace.color_image_idp,
        colmap_workspace.depth_image_idp,
        suffix_2=colmap_workspace.depth_map_suffix)

    color_image_resized_dp = colmap_workspace.color_image_resized_dp
    mkdir_safely(color_image_resized_dp)
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
                color_image_resized_fp)
    return color_image_resized_dp



def compute_scaling_value(depth_map_min, depth_map_max):

    # https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html
    #   An uint16 can store up to 65535 values
    depth_map_max_possible_value = 65535.0

    depth_scale_value = depth_map_max_possible_value / depth_map_max
    #depth_scale_value = depth_map_max_possible_value / (depth_map_max - depth_map_min)
    depth_scale_value = float(math.floor(depth_scale_value))
    print('depth_scale_value:', depth_scale_value)
    assert depth_map_max * depth_scale_value < depth_map_max_possible_value
    return depth_scale_value


def parse_colmap_rgb_and_depth_data(camera_trajectory, ordered_image_names, mesh, colmap_workspace, config):

    color_image_resized_dp = compute_resized_images(colmap_workspace, lazy=True)
    color_image_resized_fp_list = get_resized_image_fp_s(ordered_image_names, color_image_resized_dp)

    use_depth_map_from_mesh = depth_maps_from_mesh(config)
    if use_depth_map_from_mesh:
        create_depth_maps_from_mesh(
            mesh,
            camera_trajectory,
            ordered_image_names,
            colmap_workspace.depth_image_from_mesh_dp,
            colmap_workspace.depth_from_mesh_suffix,
            depth_viz_odp=None,
            show_rendering=False,
            num_images=None)

    depth_array_ifp_list = get_depth_array_fp_s(ordered_image_names, colmap_workspace, config)
    depth_map_min, depth_map_max = compute_depth_min_max(
        depth_array_ifp_list, use_depth_map_from_mesh)

    depth_scale_value = compute_scaling_value(depth_map_min, depth_map_max)

    rgbd_images = []
    for color_image_resized_fp, depth_image_ifp in zip(
            color_image_resized_fp_list, depth_array_ifp_list):

        print('depth_image_ifp', depth_image_ifp)

        if use_depth_map_from_mesh:
            depth_arr = np.load(depth_image_ifp)
        else:
            depth_arr = read_colmap_array(depth_image_ifp)

        # Scale the values, so that the cast to uint16 does not remove accuracy
        # depth_arr = (depth_arr - np.full_like(depth_arr, depth_map_min)) * depth_scale_value
        depth_arr = depth_arr * depth_scale_value
        depth_arr = np.asarray(depth_arr, dtype=np.uint16)

        color_image = o3d.io.read_image(os.path.join(color_image_resized_fp))
        depth_image = o3d.geometry.Image(depth_arr)

        rgbd_image = convert_color_depth_to_rgbd(
            color_image,
            depth_image,
            depth_scale=depth_scale_value,
            convert_rgb_to_intensity=False)

        print("max bound", rgbd_image.get_max_bound())
        rgbd_images.append(rgbd_image)

    return rgbd_images, depth_map_max

