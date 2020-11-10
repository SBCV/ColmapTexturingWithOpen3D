import os
import numpy as np
import math
from PIL import Image as PILImage
import open3d as o3d

from cto.ext.colmap.read_write_model import read_model
from cto.ext.colmap.read_dense import read_array as read_colmap_array
from cto.data_parsing.colmap_depth_map_handler import ColmapDepthMapHandler

from cto.conversion import convert_colmap_to_o3d_camera_trajectory
from cto.conversion import convert_color_depth_to_rgbd

from cto.utility.logging_extension import logger
from cto.utility.os_extension import get_corresponding_files_in_directories
from cto.utility.os_extension import mkdir_safely

from cto.utility.cache import Cache
from cto.config_api import get_caching_flag

def compute_model_fps(model_idp, ext):
    cameras_fp = os.path.join(model_idp, "cameras" + ext)
    images_fp = os.path.join(model_idp, "images" + ext)
    points3D_fp = os.path.join(model_idp, "points3D" + ext)
    return cameras_fp, images_fp, points3D_fp


def check_model_completness(cameras_fp, images_fp, points3D_fp):
    if not os.path.isfile(cameras_fp):
        return False
    if not os.path.isfile(images_fp):
        return False
    if not os.path.isfile(points3D_fp):
        return False
    return True


def examine_model_format(model_idp):
    txt_ext = '.txt'
    cameras_txt_fp, images_txt_fp, points3D_txt_fp = compute_model_fps(model_idp, txt_ext)
    txt_model_present = check_model_completness(cameras_txt_fp, images_txt_fp, points3D_txt_fp)

    bin_ext = '.bin'
    cameras_bin_fp, images_bin_fp, points3D_bin_fp = compute_model_fps(model_idp, bin_ext)
    bin_model_present = check_model_completness(cameras_bin_fp, images_bin_fp, points3D_bin_fp)

    logger.vinfo('txt_model_present', txt_model_present)
    logger.vinfo('bin_model_present', bin_model_present)
    logger.vinfo('model_idp', str(model_idp))

    # If both model formats are present, we use the txt format
    if txt_model_present:
        logger.info('Found TXT model in ' + str(model_idp))
        return txt_ext
    else:
        logger.info('Found BIN model in ' + str(model_idp))
        return bin_ext


def parse_colmap_camera_trajectory(colmap_workspace, config):

    model_ext = examine_model_format(colmap_workspace.model_idp)
    if get_caching_flag(config):
        cache = Cache()
        colmap_camera_parameter_dict, colmap_image_parameter_dict, points3D_dict = cache.get_cached_result(
            callback=read_model,
            params=[colmap_workspace.model_idp, model_ext],
            unique_id_or_path=1)
    else:
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


def write_resized_image_to_disc(ifp, new_width, new_height, ofp):

    pil_image = PILImage.open(ifp)
    pil_image = pil_image.resize((new_width, new_height), PILImage.BICUBIC)
    pil_image.save(ofp)


def compute_resized_images(colmap_workspace, lazy):

    color_image_ifp_list, depth_array_ifp_list = get_corresponding_files_in_directories(
        colmap_workspace.color_image_idp,
        colmap_workspace.depth_map_idp,
        suffix_2=colmap_workspace.depth_map_suffix)

    color_image_resized_dp = colmap_workspace.color_image_resized_dp
    mkdir_safely(color_image_resized_dp)
    for color_image_ifp, depth_map_ifp in zip(color_image_ifp_list, depth_array_ifp_list):
        color_image_resized_fp = os.path.join(
            color_image_resized_dp, os.path.basename(color_image_ifp))
        if not os.path.isfile(color_image_resized_fp) or not lazy:
            depth_arr = read_colmap_array(
                depth_map_ifp)
            height, width = depth_arr.shape
            write_resized_image_to_disc(
                color_image_ifp,
                width,
                height,
                color_image_resized_fp)
    return color_image_resized_dp


def compute_scaling_value(depth_map_max):

    # https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html
    #   An uint16 can store up to 65535 values
    depth_map_max_possible_value = 65535.0

    # TODO WHAT IF depth_map_max > depth_map_max_possible_value
    if depth_map_max < depth_map_max_possible_value:
        # To ensure that there are no overflow values we use 65534.0
        depth_scale_value = (depth_map_max_possible_value - 1) / depth_map_max
        # depth_scale_value = float(math.floor(depth_scale_value))
    elif depth_map_max < depth_map_max_possible_value:
        depth_scale_value = depth_map_max / (depth_map_max_possible_value + 1)
        # depth_scale_value = float(math.floor(depth_scale_value))
    else:
        depth_scale_value = 1.0

    logger.vinfo('depth_map_max:', depth_map_max)
    logger.vinfo('depth_scale_value:', depth_scale_value)
    assert depth_map_max * depth_scale_value < depth_map_max_possible_value
    return depth_scale_value


def parse_colmap_rgb_and_depth_data(camera_trajectory, ordered_image_names, colmap_workspace, config):

    color_image_resized_dp = compute_resized_images(colmap_workspace, lazy=True)
    color_image_resized_fp_list = get_resized_image_fp_s(ordered_image_names, color_image_resized_dp)

    depth_map_handler = ColmapDepthMapHandler(
        colmap_workspace,
        config)

    depth_map_handler.process_depth_maps(
        camera_trajectory,
        ordered_image_names)

    depth_array_ifp_list = depth_map_handler.get_depth_array_fp_s(ordered_image_names)
    depth_map_range = depth_map_handler.compute_depth_statistics(
        depth_array_ifp_list)

    overall_depth_map_max = depth_map_range[1]
    depth_scale_value = compute_scaling_value(overall_depth_map_max)

    rgbd_images = []
    for color_image_resized_fp, depth_map_ifp in zip(
            color_image_resized_fp_list, depth_array_ifp_list):

        logger.vinfo('depth_map_ifp', depth_map_ifp)

        depth_arr = depth_map_handler.read_depth_map(depth_map_ifp)
        depth_arr[depth_arr < 0] = 0
        depth_arr_min, depth_arr_max = ColmapDepthMapHandler.compute_depth_map_min_max(
            depth_arr)
        logger.vinfo('depth_arr_min', depth_arr_min)
        logger.vinfo('depth_arr_max', depth_arr_max)

        if depth_arr_min < 0:
            logger.vinfo('depth_arr_min', depth_arr_min)
            assert False

        # Scale the values, so that the cast to uint16 does not remove accuracy
        # depth_arr = (depth_arr - np.full_like(depth_arr, depth_map_min)) * depth_scale_value
        depth_arr_scaled = depth_arr * depth_scale_value
        depth_arr_scaled_min, depth_arr_scaled_max = ColmapDepthMapHandler.compute_depth_map_min_max(
            depth_arr_scaled)
        depth_arr_scaled = np.asarray(depth_arr_scaled, dtype=np.uint16)

        color_image = o3d.io.read_image(os.path.join(color_image_resized_fp))
        depth_map = o3d.geometry.Image(depth_arr_scaled)

        logger.vinfo('depth_arr_scaled_min', depth_arr_scaled_min)
        logger.vinfo('depth_arr_scaled_max', depth_arr_scaled_max)

        rgbd_image = convert_color_depth_to_rgbd(
            color_image,
            depth_map,
            depth_scale=depth_scale_value,
            convert_rgb_to_intensity=False)

        # http://www.open3d.org/docs/release/python_api/open3d.geometry.RGBDImage.html
        # logger.vinfo("dimension", rgbd_image.dimension())
        # logger.vinfo("min bound", rgbd_image.get_min_bound())
        # logger.vinfo("max bound", rgbd_image.get_max_bound())
        rgbd_images.append(rgbd_image)

    return rgbd_images, depth_map_range

