# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/color_map_optimization.py

from enum import Enum
from shutil import copyfile
import open3d as o3d

import os

from cto.data_parsing import parse_mesh
from cto.data_parsing import resize_images
from cto.data_parsing import parse_colmap_camera_trajectory
from cto.data_parsing import parse_colmap_rgb_and_depth_data
from cto.data_parsing import parse_o3d_trajectory
from cto.data_parsing import parse_o3d_data
from cto.workspace import ColmapWorkspace
from cto.workspace import O3DWorkspace
from cto.utility.config import Config
from cto.utility.logging_extension import logger
from cto.visualization import visualize_rgbd_image_list


class ReconstructionMode(Enum):
    Open3D = 'Open3D'
    Colamp = 'Colmap'


def get_reconstruction_mode(config):
    reconstruction_mode = config.get_option_value('reconstruction_mode', target_type=str)
    assert reconstruction_mode in [ReconstructionMode.Open3D.value, ReconstructionMode.Colamp.value]
    return reconstruction_mode


def get_dataset_idp(config):
    reconstruction_mode = get_reconstruction_mode(config)
    dataset_idp = config.get_option_value(
        'dataset_idp', target_type=str, section=reconstruction_mode)
    logger.vinfo('dataset_idp', dataset_idp)
    return dataset_idp


def create_config():
    config_fp = 'configs/cto.cfg'
    config_with_default_values_fp = 'configs/cto_default_values.cfg'
    if not os.path.isfile(config_fp):
        logger.info('Config file missing ... create a copy from config with default values.')
        copyfile(os.path.abspath(config_with_default_values_fp), os.path.abspath(config_fp))
        logger.info('Adjust the input path in the created config file (cto.cfg)')
        assert False
    return Config(config_fp=config_fp)


def compute_ofp(config):
    dataset_idp = get_dataset_idp(config)
    reconstruction_mode = get_reconstruction_mode(config)
    mesh_textured_max_iter_x_ofn = config.get_option_value(
        'mesh_textured_max_iter_x_ofn', target_type=str)
    maximum_iteration = config.get_option_value(
        'maximum_iteration', target_type=int, section=reconstruction_mode)
    mesh_textured_max_iter_x_ofn = mesh_textured_max_iter_x_ofn.replace('_x.', '_' + str(maximum_iteration) + '.')
    return os.path.join(dataset_idp, mesh_textured_max_iter_x_ofn)


def import_reconstruction(config):
    dataset_idp = get_dataset_idp(config)
    reconstruction_mode = get_reconstruction_mode(config)
    if reconstruction_mode.lower() == 'colmap':
        colmap_workspace = ColmapWorkspace(
            dataset_idp, use_geometric_depth_maps=True, use_poisson=True)
        resize_images(colmap_workspace, lazy=True)
        camera_trajectory, ordered_image_names = parse_colmap_camera_trajectory(
            colmap_workspace)
        rgbd_images = parse_colmap_rgb_and_depth_data(
            ordered_image_names, colmap_workspace)
        mesh = parse_mesh(colmap_workspace)
    elif reconstruction_mode.lower() == 'open3d':
        o3d_workspace = O3DWorkspace(dataset_idp)
        camera_trajectory = parse_o3d_trajectory(o3d_workspace)
        rgbd_images = parse_o3d_data(o3d_workspace)
        mesh = parse_mesh(o3d_workspace)
    else:
        assert False
    return rgbd_images, camera_trajectory, mesh


def visualize_intermediate_result(rgbd_images, camera_trajectory, mesh, config):
    viz_im_points = config.get_option_value('visualize_intermediate_points', target_type=bool)
    viz_im_mesh = config.get_option_value('visualize_intermediate_mesh', target_type=bool)
    if viz_im_points or viz_im_mesh:
        if viz_im_mesh:
            additional_point_cloud_list = [mesh]
        else:
            additional_point_cloud_list = []
        visualize_rgbd_image_list(
            rgbd_images,
            camera_trajectory,
            additional_point_cloud_list=additional_point_cloud_list)


def color_map_optimization(mesh,
                           rgbd_images,
                           camera_trajectory,
                           ofp,
                           config,
                           maximum_iteration=None):
    reconstruction_mode = get_reconstruction_mode(config)

    # Optimize texture and save the mesh as texture_mapped.ply
    # This is implementation of following paper: "Q.-Y. Zhou and V. Koltun,
    # Color Map Optimization for 3D Reconstruction with Consumer Depth Cameras, SIGGRAPH 2014"

    # Check out default option values here
    #   http://www.open3d.org/docs/latest/python_api/open3d.color_map.ColorMapOptimizationOption.html
    # option.number_of_vertical_anchors = 16
    # option.non_rigid_anchor_point_weight = 0.316
    # option.depth_threshold_for_discontinuity_check = 0.1
    # option.half_dilation_kernel_size_for_discontinuity_map = 3
    # option.image_boundary_margin = 10
    # option.invisible_vertex_color_knn = 3

    option = o3d.color_map.ColorMapOptimizationOption()
    option.non_rigid_camera_coordinate = config.get_option_value(
        'non_rigid_camera_coordinate', target_type=bool, section=reconstruction_mode)
    option.maximum_allowable_depth = config.get_option_value(
        'maximum_allowable_depth', target_type=float, section=reconstruction_mode)

    if maximum_iteration is not None:
        option.maximum_iteration = maximum_iteration
    else:
        option.maximum_iteration = config.get_option_value(
            'maximum_iteration', target_type=int, section=reconstruction_mode)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.color_map.color_map_optimization(mesh, rgbd_images, camera_trajectory, option)
    o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh(ofp, mesh)


if __name__ == "__main__":

    # http://www.open3d.org/docs/release/tutorial/Advanced/color_map_optimization.html
    logger.vinfo('o3d.__version__', o3d.__version__)

    o3d.utility.set_verbosity_level(
        o3d.utility.VerbosityLevel.Debug)

    config = create_config()
    mesh_textured_max_iter_x_ofp = compute_ofp(config)
    rgbd_images, camera_trajectory, mesh = import_reconstruction(config)

    visualize_intermediate_result(rgbd_images, camera_trajectory, mesh, config)

    color_map_optimization(
        mesh,
        rgbd_images,
        camera_trajectory,
        ofp=mesh_textured_max_iter_x_ofp,
        config=config)

