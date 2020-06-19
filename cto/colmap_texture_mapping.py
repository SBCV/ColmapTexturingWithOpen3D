# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/color_map_optimization.py

from enum import Enum
from shutil import copyfile
import open3d as o3d

import os

from cto.data_parsing import parse_mesh
from cto.data_parsing import parse_colmap_camera_trajectory
from cto.data_parsing import parse_colmap_rgb_and_depth_data
from cto.data_parsing import parse_o3d_trajectory
from cto.data_parsing import parse_o3d_data
from cto.workspace import ColmapWorkspace
from cto.workspace import O3DWorkspace
from cto.utility.config import Config
from cto.utility.logging_extension import logger
from cto.visualization import visualize_rgbd_image_list


class Settings(Enum):
    Open3D = 'Open3D'
    Colamp = 'Colmap'


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
    mesh_textured_max_iter_x_ofn = config.get_option_value(
        'mesh_textured_max_iter_x_ofn', target_type=str)
    maximum_iteration = config.get_option_value(
        'maximum_iteration', target_type=int, section=settings)
    mesh_textured_max_iter_x_ofn = mesh_textured_max_iter_x_ofn.replace('_x.', '_' + str(maximum_iteration) + '.')
    return os.path.join(dataset_idp, mesh_textured_max_iter_x_ofn)


def import_reconstruction(settings):
    if settings.lower() == 'colmap':
        colmap_workspace = ColmapWorkspace(
            dataset_idp, use_geometric_depth_maps=True, use_poisson=True)
        camera_trajectory, ordered_image_names = parse_colmap_camera_trajectory(colmap_workspace)
        rgbd_images = parse_colmap_rgb_and_depth_data(
            ordered_image_names, colmap_workspace, lazy=True)
        mesh = parse_mesh(colmap_workspace)
    elif settings.lower() == 'open3d':
        o3d_workspace = O3DWorkspace(dataset_idp)
        camera_trajectory = parse_o3d_trajectory(o3d_workspace)
        rgbd_images = parse_o3d_data(o3d_workspace)
        mesh = parse_mesh(o3d_workspace)
    else:
        assert False
    return rgbd_images, camera_trajectory, mesh


def visualize_intermediate_result(config, rgbd_images, camera_trajectory, mesh):
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
                           settings,
                           maximum_iteration=None):

    assert settings in [Settings.Open3D.value, Settings.Colamp.value]

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
        'non_rigid_camera_coordinate', target_type=bool, section=settings)
    option.maximum_allowable_depth = config.get_option_value(
        'maximum_allowable_depth', target_type=float, section=settings)

    if maximum_iteration is not None:
        option.maximum_iteration = maximum_iteration
    else:
        option.maximum_iteration = config.get_option_value(
            'maximum_iteration', target_type=int, section=settings)

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
    settings = config.get_option_value('settings', target_type=str)
    dataset_idp = config.get_option_value(
        'dataset_idp', target_type=str, section=settings)
    logger.vinfo('dataset_idp', dataset_idp)

    mesh_textured_max_iter_x_ofp = compute_ofp(config)

    rgbd_images, camera_trajectory, mesh = import_reconstruction(settings)

    # TODO HANDLE DOWNSCALED IMAGES
    # for parameters in camera_trajectory.parameters:
    #     print(parameters.intrinsic)
    #     print(parameters.intrinsic.intrinsic_matrix)

    visualize_intermediate_result(config, rgbd_images, camera_trajectory, mesh)

    color_map_optimization(
        mesh,
        rgbd_images,
        camera_trajectory,
        ofp=mesh_textured_max_iter_x_ofp,
        config=config,
        settings=settings)

