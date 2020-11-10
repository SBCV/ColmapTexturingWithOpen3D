import open3d as o3d
import cv2
import numpy as np
import copy

from cto.utility.logging_extension import logger
from cto.rendering.rendering_utility import build_render_compatible_focal_length
from cto.rendering.rendering_utility import build_affine_transformation_matrix

#############################################################
# Depth rendering of Open3D is STRONGLY LIMITED at the moment
#   * f_x and f_y must be equal
#   * c_x must be equal to width / 2 - 0.5
#   * c_y must be equal to height / 2 - 0.5
# Use the workaround provided in the following repository
#   https://github.com/samarth-robo/open3d_colormap_opt_z_buffering/blob/master/demo.py
#############################################################


def build_o3d_render_compatible_intrinsics(intrinsics):
    # The renderer of Open3D has the following (strong) restrictions
    #   * f_x and f_y must be equal
    #   * c_x must be equal to width / 2 - 0.5
    #   * c_y must be equal to height / 2 - 0.5

    width = intrinsics.width
    height = intrinsics.height
    cx_renderer = width / 2.0 - 0.5
    cy_renderer = height / 2.0 - 0.5
    f_renderer = build_render_compatible_focal_length(intrinsics)
    return width, height, f_renderer, f_renderer, cx_renderer, cy_renderer


def build_o3d_render_compatible_camera_parameters(camera_parameters):

    width, height, f_renderer, f_renderer, cx_renderer, cy_renderer = build_o3d_render_compatible_intrinsics(
        camera_parameters.intrinsic)
    render_compatible_camera_parameters = copy.deepcopy(camera_parameters)
    render_compatible_camera_parameters.intrinsic.set_intrinsics(
        width, height, f_renderer, f_renderer, cx_renderer, cy_renderer)
    return render_compatible_camera_parameters


def compute_depth_maps_from_geometry(mesh_ifp,
                                     camera_trajectory,
                                     ordered_image_names,
                                     depth_map_callback,
                                     config):
    logger.info('create_depth_maps_from_mesh: ... ')

    # https://github.com/intel-isl/Open3D/blob/master/cpp/open3d/visualization/Visualizer/ViewControl.cpp#L189
    #   bool ViewControl::ConvertFromPinholeCameraParameters(
    #       ...
    #         window_height_ != intrinsic.height_ ||
    #         window_width_ != intrinsic.width_ ||
    #         intrinsic.intrinsic_matrix_(0, 2) !=
    #                 (double)window_width_ / 2.0 - 0.5 ||
    #         intrinsic.intrinsic_matrix_(1, 2) !=
    #                 (double)window_height_ / 2.0 - 0.5) {
    #         utility::LogWarning(
    #                 "[ViewControl] ConvertFromPinholeCameraParameters() failed "
    #                 "because window height and width do not match.");
    #   Therefore, only specific intrinsic matrices are allowed

    num_params = len(camera_trajectory.parameters)
    logger.vinfo('num_params', num_params)

    camera_parameter_list = camera_trajectory.parameters
    num_images = None
    if num_images is not None:
        camera_parameter_list = camera_parameter_list[:num_images]

    # http://www.open3d.org/docs/release/python_api/open3d.visualization.html
    # http://www.open3d.org/docs/release/python_api/open3d.visualization.Visualizer.html
    vis = o3d.visualization.Visualizer()
    show_rendering = False
    for image_name, camera_parameters in zip(ordered_image_names, camera_parameter_list):

        # http://www.open3d.org/docs/release/python_api/open3d.camera.PinholeCameraIntrinsic.html
        intrinsics = camera_parameters.intrinsic

        if show_rendering:
            if intrinsics.width > 1920 or intrinsics.height > 1080:
                # https://github.com/intel-isl/Open3D/issues/2036
                logger.warning(
                    'THERE IS A KNOWN ISSUE FOR VISUALIZING WINDOW SIZES GREATER THAN THE DEFAULT VALUES: ' +
                    '({}, {}) vs ({}, {})'.format(intrinsics.width, intrinsics.height, 1920, 1080))
                logger.warning('Setting show_rendering=False should avoid this problem ')

        vis.create_window(
            width=intrinsics.width,
            height=intrinsics.height,
            left=0,
            top=0,
            visible=show_rendering)

        mesh = o3d.io.read_triangle_mesh(mesh_ifp)
        vis.add_geometry(mesh)

        view_control = vis.get_view_control()
        render_compatible_camera_parameters = build_o3d_render_compatible_camera_parameters(
            camera_parameters)

        view_control.convert_from_pinhole_camera_parameters(
            render_compatible_camera_parameters)

        # http://www.open3d.org/docs/release/tutorial/Advanced/non_blocking_visualization.html
        # vis.update_geometry(pcd)
        vis.poll_events()             # CRUCIAL
        vis.update_renderer()

        # We apply an affine transformation to the depth_map images
        # to compensate differences in the intrinsic parameters
        affine_mat = build_affine_transformation_matrix(
            camera_parameters, render_compatible_camera_parameters)

        # http://www.open3d.org/docs/release/python_api/open3d.visualization.Visualizer.html
        color_image = np.asarray(
            vis.capture_screen_float_buffer(do_render=False),
            dtype=np.float32)

        depth_map = np.asarray(
            vis.capture_depth_float_buffer(do_render=False),
            dtype=np.float32)

        color_image = cv2.warpAffine(
            color_image,
            affine_mat,
            (color_image.shape[1], color_image.shape[0]),
            cv2.WARP_INVERSE_MAP,
            cv2.BORDER_CONSTANT, 0)

        depth_map = cv2.warpAffine(
            depth_map,
            affine_mat,
            (depth_map.shape[1], depth_map.shape[0]),
            cv2.WARP_INVERSE_MAP,
            cv2.BORDER_CONSTANT, 0)

        depth_map_callback(image_name, depth_map, color_image)

    logger.info('create_depth_maps_from_mesh: Done ')
