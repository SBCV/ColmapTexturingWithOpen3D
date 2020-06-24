import open3d as o3d
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from cto.utility.os_extension import mkdir_safely
from cto.utility.logging_extension import logger
import copy

#############################################################
# Depth rendering of Open3D is STRONGLY LIMITED at the moment
#   * f_x and f_y must be equal
#   * c_x must be equal to width / 2 - 0.5
#   * c_y must be equal to height / 2 - 0.5
# Use the workaround provided in the following repository
#   https://github.com/samarth-robo/open3d_colormap_opt_z_buffering/blob/master/demo.py
#############################################################


def build_affine_matrix(camera_parameters, render_compatible_camera_parameters):
    fx, fy = camera_parameters.intrinsic.get_focal_length()
    cx, cy = camera_parameters.intrinsic.get_principal_point()
    f_renderer, f_renderer = render_compatible_camera_parameters.intrinsic.get_focal_length()
    cx_renderer, cy_renderer = render_compatible_camera_parameters.intrinsic.get_principal_point()
    # Build an affine matrix to compensate incorrect settings of renderer
    offset_x = cx - fx / f_renderer * cx_renderer
    offset_y = cy - fy / f_renderer * cy_renderer
    affine_mat = np.asarray([
        [fx / f_renderer, 0, offset_x],
        [0, fy / f_renderer, offset_y]],
        dtype=np.float32)
    return affine_mat


def build_render_compatible_camera_parameters(camera_parameters):
    # The renderer of Open3D has the following (strong) restrictions
    #   * f_x and f_y must be equal
    #   * c_x must be equal to width / 2 - 0.5
    #   * c_y must be equal to height / 2 - 0.5
    intrinsics = camera_parameters.intrinsic
    width = intrinsics.width
    height = intrinsics.height
    cx_renderer = width / 2.0 - 0.5
    cy_renderer = height / 2.0 - 0.5
    f_renderer = max(intrinsics.get_focal_length())

    render_compatible_camera_parameters = copy.deepcopy(camera_parameters)
    render_compatible_camera_parameters.intrinsic.set_intrinsics(
        width, height, f_renderer, f_renderer, cx_renderer, cy_renderer)
    return render_compatible_camera_parameters

def create_depth_maps_from_mesh(mesh,
                                camera_trajectory,
                                ordered_image_names,
                                depth_odp,
                                depth_from_mesh_suffix,
                                depth_viz_odp=None,
                                show_rendering=False,
                                num_images=None):
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

    mkdir_safely(depth_odp)
    if depth_viz_odp is not None:
        mkdir_safely(depth_odp)

    num_params = len(camera_trajectory.parameters)
    logger.vinfo('num_params', num_params)

    camera_parameter_list = camera_trajectory.parameters
    if num_images is not None:
        camera_parameter_list = camera_parameter_list[:num_images]

    # http://www.open3d.org/docs/release/python_api/open3d.visualization.html
    # http://www.open3d.org/docs/release/python_api/open3d.visualization.Visualizer.html
    vis = o3d.visualization.Visualizer()

    for image_name, camera_parameters in zip(ordered_image_names, camera_parameter_list):

        depth_ofp = os.path.join(depth_odp, image_name + depth_from_mesh_suffix)

        # http://www.open3d.org/docs/release/python_api/open3d.camera.PinholeCameraIntrinsic.html
        intrinsics = camera_parameters.intrinsic
        vis.create_window(
            width=intrinsics.width, height=intrinsics.height, left=0, top=0, visible=show_rendering)
        vis.add_geometry(mesh)

        view_control = vis.get_view_control()
        render_compatible_camera_parameters = build_render_compatible_camera_parameters(camera_parameters)
        view_control.convert_from_pinhole_camera_parameters(
            render_compatible_camera_parameters)

        # http://www.open3d.org/docs/release/tutorial/Advanced/non_blocking_visualization.html
        # vis.update_geometry(pcd)
        vis.poll_events()             # CRUCIAL
        vis.update_renderer()

        depth = np.asarray(vis.capture_depth_float_buffer(do_render=False), dtype=np.float32)

        # We apply an affine transformation to the depth images
        # to compensate differences in the intrinsic parameters
        affine_mat = build_affine_matrix(camera_parameters, render_compatible_camera_parameters)
        depth = cv2.warpAffine(
            depth,
            affine_mat,
            (depth.shape[1], depth.shape[0]),
            cv2.WARP_INVERSE_MAP,
            cv2.BORDER_CONSTANT, 0)

        np.save(depth_ofp, depth)

        if depth_viz_odp is not None:
            depth_viz_ofp = os.path.join(depth_viz_odp, image_name)
            plt.imsave(depth_viz_ofp, np.asarray(depth), dpi=1)

    logger.info('create_depth_maps_from_mesh: Done ')
