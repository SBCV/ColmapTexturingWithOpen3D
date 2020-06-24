import open3d as o3d

import os
import numpy as np
import matplotlib.pyplot as plt
from cto.utility.os_extension import mkdir_safely
from cto.utility.logging_extension import logger

#############################################################
# Depth rendering of Open3D is at the moment STRONGLY LIMITED
#   * f_x and f_y must be equal
#   * c_x must be equal to width / 2 - 0.5
#   * c_y must be equal to height / 2 - 0.5
#############################################################


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
    #
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
        width = intrinsics.width
        height = intrinsics.height
        f_x, f_y = intrinsics.get_focal_length()
        c_x, c_y = intrinsics.get_principal_point()

        if f_x != f_y:
            logger.vinfo('get_focal_length(self)', [f_x, f_y])
            assert False

        if c_x != (width / 2 - 0.5) or c_y != (height / 2 - 0.5):
            logger.vinfo('get_principal_point(self)', [c_x, c_y])
            assert False

        ##########################################################################
        # The following would remove the warning message, but obviously will cause incorrect results!
        # f_average = (f_x + f_y) / 2
        # c_x = width / 2 - 0.5
        # c_y = height / 2 - 0.5
        # intrinsics.set_intrinsics(width, height, f_average, f_average, c_x, c_y)
        ##########################################################################

        vis.create_window(
            width=width, height=height, left=0, top=0, visible=show_rendering)
        vis.add_geometry(mesh)

        view_control = vis.get_view_control()
        view_control.convert_from_pinhole_camera_parameters(
            camera_parameters)

        # http://www.open3d.org/docs/release/tutorial/Advanced/non_blocking_visualization.html
        # vis.update_geometry(pcd)
        vis.poll_events()             # CRUCIAL
        vis.update_renderer()

        depth = np.asarray(vis.capture_depth_float_buffer(do_render=False), dtype=np.float32)
        np.save(depth_ofp, depth)

        if depth_viz_odp is not None:
            depth_viz_ofp = os.path.join(depth_viz_odp, image_name)
            plt.imsave(depth_viz_ofp, np.asarray(depth), dpi=1)

    logger.info('create_depth_maps_from_mesh: Done ')
