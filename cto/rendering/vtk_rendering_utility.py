import sys
import os
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt

from cto.utility.logging_extension import logger

from VTKInterface.Interfaces.Render_Interface import RenderInterface
from cto.rendering.rendering_utility import build_render_compatible_focal_length
from cto.rendering.rendering_utility import build_affine_transformation_matrix


def build_vtk_render_compatible_intrinsics(intrinsics):
    # The renderer of Open3D has the following (strong) restrictions
    #   * f_x and f_y must be equal
    #   * c_x must be equal to width / 2 - 0.5
    #   * c_y must be equal to height / 2 - 0.5

    width = intrinsics.width
    height = intrinsics.height
    cx_renderer, cy_renderer = intrinsics.get_principal_point()
    f_renderer = build_render_compatible_focal_length(intrinsics)
    return width, height, f_renderer, f_renderer, cx_renderer, cy_renderer


def build_vtk_render_compatible_camera_parameters(camera_parameters):

    width, height, f_renderer, f_renderer, cx_renderer, cy_renderer = build_vtk_render_compatible_intrinsics(
        camera_parameters.intrinsic)
    render_compatible_camera_parameters = copy.deepcopy(camera_parameters)
    render_compatible_camera_parameters.intrinsic.set_intrinsics(
        width, height, f_renderer, f_renderer, cx_renderer, cy_renderer)
    return render_compatible_camera_parameters


def invert_transformation_mat(trans_mat):
    # Exploit that the inverse of the rotation part is equal to the transposed of the rotation part
    # This should be more robust than trans_mat_inv = np.linalg.inv(trans_mat)
    trans_mat_inv = np.zeros_like(trans_mat)
    rotation_part_inv = trans_mat[0:3, 0:3].T
    trans_mat_inv[0:3, 0:3] = rotation_part_inv
    trans_mat_inv[0:3, 3] = - np.dot(rotation_part_inv, trans_mat[0:3, 3])
    trans_mat_inv[3, 3] = 1
    return trans_mat_inv


def compute_depth_maps_from_geometry(mesh_ifp,
                                     camera_trajectory,
                                     ordered_image_names,
                                     depth_map_callback,
                                     config=None):

    logger.info('create_depth_maps_from_mesh: ... ')

    num_params = len(camera_trajectory.parameters)
    logger.vinfo('num_params', num_params)

    camera_parameter_list = camera_trajectory.parameters
    num_images = None
    if num_images is not None:
        camera_parameter_list = camera_parameter_list[:num_images]

    assert os.path.isfile(mesh_ifp)
    # required for certain methods called below
    off_screen_rendering = True
    for image_name, camera_parameters in zip(ordered_image_names, camera_parameter_list):

        extrinsics = camera_parameters.extrinsic
        cam_to_world_mat_computer_vision = invert_transformation_mat(extrinsics)

        # http://www.open3d.org/docs/release/python_api/open3d.camera.PinholeCameraIntrinsic.html
        intrinsics = camera_parameters.intrinsic
        render_compatible_camera_parameters = build_vtk_render_compatible_camera_parameters(
            camera_parameters
        )

        width = intrinsics.width
        height = intrinsics.height

        render_interface = RenderInterface(
            off_screen_rendering=off_screen_rendering,
            width=width,
            height=height,
            background_color=(0, 127, 127))

        # Can we avoid this redundant loading
        render_interface.load_vtk_mesh_or_point_cloud(
            mesh_ifp, texture_ifp=None)

        render_interface.set_active_cam_from_computer_vision_cam_to_world_mat(
            cam_to_world_mat_computer_vision,
            render_compatible_camera_parameters.intrinsic.intrinsic_matrix,
            width,
            height,
            max_clipping_range=sys.float_info.max)

        render_interface.render()
        #render_interface.show_z_buffer()
        if not off_screen_rendering:
            render_interface.render_and_start()

        # We apply an affine transformation to the depth_map images
        # to compensate differences in the intrinsic parameters
        affine_mat = build_affine_transformation_matrix(
            camera_parameters, render_compatible_camera_parameters)

        depth_map = render_interface.get_computer_vision_depth_buffer_as_numpy_arr()
        color_image = render_interface.get_rgba_buffer_as_numpy_arr()

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

    # if not off_screen_rendering:
    #     render_interface.render_and_start()
    logger.info('create_depth_maps_from_mesh: Done ')

    ######################################################################33










