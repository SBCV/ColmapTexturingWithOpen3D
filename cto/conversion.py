import os
from PIL import Image
import open3d as o3d
import numpy as np


def scale_camera(f_x, f_y, c_x, c_y, width_original, height_original, width_resized, height_resized):
    # https://github.com/colmap/colmap/blob/dev/src/base/undistortion.cc
    #   Camera UndistortCamera(const UndistortCameraOptions& options,
    #                          const Camera& camera) {
    #       ...
    #       undistorted_camera.Rescale(max_image_scale);
    # https://github.com/colmap/colmap/blob/dev/src/base/camera.cc
    #   void Camera::Rescale(const double scale) {
    #       ...
    #   void Camera::Rescale(const size_t width, const size_t height) {
    #       ...

    scale_x = width_resized / width_original
    scale_y = height_resized / height_original
    scale = min(scale_x, scale_y)
    if scale < 1.0:
        # print('scale_x', scale_x)
        # print('scale_y', scale_y)
        # print('width_resized', width_resized)
        # print('height_resized', height_resized)
        c_x = scale_x * c_x
        c_y = scale_y * c_y
        f_x = scale_x * f_x
        f_y = scale_y * f_y
        width = width_resized
        height = height_resized
    else:
        width = width_original
        height = height_original

    return c_x, c_y, f_x, f_y, width, height


def convert_colmap_to_o3d_camera_trajectory(colmap_camera_parameter_dict,
                                            colmap_image_parameter_dict,
                                            colmap_workspace):

    # http://www.open3d.org/docs/release/python_api/open3d.camera.PinholeCameraTrajectory.html
    camera_trajectory = o3d.camera.PinholeCameraTrajectory()
    camera_trajectory_parameters = []
    ordered_image_names = []

    for image_id, image_params in colmap_image_parameter_dict.items():
        camera = colmap_camera_parameter_dict[image_params.camera_id]

        # http://www.open3d.org/docs/release/python_api/open3d.camera.PinholeCameraParameters.html
        # http://www.open3d.org/docs/release/python_api/open3d.camera.PinholeCameraIntrinsic.html
        pinhole_camera_parameters = o3d.camera.PinholeCameraParameters()
        pinhole_camera_parameters.intrinsic = convert_colmap_to_o3d_intrinsics(
            camera, image_params, colmap_workspace)
        pinhole_camera_parameters.extrinsic = convert_colmap_to_o3d_extrinsics(image_params)

        camera_trajectory_parameters.append(pinhole_camera_parameters)
        ordered_image_names.append(image_params.name)

    camera_trajectory.parameters = camera_trajectory_parameters

    return camera_trajectory, ordered_image_names


def convert_colmap_to_o3d_extrinsics(image_params):
    rot_mat = image_params.qvec2rotmat()
    trans_vec_mat = np.array([image_params.tvec])

    upper_extrinsic_mat = np.concatenate((rot_mat, trans_vec_mat.T), axis=1)
    lower_extrinsic_mat = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=float)
    return np.concatenate((upper_extrinsic_mat, lower_extrinsic_mat), axis=0)


def convert_colmap_to_o3d_intrinsics(colmap_camera, image_params, colmap_workspace):

    # http://www.open3d.org/docs/release/python_api/open3d.camera.PinholeCameraIntrinsic.html#open3d.camera.PinholeCameraIntrinsic

    o3d_camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    width_original = colmap_camera.width
    height_original = colmap_camera.height

    image_resized_fp = os.path.join(colmap_workspace.color_image_resized_dp, image_params.name)
    image_resized = Image.open(image_resized_fp)
    width_resized, height_resized = image_resized.size

    params = colmap_camera.params
    if colmap_camera.model == 'PINHOLE':
        f_x = params[0]
        f_y = params[1]
        c_x = params[2]
        c_y = params[3]
        skew = 0

        c_x, c_y, f_x, f_y, width, height = scale_camera(
            f_x, f_y,
            c_x, c_y,
            width_original, height_original,
            width_resized, height_resized)

        o3d_camera_intrinsics.width = width_resized
        o3d_camera_intrinsics.height = height_resized
        o3d_camera_intrinsics.intrinsic_matrix = np.array(
            [[f_x, skew, c_x],
             [0, f_y, c_y],
             [0, 0, 1]],
            dtype=float)
        print('intrinsic_matrix')
        print(o3d_camera_intrinsics.intrinsic_matrix)
    else:
        assert False

    return o3d_camera_intrinsics

