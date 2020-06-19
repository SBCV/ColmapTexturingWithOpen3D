import open3d as o3d
import numpy as np


def convert_colmap_to_o3d_camera_trajectory(colmap_camera_parameter_dict, colmap_image_parameter_dict):

    # http://www.open3d.org/docs/release/python_api/open3d.camera.PinholeCameraTrajectory.html
    camera_trajectory = o3d.camera.PinholeCameraTrajectory()
    camera_trajectory_parameters = []
    ordered_image_names = []

    for image_id, image_params in colmap_image_parameter_dict.items():
        camera = colmap_camera_parameter_dict[image_params.camera_id]

        # http://www.open3d.org/docs/release/python_api/open3d.camera.PinholeCameraParameters.html
        # http://www.open3d.org/docs/release/python_api/open3d.camera.PinholeCameraIntrinsic.html
        pinhole_camera_parameters = o3d.camera.PinholeCameraParameters()
        pinhole_camera_parameters.intrinsic = convert_colmap_to_o3d_intrinsics(camera)
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


def convert_colmap_to_o3d_intrinsics(colmap_camera):

    # http://www.open3d.org/docs/release/python_api/open3d.camera.PinholeCameraIntrinsic.html#open3d.camera.PinholeCameraIntrinsic

    o3d_camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    o3d_camera_intrinsics.width = colmap_camera.width
    o3d_camera_intrinsics.height = colmap_camera.height

    params = colmap_camera.params
    if colmap_camera.model == 'PINHOLE':
        f_x = params[0]
        f_y = params[1]
        c_x = params[2]
        c_y = params[3]
        skew = 0
        o3d_camera_intrinsics.intrinsic_matrix = np.array(
            [[f_x, skew, c_x],
             [0, f_y, c_y],
             [0, 0, 1]],
            dtype=float)
    else:
        assert False
    return o3d_camera_intrinsics

