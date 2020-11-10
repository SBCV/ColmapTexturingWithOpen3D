import copy
from Utility.Types.Intrinsics import Intrinsics
from cto.utility.logging_extension import logger


def build_affine_transformation_matrix(camera_parameters, render_compatible_camera_parameters):
    # http://www.open3d.org/docs/release/python_api/open3d.camera.PinholeCameraIntrinsic.html

    intrinsic_original = camera_parameters.intrinsic.intrinsic_matrix
    intrinsic_renderer = render_compatible_camera_parameters.intrinsic.intrinsic_matrix

    f_x, f_y, skew, p_x, p_y = Intrinsics.split_intrinsic_mat(
        intrinsic_original)
    f_x_r, f_y_r, skew_r, p_x_r, p_y_r = Intrinsics.split_intrinsic_mat(
        render_compatible_camera_parameters.intrinsic.intrinsic_matrix)
    logger.vinfo('f_x, f_y, skew, p_x, p_y', [f_x, f_y, skew, p_x, p_y])
    assert f_x_r == f_y_r

    trans_mat_renderer_to_original = Intrinsics.compute_intrinsic_transformation(
        intrinsic_original, intrinsic_renderer, check_result=True)

    # Get the first two rows
    affine_mat = trans_mat_renderer_to_original[0:2, :]


    # # Build an affine matrix to compensate incorrect settings of renderer
    # offset_x = p_x - f_x / f_renderer * c_x_renderer
    # offset_y = p_y - f_y / f_renderer * c_y_renderer
    # affine_mat = np.asarray([
    #     [f_x / f_renderer, 0, offset_x],
    #     [0, f_y / f_renderer, offset_y]],
    #     dtype=np.float32)

    return affine_mat


def build_render_compatible_focal_length(intrinsics):
    return min(intrinsics.get_focal_length())


def build_render_test_camera_parameters(camera_parameters, width, height):
    render_compatible_camera_parameters = copy.deepcopy(camera_parameters)
    focal_length = (width + height) / 0.5
    render_compatible_camera_parameters.intrinsic.set_intrinsics(
        width, height, 1000, 1000,  width / 2.0 - 0.5, height / 2.0 - 0.5)
    return render_compatible_camera_parameters