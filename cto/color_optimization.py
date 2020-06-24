import open3d as o3d
from cto.config_api import get_reconstruction_mode


def color_map_optimization(mesh,
                           rgbd_images,
                           camera_trajectory,
                           ofp,
                           config,
                           maximum_allowable_depth=None,
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

    # If the "maximum_allowable_depth" value is too small,
    # the "[ColorMapOptimization] :: VisibilityCheck" may generate
    # > [Open3D DEBUG] [cam 0] 0.0 percents are visible
    # https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html
    #   An uint16 can store up to 65535 values
    option.maximum_allowable_depth = 65535.0
    print('option.maximum_allowable_depth', option.maximum_allowable_depth)

    # DON'T DO THIS:  option.depth_threshold_for_visibility_check = 0

    # option.maximum_allowable_depth = config.get_option_value(
    #     'maximum_allowable_depth', target_type=float, section=reconstruction_mode)

    if maximum_iteration is not None:
        option.maximum_iteration = maximum_iteration
    else:
        option.maximum_iteration = config.get_option_value(
            'maximum_iteration', target_type=int, section=reconstruction_mode)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.color_map.color_map_optimization(mesh, rgbd_images, camera_trajectory, option)
    o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh(ofp, mesh)

