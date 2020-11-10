import sys
import open3d as o3d
from cto.config_api import get_reconstruction_mode
from cto.utility.logging_extension import logger

def color_map_optimization(mesh,
                           rgbd_images,
                           camera_trajectory,
                           ofp,
                           config,
                           depth_range=None,
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

    option = o3d.pipelines.color_map.ColorMapOptimizationOption()
    option.non_rigid_camera_coordinate = config.get_option_value(
        'non_rigid_camera_coordinate', target_type=bool, section=reconstruction_mode)

    # This maximum_allowable_depth value is defined w.r.t. to the mesh
    # Therefore the original depth range values must be used
    # (and not the scaled depth maps represented as uint16)
    # One can observe this behavior by providing the depth_arr_min value for a specific image
    # and analysing the corresponding open3d debug output, i.e.
    # [Open3D DEBUG] [cam 0]: 0/951198 (0.00000%) vertices are visible

    #option.maximum_allowable_depth = depth_range[1]
    option.maximum_allowable_depth = sys.float_info.max
    logger.vinfo('depth_range', depth_range)
    logger.vinfo('option.maximum_allowable_depth', option.maximum_allowable_depth)

    # DON'T DO THIS:  option.depth_threshold_for_visibility_check = 0

    # option.maximum_allowable_depth = config.get_option_value(
    #     'maximum_allowable_depth', target_type=float, section=reconstruction_mode)

    if maximum_iteration is not None:
        option.maximum_iteration = maximum_iteration
    else:
        option.maximum_iteration = config.get_option_value(
            'maximum_iteration', target_type=int, section=reconstruction_mode)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.color_map.color_map_optimization(mesh, rgbd_images, camera_trajectory, option)
    o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh(ofp, mesh)

