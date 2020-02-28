# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/color_map_optimization.py

import open3d as o3d

import matplotlib.pyplot as plt
import numpy as np

import os
import sys

from ctm.data_parsing import parse_mesh
from ctm.data_parsing import parse_colmap_camera_trajectory
from ctm.data_parsing import parse_colmap_rgb_and_depth_data
from ctm.data_parsing import parse_o3d_trajectory
from ctm.data_parsing import parse_o3d_data
from ctm.workspace import ColmapWorkspace
from ctm.workspace import O3DWorkspace
from ctm.visualization import visualize_rgbd_image_list


def color_map_optimization(mesh,
                           rgbd_images,
                           camera_trajectory,
                           ofp,
                           maximum_allowable_depth,
                           non_rigid_camera_coordinate=True,
                           max_iter=300):
    # Optimize texture and save the mesh as texture_mapped.ply
    # This is implementation of following paper: "Q.-Y. Zhou and V. Koltun,
    # Color Map Optimization for 3D Reconstruction with Consumer Depth Cameras, SIGGRAPH 2014"
    option = o3d.color_map.ColorMapOptimizationOption()
    option.non_rigid_camera_coordinate = non_rigid_camera_coordinate
    option.number_of_vertical_anchors = 16
    option.non_rigid_anchor_point_weight = 0.316
    option.maximum_iteration = max_iter

    # If the "maximum_allowable_depth" value is too small,
    # the "[ColorMapOptimization] :: VisibilityCheck" may generate
    # > [Open3D DEBUG] [cam 0] 0.0 percents are visible
    option.maximum_allowable_depth = maximum_allowable_depth           # Default Value 2.5

    option.depth_threshold_for_visiblity_check = 0.03
    option.depth_threshold_for_discontinuity_check = 0.1
    option.half_dilation_kernel_size_for_discontinuity_map = 3
    option.image_boundary_margin = 10
    option.invisible_vertex_color_knn = 3

    o3d.color_map.color_map_optimization(mesh, rgbd_images, camera_trajectory, option)
    o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh(ofp, mesh)


if __name__ == "__main__":

    # http://www.open3d.org/docs/release/tutorial/Advanced/color_map_optimization.html

    o3d.utility.set_verbosity_level(
        o3d.utility.VerbosityLevel.Debug)

    use_colmap = True
    if use_colmap:
        dataset_idp = 'Desktop/TextureMapping/sceaux_current/dense_workspace'
        #dataset_idp = 'Desktop/TextureMapping/sceaux_current_non_scaled/workspace'
        colmap_workspace = ColmapWorkspace(
            dataset_idp, use_geometric_depth_maps=True, use_poisson=True)
        camera_trajectory, ordered_image_names = parse_colmap_camera_trajectory(colmap_workspace)
        rgbd_images = parse_colmap_rgb_and_depth_data(
            ordered_image_names, colmap_workspace, lazy=True)
        mesh = parse_mesh(colmap_workspace)

    else:
        dataset_idp = 'Desktop/fountain/fountain_small'
        o3d_workspace = O3DWorkspace(dataset_idp)
        camera_trajectory = parse_o3d_trajectory(o3d_workspace)
        rgbd_images = parse_o3d_data(o3d_workspace)
        mesh = parse_mesh(o3d_workspace)

    mesh_textured_before_opt_ofp = os.path.join(
        dataset_idp, "color_map_initial.ply")
    mesh_textured_after_opt_ofp = os.path.join(
        dataset_idp, "color_map_optimized.ply")


    # TODO HANDLE DOWNSCALED IMAGES
    # for parameters in camera_trajectory.parameters:
    #     print(parameters.intrinsic)
    #     print(parameters.intrinsic.intrinsic_matrix)


    #Visualize intermediate results
    #additional_point_cloud_list = [mesh]
    additional_point_cloud_list = []
    visualize_rgbd_image_list(
        rgbd_images,
        camera_trajectory,
        additional_point_cloud_list=additional_point_cloud_list)

    maximum_allowable_depth = 15    # TODO adjust this value

    # Before full optimization, let's just visualize texture map
    # with given geometry, RGBD images, and camera poses.
    color_map_optimization(
        mesh,
        rgbd_images,
        camera_trajectory,
        ofp=mesh_textured_before_opt_ofp,
        maximum_allowable_depth=maximum_allowable_depth,
        max_iter=10,
        non_rigid_camera_coordinate=False
    )

    # color_map_optimization(
    #     mesh,
    #     rgbd_images,
    #     camera_trajectory,
    #     ofp=mesh_textured_after_opt_ofp,
    #     max_iter=300,
    #     non_rigid_camera_coordinate=True
    # )