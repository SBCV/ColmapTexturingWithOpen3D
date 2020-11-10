import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from cto.utility.os_extension import mkdir_safely
from cto.utility.logging_extension import logger

from cto.ext.colmap.read_dense import read_array as read_colmap_array
from cto.ext.colmap.read_write_dense import write_array as write_colmap_array
from cto.config_api import get_depth_map_source
from cto.config_api import get_use_original_depth_maps_as_mask
from cto.config_api import get_depth_map_rendering_library
from cto.config_api import get_show_color_rendering_flag
from cto.config_api import get_show_depth_rendering_flag
from cto.config_api import get_show_rendering_overview_flag
from cto.rendering.o3d_rendering_utility import compute_depth_maps_from_geometry as compute_o3d_depth_maps_from_geometry
from cto.rendering.vtk_rendering_utility import compute_depth_maps_from_geometry as compute_vtk_depth_maps_from_geometry


class ColmapDepthMapHandler:

    DEPTH_MAP_SOURCE_ORIGINAL = 'original'
    DEPTH_MAP_SOURCE_FROM_MESH = 'from_mesh'
    DEPTH_MAP_SOURCE_FROM_FUSED_POINT_CLOUD = 'from_fused_point_cloud'

    def __init__(self, colmap_workspace, config):
        self.depth_map_source = get_depth_map_source(config).lower()
        assert self.depth_map_source in [
            ColmapDepthMapHandler.DEPTH_MAP_SOURCE_ORIGINAL,
            ColmapDepthMapHandler.DEPTH_MAP_SOURCE_FROM_MESH,
            ColmapDepthMapHandler.DEPTH_MAP_SOURCE_FROM_FUSED_POINT_CLOUD]
        self.use_original_depth_maps_as_mask = get_use_original_depth_maps_as_mask(config)
        self.depth_map_rendering_library = get_depth_map_rendering_library(config).lower()
        self.depth_map_idp = colmap_workspace.depth_map_idp
        self.depth_map_suffix = colmap_workspace.depth_map_suffix
        self.depth_map_from_geometry_dp = colmap_workspace.depth_map_from_geometry_dp
        self.depth_map_from_geometry_suffix = colmap_workspace.depth_map_from_geometry_suffix
        self.color_image_idp = colmap_workspace.color_image_idp
        self.mesh_ifp = colmap_workspace.mesh_ifp
        self.config = config

    @staticmethod
    def read_depth_map(depth_map_ifp):
        ext = os.path.splitext(depth_map_ifp)[1]
        if ext == '.bin':
            depth_map = read_colmap_array(depth_map_ifp)
        # elif ext == '.npy':
        #     depth_map = np.load(depth_map_ifp)
        else:
            assert False
        return depth_map

    @staticmethod
    def write_depth_map(depth_map_ofp, depth_map):
        write_colmap_array(depth_map, depth_map_ofp)
        #np.save(depth_map_ofp, depth_map)

    @staticmethod
    def write_depth_map_visualization(depth_map_visualization_ofp, depth_map):
        plt.imsave(depth_map_visualization_ofp, np.asarray(depth_map), dpi=1)

    @staticmethod
    def visualize_depth_map(depth_map, show=True):
        plt.imshow(np.asarray(depth_map))
        # if show:
        #     plt.show()

    def is_depth_map_source_from_geometry(self):
        return self.depth_map_source in [
            ColmapDepthMapHandler.DEPTH_MAP_SOURCE_FROM_MESH,
            ColmapDepthMapHandler.DEPTH_MAP_SOURCE_FROM_FUSED_POINT_CLOUD]

    def get_depth_array_fp_s(self, ordered_image_names):

        depth_array_ifp_list = []
        if self.is_depth_map_source_from_geometry():
            suffix = self.depth_map_from_geometry_suffix
            depth_map_dp = self.depth_map_from_geometry_dp
        elif self.depth_map_source == ColmapDepthMapHandler.DEPTH_MAP_SOURCE_ORIGINAL:
            suffix = self.depth_map_suffix
            depth_map_dp = self.depth_map_idp
        else:
            assert False

        for image_name in ordered_image_names:
            depth_array_ifp_list.append(
                os.path.join(depth_map_dp, image_name + suffix))
        return depth_array_ifp_list

    def process_depth_maps(self,
                           camera_trajectory,
                           ordered_image_names):

        if self.depth_map_source in ['from_mesh', 'from_fused_point_cloud']:
            mkdir_safely(self.depth_map_from_geometry_dp)
            if self.depth_map_rendering_library == 'vtk':
                compute_vtk_depth_maps_from_geometry(
                    self.mesh_ifp,
                    camera_trajectory,
                    ordered_image_names,
                    depth_map_callback=self.write_depth_map_to_disk,
                    config=self.config)
            elif self.depth_map_rendering_library == 'o3d':
                compute_o3d_depth_maps_from_geometry(
                    self.mesh_ifp,
                    camera_trajectory,
                    ordered_image_names,
                    depth_map_callback=self.write_depth_map_to_disk,
                    config=self.config)
            else:
                assert False

    def compute_depth_statistics(self, depth_array_ifp_list):
        # Determine value range
        overall_depth_map_min = float('inf')
        overall_depth_map_max = -float('inf')
        for depth_map_ifp in depth_array_ifp_list:
            depth_arr = self.read_depth_map(depth_map_ifp)

            depth_map_min, depth_map_max = ColmapDepthMapHandler.compute_depth_map_min_max(depth_arr)

            overall_depth_map_min = min(overall_depth_map_min, depth_map_min)
            overall_depth_map_max = max(overall_depth_map_max, depth_map_max)
        depth_map_range = (overall_depth_map_min, overall_depth_map_max)
        logger.vinfo(
            'Depth Value Range (colmap):',
            ["min:", overall_depth_map_min, 'max:', overall_depth_map_max])
        return depth_map_range

    def write_depth_map_to_disk(self, image_name, depth_map, color_image):

        color_image_original_ifp = os.path.join(
            self.color_image_idp, image_name)
        depth_map_original_ifp = os.path.join(
            self.depth_map_idp, image_name + self.depth_map_suffix)
        depth_map_from_mesh_ofp = os.path.join(
            self.depth_map_from_geometry_dp,
            image_name + self.depth_map_from_geometry_suffix)

        # logger.vinfo('color_image_original_ifp', color_image_original_ifp)
        # logger.vinfo('depth_map_original_ifp', depth_map_original_ifp)
        logger.vinfo('depth_map_from_mesh_ofp', depth_map_from_mesh_ofp)

        if self.use_original_depth_maps_as_mask:

            logger.vinfo('Use original depth map as mask: ', self.use_original_depth_maps_as_mask)
            depth_map_original = self.read_depth_map(depth_map_original_ifp)
            if depth_map_original.shape != depth_map.shape:
                logger.vinfo('depth_map_original.shape', depth_map_original.shape)
                logger.vinfo('depth_map_from_mesh.shape', depth_map.shape)
                assert False
            depth_map_from_mesh_masked = copy.deepcopy(depth_map)
            depth_map_from_mesh_masked[depth_map_original == 0.0] = 0
            depth_map = depth_map_from_mesh_masked
        else:
            depth_map_from_mesh_masked = None

        ColmapDepthMapHandler.write_depth_map(depth_map_from_mesh_ofp, depth_map)

        # Some visualization functions for debugging purposes
        if get_show_color_rendering_flag(self.config):
            plt.imshow(color_image)
            plt.show()

        if get_show_depth_rendering_flag(self.config):
            plt.imshow(depth_map)
            plt.show()

        if get_show_rendering_overview_flag(self.config):
            self.show_color_and_depth_renderings(
                image_name,
                color_image_original_ifp,
                depth_map_original_ifp,
                color_image,
                depth_map,
                depth_map_from_mesh_masked)

        # if depth_viz_odp is not None:
        #     depth_map_from_mesh_viz_ofp = os.path.join(depth_viz_odp, image_name)
        #
        #     ColmapDepthMapHandler.write_depth_map_visualization(depth_map_from_mesh_viz_ofp, depth_map)
        #
        #     plt.imsave(depth_map_from_mesh_viz_ofp, np.asarray(depth_map), dpi=1)
        #     self.visualize_depth_map(depth_map_from_mesh_viz_ofp)

    def show_color_and_depth_renderings(self,
                                        image_name,
                                        color_image_original_ifp,
                                        depth_map_original_ifp,
                                        color_image_from_mesh,
                                        depth_map_from_mesh,
                                        depth_map_from_mesh_masked):

        logger.info('show_color_and_depth_renderings: ...')
        # if not image_name == '100_7100_resized.JPG':
        #     return

        main_fig, ax_arr_2_by_5 = plt.subplots(
            nrows=2, ncols=5, gridspec_kw={'height_ratios': (1, 1)}
            # gridspec_kw=dict(
            #     # https://stackoverflow.com/questions/34921930/in-a-matplotlib-plot-consisting-of-histogram-subplots-how-can-the-height-and-ba
            #     height_ratios=(1, 1),
            #     wspace=0.1,
            #     hspace=0.1,
            #     top=0.9,
            #     bottom=0,
            #     left=0.1,
            #     right=0.9)
        )

        depth_map_from_mesh_nan = copy.deepcopy(depth_map_from_mesh)
        depth_map_from_mesh_nan[depth_map_from_mesh_nan == 0] = np.nan

        main_fig.suptitle(image_name)

        width_in_inches = 1080 / main_fig.dpi
        height_in_inches = 1080 / main_fig.dpi
        main_fig.set_size_inches((width_in_inches, height_in_inches))

        depth_map_original = self.read_depth_map(depth_map_original_ifp)
        depth_map_original_nan = copy.deepcopy(depth_map_original)
        depth_map_original_nan[depth_map_original_nan == 0] = np.nan

        min_original = np.nanmin(depth_map_original_nan)
        max_original = np.nanmax(depth_map_original_nan)

        min_mesh = np.nanmin(depth_map_from_mesh_nan)
        min_value = min(min_original, min_mesh)

        max_mesh = np.nanmax(depth_map_original_nan)
        max_value = max(max_original, max_mesh)

        color_image_original_ax = ax_arr_2_by_5[0, 0]
        color_image_original_ax.set_title('color_image_original')
        color_image_original_ax.imshow(mpimg.imread(color_image_original_ifp))

        color_image_original_ax = ax_arr_2_by_5[0, 1]
        color_image_original_ax.set_title('color_image_from_mesh')
        color_image_original_ax.imshow(color_image_from_mesh)

        depth_map_original_ax = ax_arr_2_by_5[0, 2]
        depth_map_original_ax.set_title('depth_map_original_nan')
        depth_map_original_ax.imshow(depth_map_original_nan, vmin=min_value, vmax=max_value)
        depth_map_original_ax.annotate(
            'min_original: ' + str(min_original), (0, 0), (0, -20),
            xycoords='axes fraction', textcoords='offset points', va='top')
        depth_map_original_ax.annotate(
            'max_original: ' + str(max_original), (0, 0), (0, -40),
            xycoords='axes fraction', textcoords='offset points', va='top')

        depth_map_from_mesh_ax = ax_arr_2_by_5[0, 3]
        depth_map_from_mesh_ax.set_title('depth_map_from_mesh_nan')
        depth_map_from_mesh_ax.imshow(depth_map_from_mesh_nan, vmin=min_value, vmax=max_value)
        depth_map_from_mesh_ax.annotate(
            'min_mesh: ' + str(min_mesh), (0, 0), (0, -20),
            xycoords='axes fraction', textcoords='offset points', va='top')
        depth_map_from_mesh_ax.annotate(
            'max_mesh: ' + str(max_mesh), (0, 0), (0, -40),
            xycoords='axes fraction', textcoords='offset points', va='top')

        if self.use_original_depth_maps_as_mask:
            depth_map_from_mesh_masked_nan = copy.deepcopy(depth_map_from_mesh_masked)
            depth_map_from_mesh_masked_nan[depth_map_from_mesh_masked == 0] = np.nan
            depth_map_from_mesh_masked_ax = ax_arr_2_by_5[0, 4]
            depth_map_from_mesh_masked_ax.set_title('depth_map_from_mesh_masked_nan')
            depth_map_from_mesh_masked_ax.imshow(depth_map_from_mesh_masked_nan, vmin=min_value, vmax=max_value)

        show_histogram = False
        if show_histogram:
            num_bins = 10
            # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html
            depth_map_original_hist_ax = ax_arr_2_by_5[1, 0]
            depth_map_original_hist_ax.hist(
                depth_map_original, range=(min_value, max_value))
            depth_map_from_mesh_hist_ax = ax_arr_2_by_5[1, 1]
            depth_map_from_mesh_hist_ax.hist(
                depth_map_from_mesh, range=(min_value, max_value))
            if self.use_original_depth_maps_as_mask:
                depth_map_from_mesh_masked_hist_ax = ax_arr_2_by_5[1, 2]
                depth_map_from_mesh_masked_hist_ax.hist(
                    depth_map_from_mesh_masked, range=(min_value, max_value))

        # main_fig.subplots_adjust(right=0.8)
        # cbar_ax = main_fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # main_fig.colorbar(im, cax=cbar_ax)

        plt.show()

        # ColmapDepthMapHandler.visualize_depth_map(depth_map_original, show=False)
        # ColmapDepthMapHandler.visualize_depth_map(depth_map_from_mesh, show=True)
        logger.info('show_color_and_depth_renderings: ...')



    @staticmethod
    def compute_depth_map_min_max(depth_map):
        depth_map_nan = copy.deepcopy(depth_map)
        depth_map_nan[depth_map_nan == 0] = np.nan
        return np.nanmin(depth_map_nan), np.nanmax(depth_map_nan)
