
import open3d as o3d

from cto.data_parsing.colmap_parsing import parse_colmap_camera_trajectory
from cto.data_parsing.colmap_parsing import compute_resized_images
from cto.data_parsing.colmap_parsing import parse_colmap_rgb_and_depth_data
from cto.data_parsing.o3d_parsing import parse_o3d_trajectory
from cto.data_parsing.o3d_parsing import parse_o3d_data

from cto.workspace import ColmapWorkspace
from cto.workspace import O3DWorkspace

from cto.config_api import get_dataset_idp
from cto.config_api import get_reconstruction_mode


def parse_mesh(workspace):
    return o3d.io.read_triangle_mesh(
         workspace.mesh_ifp)


def import_reconstruction(config):
    dataset_idp = get_dataset_idp(config)
    reconstruction_mode = get_reconstruction_mode(config)
    if reconstruction_mode.lower() == 'colmap':

        colmap_workspace = ColmapWorkspace(
            dataset_idp, use_geometric_depth_maps=True, use_poisson=True)

        camera_trajectory, ordered_image_names = parse_colmap_camera_trajectory(
            colmap_workspace, config)

        mesh = parse_mesh(colmap_workspace)

        rgbd_images, depth_map_range = parse_colmap_rgb_and_depth_data(
            camera_trajectory, ordered_image_names, colmap_workspace, config)

    elif reconstruction_mode.lower() == 'open3d':
        o3d_workspace = O3DWorkspace(dataset_idp)
        camera_trajectory = parse_o3d_trajectory(o3d_workspace)
        rgbd_images = parse_o3d_data(o3d_workspace)
        mesh = parse_mesh(o3d_workspace)
        depth_map_range = None
    else:
        assert False
    return rgbd_images, camera_trajectory, mesh, depth_map_range

