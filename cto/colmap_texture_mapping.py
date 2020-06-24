# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/color_map_optimization.py

import open3d as o3d

from cto.utility.logging_extension import logger
from cto.visualization import visualize_intermediate_result
from cto.config_api import create_config
from cto.config_api import compute_ofp
from cto.color_optimization import color_map_optimization
from cto.data_parsing.reconstruction_parsing import import_reconstruction


if __name__ == "__main__":

    # http://www.open3d.org/docs/release/tutorial/Advanced/color_map_optimization.html
    logger.vinfo('o3d.__version__', o3d.__version__)

    o3d.utility.set_verbosity_level(
        o3d.utility.VerbosityLevel.Debug)

    config = create_config()
    mesh_textured_max_iter_x_ofp = compute_ofp(config)
    rgbd_images, camera_trajectory, mesh, depth_map_max = import_reconstruction(config)

    visualize_intermediate_result(rgbd_images, camera_trajectory, mesh, config)

    color_map_optimization(
        mesh,
        rgbd_images,    # are used to compute gradient images
        camera_trajectory,
        ofp=mesh_textured_max_iter_x_ofp,
        config=config,
        maximum_allowable_depth=depth_map_max)

