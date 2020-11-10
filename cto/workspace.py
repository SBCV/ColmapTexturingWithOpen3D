import os


class ColmapWorkspace(object):

    def __init__(self, colmap_dataset_idp, use_geometric_depth_maps=True, use_poisson=True):
        self.model_idp = os.path.join(colmap_dataset_idp, "sparse")
        self.depth_map_idp = os.path.join(colmap_dataset_idp, "stereo", "depth_maps")
        self.depth_map_from_geometry_dp = os.path.join(colmap_dataset_idp, "depth_maps_from_geometry")

        self.color_image_idp = os.path.join(colmap_dataset_idp, "images")
        self.color_image_resized_dp = os.path.join(colmap_dataset_idp, "images_resized_cto")

        if use_poisson:
            self.mesh_ifp = os.path.join(colmap_dataset_idp, "meshed-poisson.ply")
        else:
            self.mesh_ifp = os.path.join(colmap_dataset_idp, "meshed-delaunay.ply")
        assert os.path.isfile(self.mesh_ifp)

        self.fused_ply_ifp = os.path.join(colmap_dataset_idp, "fused.ply")

        if use_geometric_depth_maps:
            self.depth_map_suffix = ".geometric.bin"
        else:
            self.depth_map_suffix = ".photometric.bin"

        self.depth_map_from_geometry_suffix = '.bin'


class O3DWorkspace(object):

    def __init__(self, o3d_dataset_idp):
        self.depth_map_idp = os.path.join(o3d_dataset_idp, "depth/")
        self.color_image_idp = os.path.join(o3d_dataset_idp, "image/")
        self.camera_traj_ifp = os.path.join(o3d_dataset_idp, "scene/key.log")
        self.mesh_ifp = os.path.join(o3d_dataset_idp, "scene", "integrated.ply")
