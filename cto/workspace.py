import os


class ColmapWorkspace(object):

    def __init__(self, colmap_dataset_idp, use_geometric_depth_maps=True, use_poisson=True):
        self.model_idp = os.path.join(colmap_dataset_idp, "sparse")
        self.depth_image_idp = os.path.join(colmap_dataset_idp, "stereo", "depth_maps")
        self.color_image_idp = os.path.join(colmap_dataset_idp, "images")
        self.color_image_resized_dp = os.path.join(colmap_dataset_idp, "images_resized_cto")

        if use_poisson:
            self.mesh_ifp = os.path.join(colmap_dataset_idp, "meshed-poisson.ply")
        else:
            self.mesh_ifp = os.path.join(colmap_dataset_idp, "meshed-delaunay.ply")

        if use_geometric_depth_maps:
            self.depth_map_suffix = ".geometric.bin"
        else:
            self.depth_map_suffix = ".photometric.bin"


class O3DWorkspace(object):

    def __init__(self, o3d_dataset_idp):
        self.depth_image_idp = os.path.join(o3d_dataset_idp, "depth/")
        self.color_image_idp = os.path.join(o3d_dataset_idp, "image/")
        self.camera_traj_ifp = os.path.join(o3d_dataset_idp, "scene/key.log")
        self.mesh_ifp = os.path.join(o3d_dataset_idp, "scene", "integrated.ply")

