import open3d as o3d


def visualize_rgbd_image(rgbd_image, camera_parameters):
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        camera_parameters.intrinsic)
    o3d.visualization.draw_geometries([point_cloud])


def visualize_rgbd_image_list(rgbd_image_list, camera_trajectory, num_images=None, additional_point_cloud_list=None):
    camera_parameter_list = camera_trajectory.parameters
    assert len(rgbd_image_list) == len(camera_parameter_list)

    if num_images is None:
        num_images = len(rgbd_image_list)

    point_cloud_list = []
    for rgbd_image, camera_parameters in zip(rgbd_image_list[:num_images], camera_parameter_list[:num_images]):
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            camera_parameters.intrinsic,
            camera_parameters.extrinsic)
        point_cloud_list.append(point_cloud)

    if additional_point_cloud_list is not None:
        point_cloud_list += additional_point_cloud_list

    print("Drawing " + str(num_images) + " images.")
    o3d.visualization.draw_geometries(point_cloud_list)

