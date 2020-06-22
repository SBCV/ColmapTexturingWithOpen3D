from cto.ext.colmap.read_write_model import read_model

sfm_model_idp ='model_after_sfm'
dense_model_idp = 'workspace/sparse'

# Points are definitively equal

colmap_camera_sfm_parameter_dict, colmap_image_parameter_dict, _ = read_model(
    sfm_model_idp, ext='.bin')

colmap_camera_dense_parameter_dict, colmap_image_parameter_dict, _ = read_model(
    dense_model_idp, ext='.bin')


print('colmap_camera_sfm_parameter_dict', colmap_camera_sfm_parameter_dict)
print('colmap_camera_dense_parameter_dict', colmap_camera_dense_parameter_dict)

