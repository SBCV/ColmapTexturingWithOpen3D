import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

print('o3d.__version__', o3d.__version__)

ofp = 'some/path/to/file.png'

mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)
vis.add_geometry(mesh)
vis.poll_events()
vis.update_renderer()

vis.capture_screen_image(ofp, True)

color = vis.capture_screen_float_buffer(True)
depth = vis.capture_depth_float_buffer(True)

vis.destroy_window()
color = np.asarray(color)
depth = np.asarray(depth)
plt.imshow(color)
plt.show()
plt.imshow(depth)
plt.show()
