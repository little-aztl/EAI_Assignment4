import sys
sys.path.append('.')

import numpy as np

from src.data.data_utils import depth2pcd, load_depth, get_workspace_mask
from src.utils import load_npy


depth_array = load_depth("data/0000/depth.png")
intrinsics = load_npy("data/0000/camera_intrinsic.npy")
camera_pose = load_npy("data/0000/camera_pose.npy")

point_cloud_camera = depth2pcd(depth_array, intrinsics)
np.save("sandbox/results/point_cloud.npy", point_cloud_camera)

point_cloud_world = np.einsum("ab,nb->na", camera_pose[:3, :3], point_cloud_camera) + camera_pose[:3, 3]
np.save("sandbox/results/point_cloud_world.npy", point_cloud_world)

pc_mask = get_workspace_mask(point_cloud_world)
point_cloud_masked_camera = point_cloud_camera[pc_mask]
np.save("sandbox/results/point_cloud_masked_camera.npy", point_cloud_masked_camera)