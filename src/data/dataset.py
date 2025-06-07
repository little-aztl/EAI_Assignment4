from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import cv2
from torch.utils.data import Dataset

from ..utils import load_npy
from .data_utils import depth2pcd, load_depth, get_workspace_mask, pcd_camera2world, pcd_world2object

NUM_POINTS = 1024

class DrillerPoseDataset(Dataset):
    def __init__(self, data_dir:str):
        super(DrillerPoseDataset, self).__init__()
        self.data_dir = Path(data_dir)



    def _load_data(self):
        dataset_size = len([1 for entry in self.data_dir.iterdir() if entry.is_dir()])
        self.data_list = [1] * dataset_size

        for entry in tqdm(list(self.data_dir.iterdir()), desc="Loading Driller Pose Dataset"):
            if not entry.is_dir():
                continue

            camera_pose = load_npy((entry / "camera_pose.npy").as_posix())
            driller_pose = load_npy((entry / "driller_pose.npy").as_posix())
            camera_intrinsic = load_npy((entry / "camera_intrinsic.npy").as_posix())

            depth_array = load_depth((entry / "depth.png").as_posix())
            pcd_camera = depth2pcd(depth_array, camera_intrinsic)

            pcd_world = pcd_camera2world(camera_pose=camera_pose, pcd_camera=pcd_camera)
            pcd_object = pcd_world2object(pcd_world=pcd_world, object_pose=driller_pose)
            pc_mask = get_workspace_mask(pcd_world)
            sel_pc_idx = np.random.randint(0, np.sum(pc_mask), NUM_POINTS)

            pcd_camera_masked = pcd_camera[pc_mask][sel_pc_idx]
            pcd_object_masked = pcd_object[pc_mask][sel_pc_idx]

            rel_obj_pose = np.linalg.inv(camera_pose) @ driller_pose

            cur_data_dict = dict(
                pcd=pcd_camera_masked.astype(np.float32),
                coord=pcd_object_masked.astype(np.float32),
                trans=rel_obj_pose[:3, 3].astype(np.float32),
                rot=rel_obj_pose[:3, :3].astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                obj_pose_in_world=driller_pose.astype(np.float32),
            )

            data_index = int(entry.name)
            self.data_list[data_index] = cur_data_dict


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]





