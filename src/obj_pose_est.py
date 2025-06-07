import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .model.est_coord import EstCoordNet
from .model.est_pose import EstPoseNet
from .data.data_utils import depth2pcd, get_workspace_mask, pcd_camera2world

def get_model(config: Config) -> nn.Module:
    model_type = config.model_type

    if model_type == 'est_coord':
        try:
            return EstCoordNet(config)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize EstCoordNet: {e}")
    elif model_type == 'est_pose':
        try:
            return EstPoseNet(config)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize EstPoseNet: {e}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def load_checkpoint(model:nn.Module, checkpoint_path:str) -> None:
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {e}")

    try:
        model.load_state_dict(checkpoint['model'])
    except Exception as e:
        raise RuntimeError(f"Failed to load model state from checkpoint: {e}")

def estimate_obj_pose(depth:np.ndarray, camera_intrinsic:np.ndarray, camera_pose:np.ndarray, config: Config, *args, **kwargs) -> np.ndarray:
    model = get_model(config)
    assert hasattr(config, 'checkpoint') and config.checkpoint is not None, 'Checkpoint path is required for loading the model.'
    load_checkpoint(model, config.checkpoint)

    model.eval()

    pcd_camera = depth2pcd(depth, camera_intrinsic)
    pcd_world = pcd_camera2world(pcd_camera, camera_pose)
    pc_mask = get_workspace_mask(pcd_world)
    pcd_camera_masked = pcd_camera[pc_mask]

    assert hasattr(config, 'point_num') and isinstance(config.point_num, int) and config.point_num > 0, 'point_num is not valid.'
    selected_point_indices = np.random.randint(0, pcd_camera_masked.shape[0], config.point_num)
    downsampled_pcd = pcd_camera_masked[selected_point_indices]

    with torch.no_grad():
        pcd_tensor = torch.from_numpy(downsampled_pcd).float().unsqueeze(0) # (1, N, 3)
        assert hasattr(model, 'est') and callable(model.est), 'Model must have an `est` method for pose estimation.'
        pred_trans_tensor, pred_rot_tensor = model.est(pcd_tensor)
        pred_trans = pred_trans_tensor.squeeze(0).detach().cpu().numpy() # (3,)
        pred_rot = pred_rot_tensor.squeeze(0).detach().cpu().numpy() # (3, 3)

        ret = np.zeros((4, 4), dtype=pred_trans.dtype)
        ret[:3, :3] = pred_rot
        ret[:3, 3] = pred_trans
        ret[3, 3] = 1.0
        return ret

