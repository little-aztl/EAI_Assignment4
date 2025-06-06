import argparse
from typing import Optional, Tuple, List
from dataclasses import fields
import numpy as np
import cv2
from pyapriltags import Detector
from colorama import Fore, Back, Style

from src.type import Grasp
from src.config import Config
from src.utils import to_pose
from src.sim.wrapper_env import WrapperEnvConfig, WrapperEnv
from src.sim.wrapper_env import get_grasps
from src.test.load_test import load_test_data
from src.obj_pose_est import estimate_obj_pose
from src.utils import plan_move_qpos, execute_plan


def detect_driller_pose(img, depth, camera_matrix, camera_pose, config:Config, *args, **kwargs):
    """
    Detects the pose of driller, you can include your policy in args
    """
    # implement the detection logic here
    #
    pose = np.eye(4)

    # model = kwargs.get('model')
    # if model is None:
    #     raise ValueError("Model must be provided for detection.")

    # driller_pose_in_camera = model.predict(img, depth, camera_matrix)

    driller_pose_in_camera = estimate_obj_pose(
        depth,
        camera_matrix,
        camera_pose,
        config
    )
    pose = camera_pose @ driller_pose_in_camera

    return pose

def detect_marker_pose(
        detector: Detector,
        img: np.ndarray,
        camera_params: tuple,
        camera_pose: np.ndarray,
        tag_size: float = 0.12
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    # implement

    # Convert image to grayscale if it is not already
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    # Detect AprilTags in the image
    detections = detector.detect(img_gray,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=tag_size
    )
    if not detections:
        return None, None

    # Assuming we take the first detection
    detection = detections[0]
    rot_marker_camera = detection.pose_R
    trans_marker_camera = detection.pose_t

    # Convert to world coordinates
    T_marker_camera = np.eye(4)
    T_marker_camera[:3, :3] = rot_marker_camera
    T_marker_camera[:3, 3] = trans_marker_camera.flatten()
    T_marker_world = camera_pose @ T_marker_camera
    trans_marker_world = T_marker_world[:3, 3]
    rot_marker_world = T_marker_world[:3, :3]

    return trans_marker_world, rot_marker_world

def forward_quad_policy(pose, target_pose, *args, **kwargs):
    """ guide the quadruped to position where you drop the driller """
    # implement

    # Default parameters for the controller
    Kp_linear = kwargs.get('Kp_linear', 0.1)
    Kp_angular = kwargs.get('Kp_angular', 0.1)

    # Calculate the current and target positions and orientations
    current_xy = pose[:2, 3]
    target_xy = target_pose[:2, 3]
    current_yaw = np.arctan2(pose[1, 0], pose[0, 0])
    target_yaw = np.arctan2(target_pose[1, 0], target_pose[0, 0])

    # Calculate the errors in position and orientation
    linear_error = target_xy - current_xy
    angular_error = target_yaw - current_yaw
    current_yaw_sin = np.sin(current_yaw)
    current_yaw_cos = np.cos(current_yaw)
    dx_error = linear_error[0] * current_yaw_cos + linear_error[1] * current_yaw_sin
    dy_error = -linear_error[0] * current_yaw_sin + linear_error[1] * current_yaw_cos

    # Calculate the velocities
    dx_velocity = Kp_linear * dx_error
    dy_velocity = Kp_linear * dy_error
    angular_velocity = Kp_angular * angular_error

    action = np.array([dx_velocity, dy_velocity, angular_velocity])

    return action

def backward_quad_policy(pose, target_pose, *args, **kwargs):
    """ guide the quadruped back to its initial position """
    # implement

    # Default parameters for the controller
    Kp_linear = kwargs.get('Kp_linear', 0.1)
    Kp_angular = kwargs.get('Kp_angular', 0.1)

    # Calculate the current and target positions and orientations
    current_xy = pose[:2, 3]
    target_xy = target_pose[:2, 3]
    current_yaw = np.arctan2(pose[1, 0], pose[0, 0])
    target_yaw = np.arctan2(target_pose[1, 0], target_pose[0, 0])

    # Calculate the errors in position and orientation
    linear_error = target_xy - current_xy
    angular_error = target_yaw - current_yaw
    current_yaw_sin = np.sin(current_yaw)
    current_yaw_cos = np.cos(current_yaw)
    dx_error = linear_error[0] * current_yaw_cos + linear_error[1] * current_yaw_sin
    dy_error = -linear_error[0] * current_yaw_sin + linear_error[1] * current_yaw_cos

    # Calculate the velocities
    dx_velocity = Kp_linear * dx_error
    dy_velocity = Kp_linear * dy_error
    angular_velocity = Kp_angular * angular_error

    action = np.array([dx_velocity, dy_velocity, angular_velocity])

    return action

def plan_grasp(env: WrapperEnv, grasp: Grasp, grasp_config, *args, **kwargs) -> Optional[List[np.ndarray]]:
    """Try to plan a grasp trajectory for the given grasp. The trajectory is a list of joint positions. Return None if the trajectory is not valid."""
    # implement
    reach_steps = grasp_config['reach_steps']
    lift_steps = grasp_config['lift_steps']
    delta_dist = grasp_config['delta_dist']

    traj_reach = []
    traj_lift = []
    succ = False
    if not succ: return None

    return [np.array(traj_reach), np.array(traj_lift)]

def plan_move(env: WrapperEnv, begin_qpos, begin_trans, begin_rot, end_trans, end_rot, steps = 50, *args, **kwargs):
    """Plan a trajectory moving the driller from table to dropping position"""
    # implement
    traj = []

    succ = False
    if not succ: return None
    return traj

def open_gripper(env: WrapperEnv, steps = 10):
    for _ in range(steps):
        env.step_env(gripper_open=1)
def close_gripper(env: WrapperEnv, steps = 10):
    for _ in range(steps):
        env.step_env(gripper_open=0)



TESTING = True
DISABLE_GRASP = False
DISABLE_MOVE = True

def main():
    parser = argparse.ArgumentParser(description="Launcher config - Physics")
    parser.add_argument("--robot", type=str, default="galbot")
    parser.add_argument("--obj", type=str, default="power_drill")
    parser.add_argument("--ctrl_dt", type=float, default=0.02)
    parser.add_argument("--headless", type=int, default=0)
    parser.add_argument("--reset_wait_steps", type=int, default=100)
    parser.add_argument("--test_id", type=int, default=0)
    parser.add_argument('--config_path', type=str, default="configs/est_pose.yaml", help='Path to the config file')

    args = parser.parse_args()
    config = (
        Config()
        if args.config_path is None
        else Config.from_yaml(args.config_path)
    )

    for key, value in vars(args).items():
        value = getattr(args, key)
        if value is not None:
            setattr(config, key, value)

    detector = Detector(
        families="tagStandard52h13",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )

    env_config = WrapperEnvConfig(
        humanoid_robot=args.robot,
        obj_name=args.obj,
        headless=args.headless,
        ctrl_dt=args.ctrl_dt,
        reset_wait_steps=args.reset_wait_steps,
    )


    env = WrapperEnv(env_config)
    if TESTING:
        data_dict = load_test_data(args.test_id)
        env.set_table_obj_config(
            table_pose=data_dict['table_pose'],
            table_size=data_dict['table_size'],
            obj_pose=data_dict['obj_pose']
        )
        env.set_quad_reset_pos(data_dict['quad_reset_pos'])

    env.launch()
    env.reset(humanoid_qpos=env.sim.humanoid_robot_cfg.joint_init_qpos)
    humanoid_init_qpos = env.sim.humanoid_robot_cfg.joint_init_qpos[:7]
    Metric = {
        'obj_pose': False,
        'drop_precision': False,
        'quad_return': False,
    }

    head_init_qpos = np.array([0.0,0.0]) # you can adjust the head init qpos to find the driller

    env.step_env(humanoid_head_qpos=head_init_qpos)

    observing_qpos = humanoid_init_qpos + np.array([-0.4,0,0.8,0,0,0.10,-0.12]) # you can customize observing qpos to get wrist obs
    init_plan = plan_move_qpos(humanoid_init_qpos, observing_qpos, steps = 20)
    execute_plan(env, init_plan)


    # --------------------------------------step 1: move quadruped to dropping position--------------------------------------
    if not DISABLE_MOVE:
        forward_steps = 1000 # number of steps that quadruped walk to dropping position
        steps_per_camera_shot = 5 # number of steps per camera shot, increase this to reduce the frequency of camera shots and speed up the simulation
        head_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[0].intrinsics
        head_camera_params = (head_camera_matrix[0, 0],head_camera_matrix[1, 1],head_camera_matrix[0, 2],head_camera_matrix[1, 2])

        # implement this to guide the quadruped to target dropping position
        #
        target_container_pose = []
        def is_close(pose1, pose2, threshold): return True
        for step in range(forward_steps):
            if step % steps_per_camera_shot == 0:
                obs_head = env.get_obs(camera_id=0) # head camera
                trans_marker_world, rot_marker_world = detect_marker_pose(
                    detector,
                    obs_head.rgb,
                    head_camera_params,
                    obs_head.camera_pose,
                    tag_size=0.12
                )
                if trans_marker_world is not None:
                    # the container's pose is given by follows:
                    trans_container_world = rot_marker_world @ np.array([0,0.31,0.02]) + trans_marker_world
                    rot_container_world = rot_marker_world
                    pose_container_world = to_pose(trans_container_world, rot_container_world)

            quad_command = forward_quad_policy(pose_container_world, target_container_pose)
            move_head = False
            if move_head:
                # if you need moving head to track the marker, implement this
                head_qpos = [0,0]
                env.step_env(
                    humanoid_head_qpos=head_qpos,
                    quad_command=quad_command
                )
            else:
                env.step_env(quad_command=quad_command)
            if is_close(pose_container_world, target_container_pose, threshold=0.05): break


    # --------------------------------------step 2: detect driller pose------------------------------------------------------
    if not DISABLE_GRASP:
        obs_wrist = env.get_obs(camera_id=1) # wrist camera
        rgb, depth, camera_pose = obs_wrist.rgb, obs_wrist.depth, obs_wrist.camera_pose
        # print(f"{Fore.YELLOW}camera pose: {camera_pose}{Style.RESET_ALL}")
        wrist_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[1].intrinsics
        driller_pose = detect_driller_pose(rgb, depth, wrist_camera_matrix, camera_pose, config)
        # metric judgement
        Metric['obj_pose'] = env.metric_obj_pose(driller_pose)
        print(f"Metric['obj_pose]: {Metric['obj_pose']}")


    # --------------------------------------step 3: plan grasp and lift------------------------------------------------------
    if not DISABLE_GRASP:
        obj_pose = driller_pose.copy()
        grasps = get_grasps(args.obj)
        grasps0_n = Grasp(grasps[0].trans, grasps[0].rot @ np.diag([-1,-1,1]), grasps[0].width)
        grasps2_n = Grasp(grasps[2].trans, grasps[2].rot @ np.diag([-1,-1,1]), grasps[2].width)
        valid_grasps = [grasps[0], grasps0_n, grasps[2], grasps2_n] # we have provided some grasps, you can choose to use them or yours
        grasp_config = dict(
            reach_steps=0,
            lift_steps=0,
            delta_dist=0,
        ) # the grasping design in assignment 2, you can choose to use it or design yours

        for obj_frame_grasp in valid_grasps:
            robot_frame_grasp = Grasp(
                trans=obj_pose[:3, :3] @ obj_frame_grasp.trans
                + obj_pose[:3, 3],
                rot=obj_pose[:3, :3] @ obj_frame_grasp.rot,
                width=obj_frame_grasp.width,
            )
            grasp_plan = plan_grasp(env, robot_frame_grasp, grasp_config)
            if grasp_plan is not None:
                break
        if grasp_plan is None:
            print("No valid grasp plan found.")
            env.close()
            return
        reach_plan, lift_plan = grasp_plan

        pregrasp_plan = plan_move_qpos(env, observing_qpos, reach_plan[0], steps=50) # pregrasp, change if you want
        execute_plan(env, pregrasp_plan)
        open_gripper(env)
        execute_plan(env, reach_plan)
        close_gripper(env)
        execute_plan(env, lift_plan)


    # --------------------------------------step 4: plan to move and drop----------------------------------------------------
    if not DISABLE_GRASP and not DISABLE_MOVE:
        # implement your moving plan
        #
        move_plan = plan_move(
            env=env,
        )
        execute_plan(env, move_plan)
        open_gripper(env)


    # --------------------------------------step 5: move quadruped backward to initial position------------------------------
    if not DISABLE_MOVE:
        # implement
        #
        backward_steps = 1000 # customize by yourselves
        for step in range(backward_steps):
            # same as before, please implement this
            #
            quad_command = backward_quad_policy()
            env.step_env(
                quad_command=quad_command
            )


    # test the metrics
    Metric["drop_precision"] = Metric["drop_precision"] or env.metric_drop_precision()
    Metric["quad_return"] = Metric["quad_return"] or env.metric_quad_return()

    print("Metrics:", Metric)

    print("Simulation completed.")
    env.close()

if __name__ == "__main__":
    main()
