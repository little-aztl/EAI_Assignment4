import argparse
from pathlib import Path
import shutil
import numpy as np
from tqdm import tqdm
from ..utils import to_pose, plan_move_qpos, execute_plan
from ..sim.wrapper_env import WrapperEnvConfig, WrapperEnv

generate_data_dir = Path("data")

start_index = 3000
num_data = 1000


def parse_argument():
    parser = argparse.ArgumentParser(description="Launcher config - Physics")
    parser.add_argument("--robot", type=str, default='galbot')
    parser.add_argument("--obj", type=str, default='power_drill')
    parser.add_argument("--ctrl_dt", type=float, default=0.02)
    parser.add_argument("--headless", type=int, default=0)
    parser.add_argument("--reset_wait_steps", type=int, default=100)

    args = parser.parse_args()
    print(type(args))
    return args

def construct_env(args:argparse.Namespace):
    env_config = WrapperEnvConfig(
        humanoid_robot=args.robot,
        obj_name=args.obj,
        headless=args.headless,
        ctrl_dt=args.ctrl_dt,
        reset_wait_steps=args.reset_wait_steps,
    )

    env = WrapperEnv(env_config)
    return env

def generate_data(env:WrapperEnv, save_path:str):
    env.launch()
    env.reset()

    arm_init_qpos = env.sim.humanoid_robot_cfg.joint_init_qpos[:7]
    arm_goal_qpos = arm_init_qpos + np.array([-0.4,0,0.8,0,0,0.10,-0.12])
    move_plan = plan_move_qpos(begin_qpos=arm_init_qpos, end_qpos=arm_goal_qpos, steps=20)
    execute_plan(env, move_plan)


    obs_wrist = env.get_obs(camera_id=1)
    env.debug_save_obs(obs_wrist, save_path)

    driller_pose = env.get_driller_pose()
    np.save(save_path / "driller_pose.npy", driller_pose)

    camera_intrinsic = env.sim.humanoid_robot_cfg.camera_cfg[1].intrinsics
    np.save(save_path / "camera_intrinsic.npy", camera_intrinsic)


def main():
    args = parse_argument()
    env = construct_env(args)

    for i in tqdm(range(num_data)):
        save_path = generate_data_dir / f"{i + start_index:04d}"
        save_path.mkdir(parents=True, exist_ok=True)
        generate_data(env, save_path)

    env.close()

if __name__ == "__main__":
    main()