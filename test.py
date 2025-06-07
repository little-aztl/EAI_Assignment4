import unittest
import numpy as np
from src.sim.wrapper_env import WrapperEnvConfig, WrapperEnv

@unittest.skipIf(
    False,  # Change to True if you want to skip tests when URDF is missing
    "Skipping tests for galbot IK solver due to missing URDF or chain initialization failure."
)
class TestGalbotIKSolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Use the exact initialization from main.py
        cls.args_mock = type('', (), {})() # Mock argparse object
        cls.args_mock.robot = "galbot"
        cls.args_mock.obj = "power_drill" # As per main defaults
        cls.args_mock.ctrl_dt = 0.02
        cls.args_mock.headless = 1 # Default in main is int(0)
        cls.args_mock.reset_wait_steps = 100
        # cls.args_mock.test_id = 0 # Not used by WrapperEnv init directly

        cls.env_config = WrapperEnvConfig(
            humanoid_robot=cls.args_mock.robot,
            obj_name=cls.args_mock.obj,
            headless=cls.args_mock.headless,
            ctrl_dt=cls.args_mock.ctrl_dt,
            reset_wait_steps=cls.args_mock.reset_wait_steps,
        )
        try:
            cls.env = WrapperEnv(cls.env_config)
        except Exception as e:
            # If URDF is missing or chain init fails, tests will be skipped by decorator,
            # but this provides more direct feedback if error is during WrapperEnv init.
            raise unittest.SkipTest(f"Failed to initialize WrapperEnv for galbot: {e}")


    def test_solve_ik_reachable_pose(self):
        """Test IK for a known reachable target pose with non-trivial qpos."""
        env = self.env
        num_total_dofs = env.humanoid_robot_cfg.num_dofs
        num_left_arm_dofs = len(env.left_arm_joint_ids) # Should be 7 for galbot based on mask

        # 1. Define a base_qpos (initial full robot joint angles)
        # For simplicity, let's use small non-zero values for the arm joints
        # and zeros for others.
        base_qpos = np.zeros(num_total_dofs, dtype=np.float32)
        initial_left_arm_qpos = np.array([0.1, -0.2, 0.3, 0.5, -0.1, 0.2, 0.1], dtype=np.float32) # 7 DoFs
        self.assertEqual(len(initial_left_arm_qpos), num_left_arm_dofs, "Test setup error: initial_left_arm_qpos length mismatch.")
        base_qpos[env.left_arm_joint_ids] = initial_left_arm_qpos

        # 2. Define a target_pose for the TCP (Tool Center Point)
        # This target_pose MUST be in the frame of 'left_arm_base_link'.
        # Let's try to achieve a pose that's slightly forward, down, and rotated.
        # This requires some knowledge of galbot's arm workspace.
        # We'll compute FK from a desired set of arm joint angles to get a reachable target.
        
        # Define a target set of *left arm* joint angles we want to reach
        target_left_arm_qpos_for_fk = np.array([0.2, 0.1, 0.5, 0.8, 0.1, 0.3, 0.2], dtype=np.float32)
        
        # Use the env's FK to get the target pose in 'left_arm_base_link' frame
        # This ensures the target_pose is actually reachable and consistent with the TCP definition.
        try:
            target_pose_in_arm_base = env.get_left_arm_fk(target_left_arm_qpos_for_fk)
        except ValueError as e:
            self.fail(f"FK computation failed during test setup: {e}") # FK must work for test to be valid


        self.assertIsNotNone(target_pose_in_arm_base, "FK for target setup failed.")
        # print(f"Target pose (relative to arm base) for IK:\n{target_pose_in_arm_base}")

        # 3. Call solve_ik
        # Use a slightly different base_qpos for the IK to solve from, to make it non-trivial
        ik_start_qpos = base_qpos.copy()
        ik_start_qpos[env.left_arm_joint_ids] = np.array([0.15, -0.15, 0.25, 0.55, -0.05, 0.15, 0.05], dtype=np.float32)

        solved_full_qpos = env.solve_ik(target_pose_in_arm_base, ik_start_qpos)

        self.assertIsNotNone(solved_full_qpos, "IK solver failed to find a solution for a reachable pose.")

        # 4. Verify Accuracy: Use FK on the IK solution's arm joints
        solved_left_arm_qpos = solved_full_qpos[env.left_arm_joint_ids]
        achieved_pose_in_arm_base = env.get_left_arm_fk(solved_left_arm_qpos)

        # Compare achieved_pose with target_pose (position and orientation)
        # Position error
        pos_error = np.linalg.norm(achieved_pose_in_arm_base[:3, 3] - target_pose_in_arm_base[:3, 3])
        self.assertLess(pos_error, 1e-3, f"IK position error too high: {pos_error:.6f} m") # Tolerance: 1mm

        # Orientation error (e.g., angle from axis-angle representation of R_target.T @ R_achieved)
        R_target = target_pose_in_arm_base[:3, :3]
        R_achieved = achieved_pose_in_arm_base[:3, :3]
        R_error = np.dot(R_target.T, R_achieved)
        angle_error_rad = np.arccos(np.clip((np.trace(R_error) - 1) / 2.0, -1.0, 1.0))
        self.assertLess(angle_error_rad, np.deg2rad(1.0), f"IK orientation error too high: {np.rad2deg(angle_error_rad):.4f} degrees") # Tol: 1 deg

        # 5. Verify Compatibility: Check if only left arm joints were modified
        mask_non_arm_joints = np.ones(num_total_dofs, dtype=bool)
        mask_non_arm_joints[env.left_arm_joint_ids] = False
        np.testing.assert_array_almost_equal(
            solved_full_qpos[mask_non_arm_joints],
            ik_start_qpos[mask_non_arm_joints],
            err_msg="IK solution modified non-arm joints."
        )
        # Also check that the arm joints *are* different (unless ik_start_qpos was already the solution)
        self.assertFalse(np.allclose(solved_left_arm_qpos, ik_start_qpos[env.left_arm_joint_ids]),
                         "IK solution for arm joints is identical to start; implies start was already solution or IK didn't move.")


    def test_solve_ik_unreachable_pose(self):
        """Test IK for a clearly unreachable target pose."""
        env = self.env
        base_qpos = np.zeros(env.humanoid_robot_cfg.num_dofs, dtype=np.float32)

        # Define a target_pose far outside the arm's workspace
        # (relative to 'left_arm_base_link')
        unreachable_target_pose = np.eye(4)
        unreachable_target_pose[0, 3] = 10.0 # 10 meters away in X (arm base frame)

        solved_qpos = env.solve_ik(unreachable_target_pose, base_qpos)
        self.assertIsNone(solved_qpos, "IK solver should return None for an unreachable pose.")

    def test_last_link_vector_and_tcp_definition(self):
        """
        Indirectly tests the last_link_vector (bias).
        If FK from a known q_arm results in pose_A (which includes the tool offset),
        and then IK to pose_A results in q_arm_solved, then q_arm_solved should be close to q_arm.
        This confirms the TCP definition is consistent between FK and IK.
        """
        env = self.env
        ik_start_qpos = np.zeros(env.humanoid_robot_cfg.num_dofs, dtype=np.float32)
        # A specific, non-trivial set of joint angles for the left arm
        known_left_arm_qpos = np.array([0.1, -0.1, 0.2, 0.3, -0.2, 0.1, -0.05], dtype=np.float32)
        ik_start_qpos[env.left_arm_joint_ids] = known_left_arm_qpos * 0.8 # Start IK from a bit away

        # 1. Calculate the FK pose using these known arm joint angles.
        # This pose inherently uses the `last_link_vector` from chain initialization.
        expected_tcp_pose_in_arm_base = env.get_left_arm_fk(known_left_arm_qpos)
        self.assertIsNotNone(expected_tcp_pose_in_arm_base, "FK failed in TCP definition test setup.")

        # 2. Now, use this FK-derived pose as the target for IK.
        solved_full_qpos = env.solve_ik(expected_tcp_pose_in_arm_base, ik_start_qpos)
        self.assertIsNotNone(solved_full_qpos, "IK failed to solve for a pose derived from FK.")

        # 3. The solved arm joint angles should be very close to the original known_left_arm_qpos.
        solved_left_arm_qpos = solved_full_qpos[env.left_arm_joint_ids]
        np.testing.assert_array_almost_equal(
            solved_left_arm_qpos,
            known_left_arm_qpos,
            decimal=4, # IKPy might not be perfectly exact back to the original q
            err_msg="IK solution does not match original qpos for FK-derived target, "
                    "suggests inconsistency in TCP / last_link_vector handling between FK/IK."
        )

if __name__ == '__main__':
    unittest.main()