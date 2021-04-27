"""
Test script for EXP5 Extrapolation in Robot Pick-and-Place Task

Check the result of RL agent, all results are saved as *.npy.

Author: Yucheng Tang
Email:tyc1333@gmail.com
Data:06.04.2021
"""

from classic_framework.mujoco.MujocoRobot import MujocoRobot
from classic_framework.mujoco.MujocoScene import MujocoScene as Scene
from classic_framework.utils.sim_path import sim_framework_path
from classic_framework.mujoco.mujoco_utils.mujoco_scene_object import MujocoPrimitiveObject
from classic_framework.interface.Logger import RobotPlotFlags

import numpy as np
import scipy as sp
import scipy.interpolate


class MujocoJointToCatersianTest():
    def __init__(self,
                 des_joint_trajectory, # n timestamp x 7
                 object_height,
                 collision_height) -> None:
        self.cat_pos = None
        self.collision_point_height = None
        box1 = MujocoPrimitiveObject(obj_pos=[.5, -0.2, 0.35], obj_name="box1", geom_rgba=[0.1, 0.25, 0.3, 1],
                                     geom_size=[0.02, 0.02, object_height])
        box2 = MujocoPrimitiveObject(obj_pos=[.5, 0, 0.35], obj_name="box2", geom_rgba=[1, 0, 0, 1],
                                     geom_size=[0.02, 0.02, collision_height])

        table = MujocoPrimitiveObject(obj_pos=[0.5, 0.0, 0.2],
                                      obj_name="table0",
                                      geom_size=[0.25, 0.35, 0.2],
                                      mass=2000)

        object_list = [box1, box2, table]

        duration = 2
        # Setup the scene
        scene = Scene(object_list=object_list,render=True)

        mj_Robot = MujocoRobot(scene, gravity_comp=True, num_DoF=7)

        mj_Robot.gotoCartPositionAndQuat([0.51, -0.2, 0.5], [0, 1, 0, 0], duration=duration)

        mj_Robot.set_gripper_width = 0.04

        mj_Robot.gotoCartPositionAndQuat([0.51, -0.2, 0.4], [0, 1, 0, 0], duration=duration)
        mj_Robot.set_gripper_width = 0.00

        mj_Robot.startLogging()

        # des_joint_trajectory = des_joint_trajectory[::40]
        # print(des_joint_trajectory.shape)

        des_joint_trajectory = self.interpolate_trajectory(des_joint_trajectory)

        # print(des_joint_trajectory.shape)

        mj_Robot.follow_JointTraj(des_joint_trajectory)

        mj_Robot.set_gripper_width = 0.04


        mj_Robot.stopLogging()
        # mj_Robot.logger.plot(RobotPlotFlags.JOINTS | RobotPlotFlags.END_EFFECTOR)

        self.cart_pos = mj_Robot.logger.cart_pos

        # z_max_index = np.argmax(self.cart_pos, axis=0)[-1]
        max_amplitude = (collision_height * 1.5 + object_height) * 1.5
        pick_height = 0.36 + object_height * 2

        self.collision_point_height = [[0.51, -0.2, pick_height],
                                       [0.51, 0, pick_height + max_amplitude],
                                       [0.51, 0.2, pick_height]]
        self.collision_point_height = np.array(self.collision_point_height)

    def interpolate_trajectory(self, trajectory: np.ndarray,
                               num_pt: int = 4000) -> np.array:
        x = np.arange(np.shape(trajectory.T)[1])
        new_length = num_pt
        new_x = np.linspace(x.min(), x.max(), new_length)
        new_trajectory = None
        for item in trajectory.T:
            new_item = sp.interpolate.interp1d(x, item, kind='cubic')(new_x)
            if new_trajectory is None:
                new_trajectory = new_item
            else:
                new_trajectory = np.vstack([new_trajectory, new_item])

        return new_trajectory.T

if __name__ == '__main__':
    path2file = sim_framework_path('/home/yucheng/alr_ws/ACNMP/NMP/nmp/trajectory/0.npy') # best_trajactory_5_12(report)
    des_joint_trajectory = np.load(path2file, allow_pickle=True)
    # print(des_joint_trajectory.shape)
    JointToCatersian = MujocoJointToCatersianTest(des_joint_trajectory=des_joint_trajectory,
                                              collision_height=0.028, object_height=0.018)
    print(JointToCatersian.cart_pos[::100], JointToCatersian.collision_point_height)