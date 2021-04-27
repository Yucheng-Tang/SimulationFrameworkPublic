"""
Dataset generation for EXP2 Robot Pick and Place Task

Robot places the object according to the height of the object and obstacle. A cosine-shaped trajectory with random noise
is generated for CNMP training. (A modification of initController in TrajectoryTracking.py is necessary)
Data form: time,joint_pos_0,joint_pos_1,joint_pos_2,joint_pos_3,joint_pos_4,joint_pos_5,joint_pos_6,box_ctx,collision_ctx

Author: Yucheng Tang
Email:tyc1333@gmail.com
Data:06.04.2021
"""

import classic_framework.mujoco.mujoco_utils.mujoco_controllers as mj_ctrl
from classic_framework.interface.Logger import RobotPlotFlags
from classic_framework.mujoco.MujocoRobot import MujocoRobot
from classic_framework.mujoco.MujocoScene import MujocoScene as Scene
from classic_framework.mujoco.mujoco_utils.mujoco_scene_object import MujocoPrimitiveObject
import numpy as np
import pandas as pd
import os
import shutil
from classic_framework.utils.sim_path import sim_framework_path



def remove_file_dir(path):
    """
    Remove file or directory
    Args:
        path: path to directory or file

    Returns:
        True if successfully remove file or directory

    """
    if not os.path.exists(path):
        return False
    elif os.path.isfile(path) or os.path.islink(path):
        os.unlink(path)
        return True
    else:
        shutil.rmtree(path)
        return True

def generate_cos_func(start: int = -0.2,
                      end: int = 0.2,
                      max: int = 0.02,
                      num_pt: int = 100) -> list:
    x = np.linspace(start, end, num_pt, endpoint=True)
    y = max * np.cos(x*np.pi/(2*end))
    traj = []
    for i in range(num_pt):
        traj.append([x[i],y[i]])

    return traj

def interpolate_trajectory(start: np.array,
                           end: np.array,
                           num_pt: int = 100) -> np.array:
    x = np.linspace(start[0], end[0], num_pt)
    y = np.linspace(start[1], end[1], num_pt)
    z = np.linspace(start[2], end[2], num_pt)
    return np.array([x, y, z]).T

# box_height = [0.02, 0.03, 0.015, 0.025]
# collision_height = [0.01, 0.015, 0.02, 0.025, 0.03]
box_height = [0.05]
collision_height = [0.05]
save_path = "/home/yucheng/alr_ws/SimulationFrameworkPublic/Demo_dataset/report/"
num_points = 100  # 4000 timeStamp for pick and place

if __name__ == '__main__':
    # Generate a random list for system error
    random_list = []
    for i in range(len(box_height)*len(collision_height)):
        random_list.append(np.random.uniform(-0.01, 0.01))  # random -0.2~0.2
    # print(random_list)

    # Remove existing directory
    remove_file_dir(save_path)

    # Generate directory in path
    os.makedirs(save_path)

    for box_index in range(len(box_height)):
        for collision_index in range(len(collision_height)):
            df = None

            # Generate scene in simulation with one box and one collision
            box1 = MujocoPrimitiveObject(obj_pos=[.5, -0.2, 0.35], obj_name="box1", geom_rgba=[0.1, 0.25, 0.3, 1],
                                         geom_size=[0.02, 0.02, box_height[box_index]],
                                         mass=1)
            box2 = MujocoPrimitiveObject(obj_pos=[.5, 0, 0.35], obj_name="box2", geom_rgba=[1, 0, 0, 1],
                                         geom_size=[0.02, 0.02, collision_height[collision_index]],
                                         mass=2000)

            table = MujocoPrimitiveObject(obj_pos=[0.5, 0.0, 0.2],
                                          obj_name="table0",
                                          geom_size=[0.25, 0.35, 0.2],
                                          mass=2000)

            # object_list = [box1, box2]
            object_list = [box1, box2, table]

            scene = Scene(object_list=object_list, control=mj_ctrl.MocapControl(),render=True)
                          # , render=False)  # if we want to do mocap control
            # scene = Scene(object_list=object_list)                        # ik control is default

            mj_Robot = MujocoRobot(scene, gravity_comp=True, num_DoF=7)

            duration = 2  # you can specify how long a trajectory can be executed witht the duration

            # Generate nonlinear trajectory, we use cosine function here
            random_index = box_index*len(collision_height)+ collision_index
            max_amplitude = (collision_height[collision_index]*1.5 + box_height[box_index] + random_list[random_index])*1.5 # 1.3
            cos_traj = generate_cos_func(max=max_amplitude, num_pt=100)
            # print(cos_traj)
            cos_traj = cos_traj[:80:16]+cos_traj[80:]
            # print(cos_traj)
            print(len(cos_traj))

            # Define a pick height according to the box height
            pick_height = 0.36 + box_height[box_index]*2

            # for continious trajectory follow
            home_position = mj_Robot.current_c_pos.copy()  # store home position
            home_orientation = mj_Robot.current_c_quat.copy()  # store initial orientation

            # mj_Robot.startLogging()  # this will start logging robots internal state

            position_step1 = np.array([0.51, -0.2, 0.6])
            pos = interpolate_trajectory(home_position,position_step1,3)
            ori = np.array([[0, 1, 0, 0], ] * pos.shape[0])
            t = np.linspace(1, 4, 3)
            print(pos.shape, ori.shape, t.shape)
            mj_Robot.gotoCartPositionAndQuat(pos[1:],
                                             ori[1:],
                                             duration=t[1:])
            mj_Robot.set_gripper_width = 0.04


            position_step2 = np.array([0.51, -0.2, pick_height])
            pos = interpolate_trajectory(position_step1, position_step2, 3)
            ori = np.array([[0, 1, 0, 0], ] * pos.shape[0])
            t = np.linspace(1, 4, 3)
            mj_Robot.gotoCartPositionAndQuat(pos[1:],
                                             ori[1:],
                                             duration=t[1:])
            print([0.51, -0.2, pick_height])

            mj_Robot.set_gripper_width = 0.00
            # Execute pick up movement
            # mj_Robot.gotoCartPositionAndQuat([0.51, -0.2, 0.6], [0, 1, 0, 0], duration=duration)
            # mj_Robot.set_gripper_width = 0.04
            # mj_Robot.gotoCartPositionAndQuat([0.51, -0.2, pick_height], [0, 1, 0, 0], duration=duration)
            # mj_Robot.set_gripper_width = 0.00

            mj_Robot.startLogging()  # this will start logging robots internal state

            # Follow the trajectory we generated
            # TODO solve velocity continuity issues
            # for index in range(len(cos_traj)):
            #     mj_Robot.gotoCartPositionAndQuat([0.5, cos_traj[index][0], pick_height + cos_traj[index][1]], [0, 1, 0, 0], duration=duration/5)
            # mj_Robot.gotoCartPositionAndQuat([0.5, 0.2, pick_height], [0, 1, 0, 0], duration=duration)

            x = np.ones(len(cos_traj)) * 0.51
            # print(x)
            # print(np.shape(np.array(cos_traj)))
            t = np.array(np.linspace(1, 4, 100))
            t = np.append(t[:80:16], t[80:], axis=0)
            print(len(t))
            pos = np.stack((x, np.array(cos_traj).T[0],pick_height + np.array(cos_traj).T[1])).T
            print(np.array(cos_traj).T[0], np.array(cos_traj).T[1])
            ori = np.array([[0, 1, 0, 0], ] * pos.shape[0])

            mj_Robot.gotoCartPositionAndQuat(pos[1:],
                                             ori[1:],
                                             duration=t[1:])


            mj_Robot.stopLogging()

            print(len(mj_Robot.logger.time_stamp_list))

            # timeStamp for csv file
            step = int(4000/num_points)
            time_stamp = mj_Robot.logger.time_stamp_list[::step]
            # start_time = time_stamp[0]
            # time_stamp[:] = [x - start_time for x in time_stamp]
            df = pd.DataFrame({'time': np.array(time_stamp)-time_stamp[0]})

            # save pos information for trajectory follow function in joint space
            # run Demo_Follow_Trajectory_test.py to reproduce the trajectory (with oscillation)
            pos_res = []
            pos = mj_Robot.logger.joint_pos
            # pos = pos[::4]
            # print(len(pos))


            np.save("/home/yucheng/alr_ws/SimulationFrameworkPublic/demos/mujoco/robot_control_scene_creation/des_joint_traj_nonlinear.npy", pos)

            # box1 = MujocoPrimitiveObject(obj_pos=[.5, -0.2, 0.35], obj_name="box1", geom_rgba=[0.1, 0.25, 0.3, 1],
            #                              geom_size=[0.02, 0.02, 0.04])
            # box2 = MujocoPrimitiveObject(obj_pos=[.5, 0, 0.35], obj_name="box2", geom_rgba=[1, 0, 0, 1],
            #                              geom_size=[0.02, 0.02, box_height[i]])
            #
            # table = MujocoPrimitiveObject(obj_pos=[0.5, 0.0, 0.2],
            #                               obj_name="table0",
            #                               geom_size=[0.25, 0.35, 0.2],
            #                               mass=2000)
            #
            # object_list = [box1, box2, table]
            #
            # scene = Scene(object_list=object_list, control=mj_ctrl.MocapControl())  # if we want to do mocap control
            # # scene = Scene(object_list=object_list)                        # ik control is default
            #
            # mj_Robot = MujocoRobot(scene, gravity_comp=True, num_DoF=7)
            #
            # duration = 2  # you can specify how long a trajectory can be executed witht the duration
            #
            # mj_Robot.gotoCartPositionAndQuat([0.5, 0.2, 0.4 + 0.04], [0, 1, 0, 0], duration=duration)
            #
            # mj_Robot.startLogging()
            # # load the trajectory you want to follow
            # path2file = sim_framework_path('demos/mujoco/robot_control_scene_creation/des_joint_traj_nonlinear.npy')
            # des_joint_trajectory = np.load(path2file)
            # print(len(des_joint_trajectory), len(des_joint_trajectory[0]))
            # mj_Robot.follow_JointTraj(des_joint_trajectory)
            #
            # mj_Robot.stopLogging()
            # df = pd.DataFrame({'time': np.array(mj_Robot.logger.time_stamp_list[::60])})

            # pos and vel with 7 DoF for csv file
            for j in range(len(pos[0])):
                for k in range(len(pos)):
                    pos_res.append(pos[k][j])
                if df is None:
                    df = pd.DataFrame({"joint_pos_%d" % (j): pos_res[::step]})
                else:
                    df2 = pd.DataFrame({"joint_pos_%d"%(j): pos_res[::step]})
                    df = pd.concat([df, df2], axis=1)
                pos_res.clear()

            # vel_res = []
            # vel = mj_Robot.logger.joint_vel
            # for j in range(len(vel[0])):
            #     for k in range(len(vel)):
            #         vel_res.append(vel[k][j])
            #     df2 = pd.DataFrame({"joint_vel_%d" % (j): vel_res[::step]})
            #     print(len(vel_res[::step]))
            #     df = pd.concat([df, df2], axis=1)
            #     vel_res.clear()

            df2 = pd.DataFrame({"box_ctx": np.array([box_height[box_index]]*len(mj_Robot.logger.time_stamp_list[::step]))})
            df = pd.concat([df, df2], axis=1)

            df2 = pd.DataFrame({"collision_ctx": np.array([collision_height[collision_index]]
                                                            * len(mj_Robot.logger.time_stamp_list[::step]))})
            df = pd.concat([df, df2], axis=1)

            if save_path is not None:
                # save as *.csv file
                df.to_csv(path_or_buf=save_path + "/" + str(random_index) + ".csv",
                                index=False)

                # save as *.npy file
                # dict = {'time': np.array(mj_Robot.logger.time_stamp_list[::step]),
                #         'joint_pos': np.array(mj_Robot.logger.joint_pos_list[::step]),
                #         'box_height':np.array(box_height[box_index]*len(mj_Robot.logger.time_stamp_list[::step])),
                #         'collision_height':np.array(collision_height[collision_index]*len(mj_Robot.logger.time_stamp_list[::step]))}
                #         # 'joint_vel': np.array(mj_Robot.logger.joint_pos_list[::step])}
                dict = np.array(mj_Robot.logger.joint_pos_list[::])
                np.save("%s/%d" % (save_path, random_index), dict)

            # Execute place movement
            mj_Robot.set_gripper_width = 0.04
            # mj_Robot.logger.plot(RobotPlotFlags.JOINTS | RobotPlotFlags.END_EFFECTOR)
            # print(scene.object_list[0].obj_pos)
            # mj_Robot.logger.plot(RobotPlotFlags.GRIPPER_POS)

            # mj_Robot.gotoCartPositionAndQuat([0.5, 0.2, 0.65], [0, 1, 0, 0], duration=duration)
        # mj_Robot.gotoCartPositionAndQuat([0.5, 0.2, 0.4 + 0.04], [0, 1, 0, 0], duration=duration)
        # mj_Robot.set_gripper_width = 0.00

        # mj_Robot.gotoCartPositionAndQuat([0.5, 0.2, 0.65], [0, 1, 0, 0], duration=duration)
        # mj_Robot.gotoCartPositionAndQuat([0.5, -0.2, 0.65], [0, 1, 0, 0], duration=duration)
        # mj_Robot.gotoCartPositionAndQuat([0.5, -0.2, 0.4 + 0.04], [0, 1, 0, 0], duration=duration)
        # mj_Robot.set_gripper_width = 0.04