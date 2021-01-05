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

box_height = [0.01, 0.01, 0.02, 0.02, 0.04, 0.04, 0.06, 0.06]
save_path = "/home/yucheng/alr_ws/SimulationFrameworkPublic/Demo_dataset/nonlinear"

if __name__ == '__main__':
    random_list = []
    for i in range(8):
        random_list.append(0.05+np.random.random()/20)  # random 0.05~0.1
    # print(random_list[1])

    # Remove existing directory
    remove_file_dir(save_path)

    # Generate directory in path
    os.makedirs(save_path)

    for i in range(8):
        box1 = MujocoPrimitiveObject(obj_pos=[.5, -0.2, 0.35], obj_name="box1", geom_rgba=[0.1, 0.25, 0.3, 1],
                                     geom_size=[0.02, 0.02, 0.04])
        box2 = MujocoPrimitiveObject(obj_pos=[.5, 0, 0.35], obj_name="box2", geom_rgba=[1, 0, 0, 1],
                                     geom_size=[0.02, 0.02, box_height[i]])

        table = MujocoPrimitiveObject(obj_pos=[0.5, 0.0, 0.2],
                                      obj_name="table0",
                                      geom_size=[0.25, 0.35, 0.2],
                                      mass=2000)

        object_list = [box1, box2, table]

        scene = Scene(object_list=object_list, control=mj_ctrl.MocapControl())  # if we want to do mocap control
        # scene = Scene(object_list=object_list)                        # ik control is default

        mj_Robot = MujocoRobot(scene, gravity_comp=True, num_DoF=7)

        duration = 2  # you can specify how long a trajectory can be executed witht the duration

        cos_traj = generate_cos_func(max=box_height[i]+0.05+random_list[i], num_pt=10)

        # override_height = 0.5+box_height[i]+random_list[i]
        mj_Robot.gotoCartPositionAndQuat([0.51, -0.2, 0.6], [0, 1, 0, 0], duration=duration)
        mj_Robot.set_gripper_width = 0.04
        mj_Robot.gotoCartPositionAndQuat([0.51, -0.2, 0.4 + 0.04], [0, 1, 0, 0], duration=duration)
        mj_Robot.set_gripper_width = 0.00

        mj_Robot.startLogging()  # this will start logging robots internal state

        # execute the pick and place movements
        # mj_Robot.gotoCartPositionAndQuat([0.5, -0.2, override_height], [0, 1, 0, 0], duration=duration)
        # mj_Robot.gotoCartPositionAndQuat([0.5, 0.2, override_height], [0, 1, 0, 0], duration=duration)
        for index in range(len(cos_traj)):
            mj_Robot.gotoCartPositionAndQuat([0.5, cos_traj[index][0], 0.4 + 0.04 + cos_traj[index][1]], [0, 1, 0, 0], duration=duration/5)
        mj_Robot.gotoCartPositionAndQuat([0.5, 0.2, 0.4 + 0.04], [0, 1, 0, 0], duration=duration)

        mj_Robot.stopLogging()
        # mj_Robot.logger.joint_pos
        df = pd.DataFrame({'time': np.array(mj_Robot.logger.time_stamp_list[::60])})

        pos_res = []
        pos = mj_Robot.logger.joint_pos
        # pos = pos[::60]
        # print("11111111", len(pos), len(pos[0]))

        np.save("/home/yucheng/alr_ws/SimulationFrameworkPublic/demos/mujoco/robot_control_scene_creation/des_joint_traj_nonlinear.npy", pos)
        # # print("11111111", len(pos), len(pos[0]))
        #
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

        for j in range(len(pos[0])):
            for k in range(len(pos)):
                pos_res.append(pos[k][j])
            df2 = pd.DataFrame({"joint_pos_%d"%(j): pos_res[::60]})
            df = pd.concat([df, df2], axis=1)
            pos_res.clear()

        vel_res = []
        vel = mj_Robot.logger.joint_vel
        for j in range(len(vel[0])):
            for k in range(len(vel)):
                vel_res.append(vel[k][j])
            df2 = pd.DataFrame({"joint_vel_%d" % (j): vel_res[::60]})
            df = pd.concat([df, df2], axis=1)
            vel_res.clear()

        if save_path is not None:
            df.to_csv(path_or_buf=save_path + "/" + str(i) + ".csv",
                            index=False)
            dict = {'time': np.array(mj_Robot.logger.time_stamp_list[::60]),
                    'joint_pos': np.array(mj_Robot.logger.joint_pos_list[::60]),
                    'joint_vel': np.array(mj_Robot.logger.joint_pos_list[::60])}
            # print(dict)
            np.save("%s/%d" % (save_path, i), dict)

        mj_Robot.set_gripper_width = 0.04
        mj_Robot.gotoCartPositionAndQuat([0.5, 0.2, 0.65], [0, 1, 0, 0], duration=duration)
        # mj_Robot.gotoCartPositionAndQuat([0.5, 0.2, 0.4 + 0.04], [0, 1, 0, 0], duration=duration)
        # mj_Robot.set_gripper_width = 0.00

        # mj_Robot.gotoCartPositionAndQuat([0.5, 0.2, 0.65], [0, 1, 0, 0], duration=duration)
        # mj_Robot.gotoCartPositionAndQuat([0.5, -0.2, 0.65], [0, 1, 0, 0], duration=duration)
        # mj_Robot.gotoCartPositionAndQuat([0.5, -0.2, 0.4 + 0.04], [0, 1, 0, 0], duration=duration)
        # mj_Robot.set_gripper_width = 0.04