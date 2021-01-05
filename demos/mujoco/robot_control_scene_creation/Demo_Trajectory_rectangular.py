import classic_framework.mujoco.mujoco_utils.mujoco_controllers as mj_ctrl
from classic_framework.interface.Logger import RobotPlotFlags
from classic_framework.mujoco.MujocoRobot import MujocoRobot
from classic_framework.mujoco.MujocoScene import MujocoScene as Scene
from classic_framework.mujoco.mujoco_utils.mujoco_scene_object import MujocoPrimitiveObject
import numpy as np
import pandas as pd
import os
import shutil


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

box_height = [0.01, 0.02, 0.04, 0.04, 0.06, 0.06, 0.08, 0.08]
save_path = "/home/yucheng/alr_ws/SimulationFrameworkPublic/Demo_dataset/rect"

if __name__ == '__main__':
    random_list = []
    for i in range(len(box_height)):
        random_list.append(np.random.random()/10)  # random 0~0.1
    # print(random_list[1])

    # Remove existing directory
    remove_file_dir(save_path)

    # Generate directory in path
    os.makedirs(save_path)

    for i in range(len(box_height)):
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

        override_height = 0.5+box_height[i]+random_list[i]
        mj_Robot.gotoCartPositionAndQuat([0.51, -0.2, 0.6], [0, 1, 0, 0], duration=duration)
        mj_Robot.set_gripper_width = 0.04
        mj_Robot.gotoCartPositionAndQuat([0.51, -0.2, 0.4 + 0.04], [0, 1, 0, 0], duration=duration)
        mj_Robot.set_gripper_width = 0.00

        mj_Robot.startLogging()  # this will start logging robots internal state
        mj_Robot.set_gripper_width = 0.0  # we set the gripper to clos at the beginning

        # execute the pick and place movements
        mj_Robot.gotoCartPositionAndQuat([0.5, -0.2, override_height], [0, 1, 0, 0], duration=duration)
        mj_Robot.gotoCartPositionAndQuat([0.5, 0.2, override_height], [0, 1, 0, 0], duration=duration)
        mj_Robot.gotoCartPositionAndQuat([0.5, 0.2, 0.4 + 0.04], [0, 1, 0, 0], duration=duration)

        mj_Robot.stopLogging()
        # mj_Robot.logger.joint_pos
        df = pd.DataFrame({'time': np.array(mj_Robot.logger.time_stamp_list[::60])})

        pos_res = []
        pos = mj_Robot.logger.joint_pos
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
            dict = {'time' : np.array(mj_Robot.logger.time_stamp_list[::60]),
                    'joint_pos' : np.array(mj_Robot.logger.joint_pos_list[::60]),
                    'joint_vel' : np.array(mj_Robot.logger.joint_pos_list[::60])}
            print(dict)
            np.save("%s/%d"%(save_path, i), dict)

        mj_Robot.set_gripper_width = 0.04
        mj_Robot.gotoCartPositionAndQuat([0.5, 0.2, 0.6], [0, 1, 0, 0], duration=duration)
        # mj_Robot.gotoCartPositionAndQuat([0.5, 0.2, 0.4 + 0.04], [0, 1, 0, 0], duration=duration)
        # mj_Robot.set_gripper_width = 0.00
        #
        # mj_Robot.gotoCartPositionAndQuat([0.5, 0.2, 0.8], [0, 1, 0, 0], duration=duration)
        # mj_Robot.gotoCartPositionAndQuat([0.5, -0.2, 0.8], [0, 1, 0, 0], duration=duration)
        # mj_Robot.gotoCartPositionAndQuat([0.5, -0.2, 0.4 + 0.04], [0, 1, 0, 0], duration=duration)
        # mj_Robot.set_gripper_width = 0.04
