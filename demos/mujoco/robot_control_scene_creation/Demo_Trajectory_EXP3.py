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
    # print(start[0], end[0])
    x = np.linspace(start[0], end[0], num_pt)
    y = np.linspace(start[1], end[1], num_pt)
    z = np.linspace(start[2], end[2], num_pt)
    return np.array([x, y, z]).T

object_type = ["Cylinder", "Cube"]
weight = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
save_path = "/home/yucheng/alr_ws/SimulationFrameworkPublic/Demo_dataset/EXP3"
num_points = 100  # 4000 timeStamp for pick and place

if __name__ == '__main__':
    # Generate a random list for system error
    target_x = None
    target_y = None
    box_dim = None
    # print(random_list)

    # Remove existing directory
    remove_file_dir(save_path)

    # Generate directory in path
    os.makedirs(save_path)

    for object in object_type:
        if object is "Cube":
            target_x = 0.66
            box_dim = 0.02

        elif object is "Cylinder":
            target_x = 0.36
            box_dim = 0.03
        else:
            print("Unkown object type")
            break
        # print(target_x)
        for weight_value in weight:
            # Generate scene in simulation with one box and one collision
            box1 = MujocoPrimitiveObject(obj_pos=[.5, 0, 0.35], obj_name="box1", geom_rgba=[0.1, 0.25, 0.3, 1],
                                         geom_size=[box_dim, box_dim, 0.04],
                                         mass=0.1)

            table = MujocoPrimitiveObject(obj_pos=[0.5, 0.0, 0.2],
                                          obj_name="table0",
                                          geom_size=[0.25, 0.35, 0.2],
                                          mass=2000)

            # object_list = [box1, box2]
            object_list = [box1, table]

            scene = Scene(object_list=object_list,render=False)
                          # , render=False)  # if we want to do mocap control
            # scene = Scene(object_list=object_list)                        # ik control is default

            mj_Robot = MujocoRobot(scene, gravity_comp=True, num_DoF=7)

            duration = 1  # you can specify how long a trajectory can be executed witht the duration

            home_position = [0.51, 0, 0.6]

            mj_Robot.gotoCartPositionAndQuat(home_position, [0, 1, 0, 0], duration=2*duration)

            mj_Robot.set_gripper_width = 0.04

            mj_Robot.gotoCartPositionAndQuat([0.51, 0, 0.43], [0, 1, 0, 0], duration=2*duration)
            mj_Robot.set_gripper_width = 0.00

            mj_Robot.gotoCartPositionAndQuat(home_position, [0, 1, 0, 0], duration=2*duration)

            mj_Robot.startLogging()

            if weight_value <= 5:
                target_y = -0.2
            elif weight_value > 5:
                target_y = 0.2

            target_position =[target_x, target_y, 0.46]
            pos = interpolate_trajectory(home_position, target_position, 10)
            for pos_value in pos:
                mj_Robot.gotoCartPositionAndQuat(pos_value, [0, 1, 0, 0], duration=duration)

            mj_Robot.stopLogging()

            # print(len(mj_Robot.logger.time_stamp_list))

            # timeStamp for csv file
            step = int(10000/num_points)
            time_stamp = mj_Robot.logger.time_stamp_list[::step]
            df = pd.DataFrame({'time': np.array(time_stamp)-time_stamp[0]})

            # save pos information for trajectory follow function in joint space
            # run Demo_Follow_Trajectory_test.py to reproduce the trajectory (with oscillation)
            pos_res = []
            pos = mj_Robot.logger.cart_pos

            # pos and vel with 7 DoF for csv file
            for j in range(len(pos[0])):
                for k in range(len(pos)):
                    pos_res.append(pos[k][j])
                if df is None:
                    df = pd.DataFrame({"cart_pos_%d" % (j): pos_res[::step]})
                else:
                    df2 = pd.DataFrame({"cart_pos_%d"%(j): pos_res[::step]})
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
            if object is "Cube":
                object2num = 1
            elif object is "Cylinder":
                object2num = 2
            df2 = pd.DataFrame({"object_ctx": np.array([object2num]*len(mj_Robot.logger.time_stamp_list[::step]))})
            df = pd.concat([df, df2], axis=1)

            df2 = pd.DataFrame({"weight_ctx": np.array([weight_value]
                                                            * len(mj_Robot.logger.time_stamp_list[::step]))})
            df = pd.concat([df, df2], axis=1)

            if save_path is not None:
                # save as *.csv file
                df.to_csv(path_or_buf=save_path + "/" + str(object) + "_" + str(int(weight_value)) + ".csv",
                                index=False)


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