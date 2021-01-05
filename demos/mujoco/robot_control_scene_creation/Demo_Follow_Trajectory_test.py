from classic_framework.mujoco.MujocoRobot import MujocoRobot
from classic_framework.mujoco.MujocoScene import MujocoScene as Scene
from classic_framework.interface.Logger import RobotPlotFlags
from classic_framework.utils.sim_path import sim_framework_path
from classic_framework.mujoco.mujoco_utils.mujoco_scene_object import MujocoPrimitiveObject


import numpy as np

if __name__ == '__main__':
    box1 = MujocoPrimitiveObject(obj_pos=[.5, -0.2, 0.35], obj_name="box1", geom_rgba=[0.1, 0.25, 0.3, 1],
                                 geom_size=[0.02, 0.02, 0.04])
    box2 = MujocoPrimitiveObject(obj_pos=[.5, 0, 0.35], obj_name="box2", geom_rgba=[1, 0, 0, 1],
                                 geom_size=[0.02, 0.02, 0.04])

    table = MujocoPrimitiveObject(obj_pos=[0.5, 0.0, 0.2],
                                  obj_name="table0",
                                  geom_size=[0.25, 0.35, 0.2],
                                  mass=2000)

    object_list = [box1, box2, table]

    duration = 4
    # Setup the scene
    scene = Scene(object_list=object_list)

    mj_Robot = MujocoRobot(scene, gravity_comp=True, num_DoF=7)
    mj_Robot.startLogging()
    mj_Robot.gotoCartPositionAndQuat([0.51, -0.2, 0.4 + 0.04], [0, 1, 0, 0], duration=duration)
    # load the trajectory you want to follow
    path2file = sim_framework_path('demos/mujoco/robot_control_scene_creation/des_joint_traj_nonlinear.npy')
    des_joint_trajectory = np.load(path2file)
    print(len(des_joint_trajectory), len(des_joint_trajectory[0]))
    mj_Robot.follow_JointTraj(des_joint_trajectory)

    mj_Robot.stopLogging()
    mj_Robot.logger.plot(plotSelection=RobotPlotFlags.JOINTS)