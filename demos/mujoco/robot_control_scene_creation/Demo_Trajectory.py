import classic_framework.mujoco.mujoco_utils.mujoco_controllers as mj_ctrl
from classic_framework.interface.Logger import RobotPlotFlags
from classic_framework.mujoco.MujocoRobot import MujocoRobot
from classic_framework.mujoco.MujocoScene import MujocoScene as Scene
from classic_framework.mujoco.mujoco_utils.mujoco_scene_object import MujocoPrimitiveObject
import pygame
from pygame.locals import *

class keyboard_simulation(object):
    def __init__(self):
        box1 = MujocoPrimitiveObject(obj_pos=[.5, -0.2, 0.35], obj_name="box1", geom_rgba=[0.1, 0.25, 0.3, 1])

        table = MujocoPrimitiveObject(obj_pos=[0.5, 0.0, 0.2],
                                    obj_name="table0",
                                    geom_size=[0.25, 0.35, 0.2],
                                    mass=2000)
        
        object_list = [box1, table]

        scene = Scene(object_list=object_list, control=mj_ctrl.MocapControl())

        self.mj_Robot = MujocoRobot(scene, gravity_comp=True, num_DoF=7)

        duration = 2

        self.mj_Robot.startLogging()  # this will start logging robots internal state
        self.mj_Robot.set_gripper_width = 0.0  # we set the gripper to clos at the beginning

        home_position = self.mj_Robot.current_c_pos.copy()  # store home position
        home_orientation = self.mj_Robot.current_c_quat.copy()  # store initial orientation

        # while True:
        #     for event in pygame.event.get():
        #         curr_position = mj_Robot.current_c_pos.copy()
        #         cur_orientation = mj_Robot.current_c_quat.copy()
        #         if event.type == QUIT:
        #             mj_Robot.stopLogging()
        #             mj_Robot.logger.plot(RobotPlotFlags.JOINTS | RobotPlotFlags.END_EFFECTOR)
        #         if enevnt.type == KEYDOWN:
        #             if event.key ==K_DOWN:
        #                 curr_position[2] -= 0.1
        #                 mj_Robot.gotoCartPositionAndQuat(curr_position, cur_orientation, duration=duration)
        #         elif event.type == KEYUP:
        #             if event.key ==K_DOWN:
        #                 mj_Robot.set_gripper_width = 0.04
        
        while True:
            # print("input for moving. position: X q+ w-, Y a+ s-, Z z+ x-")
            curr_position = self.mj_Robot.current_c_pos.copy()
            cur_orientation = self.mj_Robot.current_c_quat.copy()
            keyboard = input("input for moving. position: X q+ w-, Y a+ s-, Z z+ x-")
            # print(keyboard)
            if keyboard == " ":
                self.mj_Robot.stopLogging()
                self.mj_Robot.logger.plot(RobotPlotFlags.JOINTS | RobotPlotFlags.END_EFFECTOR)
            if keyboard == "z":
                curr_position[2] += 0.1
                print(curr_position, cur_orientation)
                self.mj_Robot.gotoCartPositionAndQuat(curr_position, cur_orientation, duration=duration)
            if keyboard == "x":
                curr_position[2] -= 0.1
                print(curr_position, cur_orientation)
                self.mj_Robot.gotoCartPositionAndQuat(curr_position, cur_orientation, duration=duration)
            if keyboard == "a":
                curr_position[1] += 0.1
                print(curr_position, cur_orientation)
                self.mj_Robot.gotoCartPositionAndQuat(curr_position, cur_orientation, duration=duration)
            if keyboard == "s":
                curr_position[1] -= 0.1
                print(curr_position, cur_orientation)
                self.mj_Robot.gotoCartPositionAndQuat(curr_position, cur_orientation, duration=duration)
            if keyboard == "q":
                curr_position[0] += 0.1
                print(curr_position, cur_orientation)
                self.mj_Robot.gotoCartPositionAndQuat(curr_position, cur_orientation, duration=duration)
            if keyboard == "w":
                curr_position[0] -= 0.1
                print(curr_position, cur_orientation)
                self.mj_Robot.gotoCartPositionAndQuat(curr_position, cur_orientation, duration=duration)
            # self.input_to_command(keyboard)
            switcher = {
                "o": self.gripper_open,
                "p": self.gripper_close
            }
            command = switcher.get(keyboard, lambda:"Invalid input")
            print(command)
            command()
    
    def gripper_open(self):
        self.mj_Robot.set_gripper_width = 0.04

    def gripper_close(self):
        self.mj_Robot.set_gripper_width = 0.0

    def input_to_command(self, argument):
        switcher = {
            "o": self.gripper_open,
            "p": self.gripper_close
        }
        command = switcher.get(argument, lambda:"Invalid input")
        print(command)
        command()
    



if __name__ == '__main__':
    keyboard_simulation()





