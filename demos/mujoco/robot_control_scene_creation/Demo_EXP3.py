import sys
sys.path.append("/home/yucheng/alr_ws/ACNMP/NMP")
# Import Python libs
import torch
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from nmp.net import MPNet
from nmp import util
from nmp.data_process import BatchProcess

from classic_framework.mujoco.MujocoRobot import MujocoRobot
from classic_framework.mujoco.MujocoScene import MujocoScene as Scene
from classic_framework.mujoco.mujoco_utils.mujoco_scene_object import MujocoPrimitiveObject
from classic_framework.utils.sim_path import sim_framework_path
from classic_framework.interface.Logger import RobotPlotFlags

# Check if GPU is available
util.check_torch_device()

def cnmp_experiment(object_type, weight):
    util.print_wrap_title("CNMP Experiment 3 of Robot Pick and Place")
    config_path = util.get_config_path("CNMP_EXP3_config")
    config = util.parse_config(config_path)
    mp_net = MPNet(config, max_epoch=100000, init_epoch=100000)
    mp_net.fit()

    exp_title = "{}, Epoch: {}, {}".format("CNMP_PICK3",
                                           mp_net.epoch,
                                           "cov_act=softplus, "
                                           "min_num_ctx=0, wd=1e-3")
    desire_trajectory = soc_test(mp_net, object_type, weight, exp_title)
    return desire_trajectory

def soc_test(mp_net, object_type, weight, exp_title: str = None):
    # Prepare different context combinations for experiment
    condition_object = list()
    condition_weight = list()
    condition_cart = list()


    # Combination with 1 context point
    condition_object.append(np.array([object_type]))
    condition_weight.append(np.array([weight]))
    condition_cart.append(np.array(condition_cart))


    # Initialize of trajectories of each context combination
    # experiment_trajs = list()

    # Loop over combinations
    # Numpy to Torch
    condition_object_tensor = torch.Tensor(condition_object[0])
    condition_weight_tensor = torch.Tensor(condition_weight[0])
    condition_cart_tensor = torch.Tensor(condition_cart[0])
    condition_cart_time_tensor = torch.Tensor([])


    # Expand dimensions
    condition_object_tensor = condition_object_tensor[None, :, None]
    condition_weight_tensor = condition_weight_tensor[None, :, None]
    condition_cart_tensor = condition_cart_tensor[None, :, None]
    condition_cart_time_tensor = condition_cart_time_tensor[None, :, None]

    conditon_pair_dict = {"time": condition_cart_time_tensor,
                          "value": condition_cart_tensor}

    target_times = torch.Tensor(np.arange(0, 10, 0.1))
    target_times = target_times[None, :, None]

    # Normalize context and target points
    # TODO create dict for each joint_pos and normalize them seperatly?
    dict_obs = {"object_ctx": {"value": condition_object_tensor},
                "weight_ctx": {"value": condition_weight_tensor},
                "cart_pos_0":  conditon_pair_dict,
                "cart_pos_1":  conditon_pair_dict,
                "cart_pos_2":  conditon_pair_dict
                }
    dict_obs = BatchProcess.batch_normalize(dict_obs, mp_net.normalizer)
    dict_tar = {"cart_pos_0": {"time": target_times}}
    dict_tar = BatchProcess.batch_normalize(dict_tar,
                                            mp_net.normalizer)

    mean_val, diag_cov_val, non_diag_cov_val \
        = mp_net.predict(dict_obs=dict_obs,
                         tar_pos=dict_tar["cart_pos_0"]["time"])

    # Form up normal distribution
    mean_val, diag_cov_val \
        = BatchProcess.mean_std_denormalize({"cart_pos_0": mean_val[:, :, 0],
                                             "cart_pos_1": mean_val[:, :, 1],
                                             "cart_pos_2": mean_val[:, :, 2]
                                             },
                                            {"cart_pos_0": diag_cov_val[:, :, 0],
                                             "cart_pos_1": diag_cov_val[:, :, 1],
                                             "cart_pos_2": diag_cov_val[:, :, 2]
                                             },
                                            mp_net.normalizer)

    exp_trajs = list()
    for i in range(3):
        traj = mean_val["cart_pos_" + str(i)].squeeze().cpu().numpy()
        exp_trajs.append(traj)
    exp_trajs = np.stack(
        (exp_trajs[0], exp_trajs[1], exp_trajs[2]), axis=1)

    exp_trajs = exp_trajs[::10]

    # print(exp_trajs)

    return  exp_trajs

if __name__ == '__main__':
    # cnmp_experiment()
    object_type_arr = np.ones(10)
    weight_arr =np.array([1, 1, 1, 1, 1, 8, 8, 8, 8, 8])

    box1 = MujocoPrimitiveObject(obj_pos=[.5, 0, 0.35], obj_name="box1", geom_rgba=[0.1, 0.25, 0.3, 1],
                                 geom_size=[0.03, 0.03, 0.04],
                                 mass=0.1)

    table = MujocoPrimitiveObject(obj_pos=[0.5, 0.0, 0.2],
                                  obj_name="table0",
                                  geom_size=[0.25, 0.35, 0.2],
                                  mass=2000)

    # object_list = [box1, box2]
    object_list = [box1, table]

    scene = Scene(object_list=object_list, render=False)
    # , render=False)  # if we want to do mocap control
    # scene = Scene(object_list=object_list)                        # ik control is default

    mj_Robot = MujocoRobot(scene, gravity_comp=True, num_DoF=7)

    duration = 1  # you can specify how long a trajectory can be executed witht the duration

    home_position = [0.51, 0, 0.6]

    mj_Robot.gotoCartPositionAndQuat(home_position, [0, 1, 0, 0], duration=2 * duration)

    mj_Robot.set_gripper_width = 0.04

    mj_Robot.gotoCartPositionAndQuat([0.51, 0, 0.43], [0, 1, 0, 0], duration=2 * duration)
    mj_Robot.set_gripper_width = 0.00

    mj_Robot.gotoCartPositionAndQuat(home_position, [0, 1, 0, 0], duration=2 * duration)

    mj_Robot.startLogging()

    previous_trajectory = cnmp_experiment(object_type_arr[0],weight_arr[0])
    changed_trajectory = cnmp_experiment(object_type_arr[9],weight_arr[9])

    for i in range(10):
        current_trajectory = cnmp_experiment(object_type_arr[i],weight_arr[i])
        mj_Robot.gotoCartPositionAndQuat(current_trajectory[i], [0, 1, 0, 0], duration=duration)

    mj_Robot.stopLogging()
    print(mj_Robot.logger.cart_pos)
    x_pos = mj_Robot.logger.cart_pos.T[0]
    y_pos = mj_Robot.logger.cart_pos.T[1]
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.plot(x_pos, y_pos, linewidth=4)
    ax1.plot(previous_trajectory.T[0], previous_trajectory.T[1],'r-')
    ax1.plot(changed_trajectory.T[0], changed_trajectory.T[1], 'g-')
    circle1 = plt.Circle((0.36, 0.2), 0.02, color='r')
    circle2 = plt.Circle((0.36, -0.2), 0.02, color='b')
    square1 = patches.Rectangle((0.64, 0.18), 0.04, 0.04, color='r')
    square2 = patches.Rectangle((0.64, -0.22), 0.04, 0.04, color='b')
    ax1.add_patch(circle1)
    ax1.add_patch(circle2)
    ax1.add_patch(square2)
    ax1.add_patch(square1)
    ax1.set_xlim(0.2, 0.8)
    ax1.set_ylim(-0.3, 0.3)
    #ax1.set_title("CNMP prediction and executed trajectory")
    # fig, axs = plt.subplots(2)
    x_axis = np.linspace(0,9,10)
    ax2.plot(weight_arr/10)
    #set_title("normalized weight")
    plt.show()
    # mj_Robot.logger.plot(RobotPlotFlags.CART_POS)# | RobotPlotFlags.END_EFFECTOR)