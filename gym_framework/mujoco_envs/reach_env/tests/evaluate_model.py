import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy as PpoMlpPolicy
from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy as SacMlpPolicy

from gym_framework.mujoco_envs.reach_env.reach_env import ReachEnvMocapCtrl, ReachEnvJointVelCtrl

EPISODE_LENGTH = 250
NSUBSTEPS = 12

NUM_EVALS = 5


def eval_model(env, model, deterministic=True, gamma=0.999):
    obs = env.reset()
    done = False
    rewards = []
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, done, _ = env.step(action.reshape(-1, ))
        rewards.append(r)
        obs = env.reset() if done else obs
    return np.sum([gamma ** t * rew for t, rew in enumerate(rewards)])


if __name__ == '__main__':
    env = ReachEnvJointVelCtrl(render=1, max_steps=EPISODE_LENGTH, nsubsteps=NSUBSTEPS, random_env=False)
    # model = PPO.load(path="models/ppo_pickplace.zip")
    model = SAC.load(path="models/sac_pickplace_best.zip")
    for _ in range(NUM_EVALS):
        print(eval_model(model=model, env=env, deterministic=True))
