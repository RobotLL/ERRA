import stable_baselines3
from rl_env import Sim
from tqdm import tqdm
import numpy as np
CASE_ENVIRONMENT_MAP = {
    1: 1,
    2: 1,
    3: 1,
    4: 1,

    6: 2,
    7: 2,


    9: 3,
    10: 3,


    30: 4,
    31: 4,

    12: 5,
    13: 5,

    15: 6,
    16: 6,

    18: 7,
    19: 7,
    20: 7,
    21: 7,
    22: 7,
    23: 7,

    25: 8,
    26: 8,
    27: 8,
    28: 8,

}

env = Sim(gui=False, discrete=True, number_of_objects=1, reset_interval=1, classification_model=None, test=False)
model = stable_baselines3.PPO.load('./rl.zip', env=env)
print('model load')
# %%
case_id_list = CASE_ENVIRONMENT_MAP.keys()
# [0.88, 0.84, 1, 1, 1, 0.91, 0.95, 1 ,0.96, 1]
case_id_list = [1, 2, 3, 4, 6, 7, 9, 10, 30, 31]
# case_id_list = [1, 2, 3, 4, 6, 7, 9, 10, 30, 31, 12, 13, 15, 16, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28]
test_time = 10
reward_mean = []
for case_id in case_id_list:
    number_of_episodes = 0
    t_reward = 0

    environment_id = CASE_ENVIRONMENT_MAP[case_id]
    env.force_environment_id = environment_id
    env.force_case_id = case_id
    env.benchmark = True

    for i in tqdm(range(test_time)):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            t_reward += reward
            if done:
                number_of_episodes += 1
                obs = env.reset()
    print(f'Case #{case_id}, mean reward: {t_reward/number_of_episodes}')
    reward_mean.append(t_reward/number_of_episodes)
print(reward_mean)
print(np.mean(np.array(reward_mean)))

