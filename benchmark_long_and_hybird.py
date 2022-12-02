import stable_baselines3
from rl_env import Sim
import numpy as np
from nlp_infer import Planner
import random
from env_test_util import env_pick_sth_into_drawer, env_cut, env_clean, env_round, env_multi1, env_multi2, env_multi3, env_multi4

CASE_ENVIRONMENT_MAP = {
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,

    6: 2,
    7: 2,
    8: 2,


    9: 3,
    10: 3,
    11: 3,


    30: 4,
    31: 4,
    32: 4,

    12: 5,
    13: 5,
    14: 5,

    15: 6,
    16: 6,
    17: 6,

    18: 7,
    19: 7,
    20: 7,
    21: 7,
    22: 7,
    23: 7,
    24: 7,

    25: 8,
    26: 8,
    27: 8,
    28: 8,
    29: 8,

}

device = "cpu"
prompt_path = './NLP/prompt_model'
planner = Planner(prompt_path, device)

env = Sim(gui=False, discrete=True, number_of_objects=1, reset_interval=1, classification_model=None, test=False)
model = stable_baselines3.PPO.load('./rl.zip', env=env)
print('model load')

hyper_number = 100
strict_infer = True
nlp_only = False
override_tac = True
plan_suc = []
task_suc = []
# %% inferall
# exec
# 7.099999999999999645e-01
# 9.300000000000000488e-01
# 9.399999999999999467e-01
# 7.299999999999999822e-01
# 8.000000000000000444e-01
# 9.899999999999999911e-01
# 2.000000000000000111e-01
# 6.199999999999999956e-01
# %%ours table
# long_plan_succ = np.array([0.98, 0.97, 0.96, 0.72])
# print(np.mean(long_plan_succ))
# long_task_succ = np.array([0.68, 0.92, 0.94, 0.7])
# print(np.mean(long_task_succ))

# hybird_plan_succ = np.array([0.86, 0.99, 0.49, 0.79])
# hybird_plan_succ = np.array([0.99, 0.49, 0.79])
# print(np.mean(hybird_plan_succ))
# hybird_task_succ = np.array([0.74, 0.99, 0.31, 0.54])
# hybird_task_succ = np.array([0.99, 0.31, 0.54])
# print(np.mean(hybird_task_succ))
# %%ours-w/o touch

# long_plan_succ = np.array([0.69, 0.91, 0.1, 0.23])
# print(np.mean(long_plan_succ))
# hybird_plan_succ = np.array([0.99, 0.1, 0.18])
# print(np.mean(hybird_plan_succ))

# long_task_succ = np.array([0.51, 0.86, 0.14, 0.13])
# print(np.mean(long_task_succ))
# hybird_task_succ = np.array([0.95, 0.02, 0.07])
# print(np.mean(hybird_task_succ))
# %% env_pick_sth_into_drawer 88% plan
case_id_list = CASE_ENVIRONMENT_MAP.keys()
case_id_list = [1, 2, 3, 4, 5]

test_numbers = hyper_number
plan_fail = 0
task_fail = 0
failed_count = 0
for pr in range(test_numbers):
    plan_f = True
    task_f = True
    for case_id in case_id_list:
        number_of_episodes = 0
        t_reward = 0

        environment_id = CASE_ENVIRONMENT_MAP[case_id]
        env.force_environment_id = environment_id
        env.force_case_id = case_id
        env.benchmark = True

        obs = env.reset()
        done = False

        sentence, tactile, label = env_pick_sth_into_drawer(env)
        if override_tac is True:
            tactile = 1
        output_id, output_sentence = planner.test_nlp_infer('image.jpg', sentence, tactile)
        env.sentence_id = output_id
        obs = env.get_observation(0, 0, 0.1)

        if output_sentence != label and strict_infer == True:
            print(sentence, output_sentence, label)
            plan_f = False
            break
        elif output_sentence != label and strict_infer == False:
            plan_f = False
            pass

        if case_id == 5:
            break
        # print(output_id, output_sentence, label)

        while not done and nlp_only == False:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            t_reward += reward
            if done:
                number_of_episodes += 1
                obs = env.reset()
        if t_reward < 1 and nlp_only == False:
            task_f = False
            break
    if plan_f == False or task_f == False:
        failed_count += 1
        if plan_f == False:
            plan_fail += 1
        if task_f == False:
            task_fail += 1
    print(pr, failed_count)
plan_suc.append((hyper_number-plan_fail)/hyper_number)
task_suc.append((hyper_number-failed_count)/hyper_number)
print(plan_fail,task_fail)
np.savetxt('plan_suc.txt', plan_suc)
np.savetxt('task_suc.txt', task_suc)
print('task_done')
# %% env_cut
case_id_list = CASE_ENVIRONMENT_MAP.keys()
case_id_list = [6, 7, 8]

test_numbers = hyper_number
plan_fail = 0
task_fail = 0
failed_count = 0
for pr in range(test_numbers):
    plan_f = True
    task_f = True
    for case_id in case_id_list:
        number_of_episodes = 0
        t_reward = 0

        environment_id = CASE_ENVIRONMENT_MAP[case_id]
        env.force_environment_id = environment_id
        env.force_case_id = case_id
        env.benchmark = True

        obs = env.reset()
        done = False

        sentence, tactile, label = env_cut(env)
        if override_tac is True:
            tactile = 1
        output_id, output_sentence = planner.test_nlp_infer('image.jpg', sentence, tactile)
        env.sentence_id = output_id
        obs = env.get_observation(0, 0, 0.1)
        
        if case_id == 8:
            break

        if output_sentence != label and strict_infer == True:
            print(sentence, output_sentence, label, case_id)
            plan_f = False
            break
        elif output_sentence != label and strict_infer == False:
            plan_f = False
            pass
        
        # print(output_id, output_sentence, label)

        while not done and nlp_only == False:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            t_reward += reward
            if done:
                number_of_episodes += 1
                obs = env.reset()
        if t_reward < 1 and nlp_only == False:
            task_f = False
            break
    if plan_f == False or task_f == False:
        failed_count += 1
        if plan_f == False:
            plan_fail += 1
        if task_f == False:
            task_fail += 1
    print(pr, failed_count)
plan_suc.append((hyper_number-plan_fail)/hyper_number)
task_suc.append((hyper_number-failed_count)/hyper_number)
print(plan_fail,task_fail)
np.savetxt('plan_suc.txt', plan_suc)
np.savetxt('task_suc.txt', task_suc)
print('task_done')
# %% env_clean
case_id_list = CASE_ENVIRONMENT_MAP.keys()
case_id_list = [9, 10, 11]

test_numbers = hyper_number
plan_fail = 0
task_fail = 0
failed_count = 0
for pr in range(test_numbers):
    plan_f = True
    task_f = True
    for case_id in case_id_list:
        number_of_episodes = 0
        t_reward = 0

        environment_id = CASE_ENVIRONMENT_MAP[case_id]
        env.force_environment_id = environment_id
        env.force_case_id = case_id
        env.benchmark = True

        obs = env.reset()
        done = False

        sentence, tactile, label = env_clean(env)
        if override_tac is True:
            tactile = 1
        output_id, output_sentence = planner.test_nlp_infer('image.jpg', sentence, tactile)
        env.sentence_id = output_id
        obs = env.get_observation(0, 0, 0.1)

        if output_sentence != label and strict_infer == True:
            print(sentence, output_sentence, label, case_id)
            plan_f = False
            break
        elif output_sentence != label and strict_infer == False:
            plan_f = False
            pass

        if case_id == 11:
            break
        # print(output_id, output_sentence, label)

        while not done and nlp_only == False:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            t_reward += reward
            if done:
                number_of_episodes += 1
                obs = env.reset()
        if t_reward < 1 and nlp_only == False:
            task_f = False
            break
    if plan_f == False or task_f == False:
        failed_count += 1
        if plan_f == False:
            plan_fail += 1
        if task_f == False:
            task_fail += 1
    print(pr, failed_count)
plan_suc.append((hyper_number-plan_fail)/hyper_number)
task_suc.append((hyper_number-failed_count)/hyper_number)
print(plan_fail,task_fail)
np.savetxt('plan_suc.txt', plan_suc)
np.savetxt('task_suc.txt', task_suc)
print('task_done')
# %% env_round
case_id_list = CASE_ENVIRONMENT_MAP.keys()
case_id_list = [30, 31, 32]

test_numbers = hyper_number
plan_fail = 0
task_fail = 0
failed_count = 0
for pr in range(test_numbers):
    plan_f = True
    task_f = True
    for case_id in case_id_list:
        number_of_episodes = 0
        t_reward = 0

        environment_id = CASE_ENVIRONMENT_MAP[case_id]
        env.force_environment_id = environment_id
        env.force_case_id = case_id
        env.benchmark = True

        obs = env.reset()
        done = False

        sentence, tactile, label = env_round(env)
        if override_tac is True:
            tactile = 1
        output_id, output_sentence = planner.test_nlp_infer('image.jpg', sentence, tactile)
        env.sentence_id = output_id
        obs = env.get_observation(0, 0, 0.1)

        if output_sentence != label and strict_infer == True:
            print(sentence, output_sentence, label, case_id)
            plan_f = False
            break
        elif output_sentence != label and strict_infer == False:
            plan_f = False
            pass

        if case_id == 32:
            break
        # print(output_id, output_sentence, label)

        while not done and nlp_only == False:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            t_reward += reward
            if done:
                number_of_episodes += 1
                obs = env.reset()
        if t_reward < 1 and nlp_only == False:
            task_f = False
            break
    if plan_f == False or task_f == False:
        failed_count += 1
        if plan_f == False:
            plan_fail += 1
        if task_f == False:
            task_fail += 1
    print(pr, failed_count)
plan_suc.append((hyper_number-plan_fail)/hyper_number)
task_suc.append((hyper_number-failed_count)/hyper_number)
print(plan_fail,task_fail)
np.savetxt('plan_suc.txt', plan_suc)
np.savetxt('task_suc.txt', task_suc)
print('task_done')
# # %% multi1
# case_id_list = CASE_ENVIRONMENT_MAP.keys()
# case_id_list = [12, 13, 14]

# test_numbers = hyper_number
# plan_fail = 0
# task_fail = 0
# failed_count = 0
# for pr in range(test_numbers):
#     plan_f = True
#     task_f = True
#     for case_id in case_id_list:
#         number_of_episodes = 0
#         t_reward = 0

#         environment_id = CASE_ENVIRONMENT_MAP[case_id]
#         env.force_environment_id = environment_id
#         env.force_case_id = case_id
#         env.benchmark = True

#         obs = env.reset()
#         done = False

#         sentence, tactile, label = env_multi1(env)
#         if override_tac is True:
#             tactile = 1
#         output_id, output_sentence = planner.test_nlp_infer('image.jpg', sentence, tactile)
#         env.sentence_id = output_id
#         obs = env.get_observation(0, 0, 0.1)

#         if output_sentence != label and strict_infer == True:
#             print(sentence, output_sentence, label, case_id)
#             plan_f = False
#             break
#         elif output_sentence != label and strict_infer == False:
#             plan_f = False
#             pass

#         if case_id == 14:
#             break
#         # print(output_id, output_sentence, label)

#         while not done and nlp_only == False:
#             action, _ = model.predict(obs, deterministic=True)
#             obs, reward, done, _ = env.step(action)
#             t_reward += reward
#             if done:
#                 number_of_episodes += 1
#                 obs = env.reset()
#         if t_reward < 1 and nlp_only == False:
#             task_f = False
#             break
#     if plan_f == False or task_f == False:
#         failed_count += 1
#         if plan_f == False:
#             plan_fail += 1
#         if task_f == False:
#             task_fail += 1
#     print(pr, failed_count)
# plan_suc.append((hyper_number-plan_fail)/hyper_number)
# task_suc.append((hyper_number-failed_count)/hyper_number)
# print(plan_fail,task_fail)
# np.savetxt('plan_suc.txt', plan_suc)
# np.savetxt('task_suc.txt', task_suc)
# print('task_done')
# %% multi2
case_id_list = CASE_ENVIRONMENT_MAP.keys()
case_id_list = [15, 16, 17]

test_numbers = hyper_number
plan_fail = 0
task_fail = 0
failed_count = 0
for pr in range(test_numbers):
    plan_f = True
    task_f = True
    for case_id in case_id_list:
        number_of_episodes = 0
        t_reward = 0

        environment_id = CASE_ENVIRONMENT_MAP[case_id]
        env.force_environment_id = environment_id
        env.force_case_id = case_id
        env.benchmark = True

        obs = env.reset()
        done = False

        sentence, tactile, label = env_multi2(env)
        if override_tac is True:
            tactile = 1
        output_id, output_sentence = planner.test_nlp_infer('image.jpg', sentence, tactile)
        env.sentence_id = output_id
        obs = env.get_observation(0, 0, 0.1)

        if output_sentence != label and strict_infer == True:
            print(sentence, output_sentence, label, case_id)
            plan_f = False
            break
        elif output_sentence != label and strict_infer == False:
            plan_f = False
            pass

        if case_id == 17:
            break
        # print(output_id, output_sentence, label)

        # print(output_id, output_sentence)

        while not done and nlp_only == False:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            t_reward += reward
            if done:
                number_of_episodes += 1
                obs = env.reset()
        if t_reward < 1 and nlp_only == False:
            task_f = False
            break
    if plan_f == False or task_f == False:
        failed_count += 1
        if plan_f == False:
            plan_fail += 1
        if task_f == False:
            task_fail += 1
    print(pr, failed_count)
plan_suc.append((hyper_number-plan_fail)/hyper_number)
task_suc.append((hyper_number-failed_count)/hyper_number)
print(plan_fail,task_fail)
np.savetxt('plan_suc.txt', plan_suc)
np.savetxt('task_suc.txt', task_suc)
print('task_done')
# %% multi3
case_id_list = CASE_ENVIRONMENT_MAP.keys()
case_id_list = [18, 19, 20 ,21, 22, 23, 24]
test_numbers = hyper_number
plan_fail = 0
task_fail = 0
failed_count = 0
for pr in range(test_numbers):
    if random.random()>0.5:
        case_id_list = [19, 20 ,21, 22, 23, 24]
    else:
        case_id_list = [19, 20 ,21, 22, 23, 24]
    plan_f = True
    task_f = True
    for case_id in case_id_list:
        number_of_episodes = 0
        t_reward = 0

        environment_id = CASE_ENVIRONMENT_MAP[case_id]
        env.force_environment_id = environment_id
        env.force_case_id = case_id
        env.benchmark = True

        obs = env.reset()
        done = False

        sentence, tactile, label = env_multi3(env)
        if override_tac is True:
            tactile = 1
        output_id, output_sentence = planner.test_nlp_infer('image.jpg', sentence, tactile)
        env.sentence_id = output_id
        obs = env.get_observation(0, 0, 0.1)

        if output_sentence != label and strict_infer == True:
            # print(output_sentence, label, case_id)
            plan_f = False
            break
        elif output_sentence != label and strict_infer == False:
            plan_f = False
            pass

        if case_id == 24:
            break
        # print(output_id, output_sentence, label)

        while not done and nlp_only == False:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            t_reward += reward
            if done:
                number_of_episodes += 1
                obs = env.reset()
        if t_reward < 1 and nlp_only == False:
            task_f = False
            break
    if plan_f == False or task_f == False:
        failed_count += 1
        if plan_f == False:
            plan_fail += 1
        if task_f == False:
            task_fail += 1
    print(pr, failed_count)
plan_suc.append((hyper_number-plan_fail)/hyper_number)
task_suc.append((hyper_number-failed_count)/hyper_number)
print(plan_fail,task_fail)
np.savetxt('plan_suc.txt', plan_suc)
np.savetxt('task_suc.txt', task_suc)
print('task_done')
# %% multi4
case_id_list = CASE_ENVIRONMENT_MAP.keys()
case_id_list = [25, 26, 27, 28, 29]

test_numbers = hyper_number
plan_fail = 0
task_fail = 0
failed_count = 0
for pr in range(test_numbers):
    plan_f = True
    task_f = True
    for case_id in case_id_list:
        number_of_episodes = 0
        t_reward = 0

        environment_id = CASE_ENVIRONMENT_MAP[case_id]
        env.force_environment_id = environment_id
        env.force_case_id = case_id
        env.benchmark = True

        obs = env.reset()
        done = False

        sentence, tactile, label = env_multi4(env)
        if override_tac is True:
            tactile = 1
        output_id, output_sentence = planner.test_nlp_infer('image.jpg', sentence, tactile)
        env.sentence_id = output_id
        obs = env.get_observation(0, 0, 0.1)

        if output_sentence != label and strict_infer == True:
            print(sentence, output_sentence, label, case_id)
            plan_f = False
            break
        elif output_sentence != label and strict_infer == False:
            plan_f = False
            pass

        if case_id == 29:
            break
        # print(output_id, output_sentence, label)

        while not done and nlp_only == False:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            t_reward += reward
            if done:
                number_of_episodes += 1
                obs = env.reset()
        if t_reward < 1 and nlp_only == False:
            task_f = False
            break
    if plan_f == False or task_f == False:
        failed_count += 1
        if plan_f == False:
            plan_fail += 1
        if task_f == False:
            task_fail += 1
    print(pr, failed_count)
plan_suc.append((hyper_number-plan_fail)/hyper_number)
task_suc.append((hyper_number-failed_count)/hyper_number)
print(plan_fail,task_fail)
np.savetxt('plan_suc.txt', plan_suc)
np.savetxt('task_suc.txt', task_suc)
print('task_done')
# %%
print('plan_suc:', plan_suc)
print('task_suc:', task_suc)
