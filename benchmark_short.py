import stable_baselines3
from rl_env import Sim
from nlp_infer import Planner
from env_test_util import env_pick_sth_into_drawer, env_cut, env_clean, env_round, env_multi1, env_multi2, env_multi3, env_multi4
import numpy as np
import random
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

test_number = 10
inf = []
c_list = []

UNSEEN_VERB = False
UNSEEN_NOUN = False
UNSEEN_VN = False
override_tac = True
# %% env_pick_sth_into_drawer
case_id_list = CASE_ENVIRONMENT_MAP.keys()
case_id_list = [1, 2, 3, 4, 5]
for case_id in case_id_list:
    infer_success = 0
    for _ in range(test_number):
        environment_id = CASE_ENVIRONMENT_MAP[case_id]
        env.force_environment_id = environment_id
        env.force_case_id = case_id
        obs = env.reset()

        sentence, tactile, label = env_pick_sth_into_drawer(env)
        if override_tac is True:
            tactile = 1
        sentence = label

        if UNSEEN_VERB == True:
            replaced_verb = random.choice(['Unclose', 'Unlock'])
            sentence.replace('Open', replaced_verb)
            replaced_verb = random.choice(['Grip', 'Catch'])
            sentence.replace('Grasp', replaced_verb)
            replaced_verb = random.choice(['Put', 'Set'])
            sentence.replace('Place', replaced_verb)
            replaced_verb = random.choice(['Shut'])
            sentence.replace('Close', replaced_verb)
        if UNSEEN_NOUN == True:
            replaced_noun = random.choice(['jar', 'cola'])
            sentence.replace('can', replaced_noun)
            replaced_noun = random.choice(['makeup'])
            sentence.replace('cosmetic', replaced_noun)
        if UNSEEN_VN == True:
            replaced_verb = random.choice(['Unclose', 'Unlock'])
            sentence.replace('Open', replaced_verb)
            replaced_verb = random.choice(['Grip', 'Catch'])
            sentence.replace('Grasp', replaced_verb)
            replaced_verb = random.choice(['Put', 'Set'])
            sentence.replace('Place', replaced_verb)
            replaced_verb = random.choice(['Shut'])
            sentence.replace('Close', replaced_verb)
            replaced_noun = random.choice(['jar', 'cola'])
            sentence.replace('can', replaced_noun)
            replaced_noun = random.choice(['makeup'])
            sentence.replace('cosmetic', replaced_noun)

        output_id, output_sentence = planner.test_nlp_infer('image.jpg', sentence, tactile)
        if output_sentence == label:
            infer_success += 1
        else:
            print(sentence, tactile, output_sentence, label)
    inf.append(infer_success)
    c_list.append(case_id)
# %% env_cut
case_id_list = CASE_ENVIRONMENT_MAP.keys()
case_id_list = [6, 7, 8]

for case_id in case_id_list:
    infer_success = 0
    for _ in range(test_number):
        environment_id = CASE_ENVIRONMENT_MAP[case_id]
        env.force_environment_id = environment_id
        env.force_case_id = case_id
        obs = env.reset()

        sentence, tactile, label = env_cut(env)
        if override_tac is True:
            tactile = 1
        sentence = label

        if UNSEEN_VERB == True:
            replaced_verb = random.choice(['Grip', 'Catch'])
            sentence.replace('Grasp', replaced_verb)
            replaced_verb = random.choice(['chop', 'slice'])
            sentence.replace('Cut', replaced_verb)
        if UNSEEN_NOUN == True:
            replaced_noun = random.choice(['fruit'])
            sentence.replace('apple', replaced_noun)
            replaced_noun = random.choice(['fruit'])
            sentence.replace('banana', replaced_noun)
        if UNSEEN_VN == True:
            replaced_verb = random.choice(['Grip', 'Catch'])
            sentence.replace('Grasp', replaced_verb)
            replaced_verb = random.choice(['chop', 'slice'])
            sentence.replace('Cut', replaced_verb)
            replaced_noun = random.choice(['fruit'])
            sentence.replace('apple', replaced_noun)
            replaced_noun = random.choice(['fruit'])
            sentence.replace('banana', replaced_noun)

        output_id, output_sentence = planner.test_nlp_infer('image.jpg', sentence, tactile)
        if output_sentence == label:
            infer_success += 1
        else:
            print(sentence, tactile, output_sentence, label)
    inf.append(infer_success)
    c_list.append(case_id)
# %% env_clean
case_id_list = CASE_ENVIRONMENT_MAP.keys()
case_id_list = [9, 10, 11]

for case_id in case_id_list:
    infer_success = 0
    for _ in range(test_number):
        environment_id = CASE_ENVIRONMENT_MAP[case_id]
        env.force_environment_id = environment_id
        env.force_case_id = case_id
        obs = env.reset()

        sentence, tactile, label = env_clean(env)
        if override_tac is True:
            tactile = 1
        sentence = label

        if UNSEEN_VERB == True:
            replaced_verb = random.choice(['Grip', 'Catch'])
            sentence.replace('Grasp', replaced_verb)
            replaced_verb = random.choice(['Put', 'Set'])
            sentence.replace('Place', replaced_verb)
        if UNSEEN_NOUN == True:
            replaced_noun = random.choice(['thing', 'item'])
            sentence.replace('object', replaced_noun)
        if UNSEEN_VN == True:
            replaced_verb = random.choice(['Grip', 'Catch'])
            sentence.replace('Grasp', replaced_verb)
            replaced_verb = random.choice(['Put', 'Set'])
            sentence.replace('Place', replaced_verb)
            replaced_noun = random.choice(['thing', 'item'])
            sentence.replace('object', replaced_noun)

        output_id, output_sentence = planner.test_nlp_infer('image.jpg', sentence, tactile)
        if output_sentence == label:
            infer_success += 1
        else:
            print(sentence, tactile, output_sentence, label)
    inf.append(infer_success)
    c_list.append(case_id)
# %% env_round
case_id_list = CASE_ENVIRONMENT_MAP.keys()
case_id_list = [30, 31, 32]

for case_id in case_id_list:
    infer_success = 0
    for _ in range(test_number):
        environment_id = CASE_ENVIRONMENT_MAP[case_id]
        env.force_environment_id = environment_id
        env.force_case_id = case_id
        obs = env.reset()

        sentence, tactile, label = env_round(env)
        if override_tac is True:
            tactile = 1
        sentence = label

        if UNSEEN_VERB == True:
            replaced_verb = random.choice(['Grip', 'Catch'])
            sentence.replace('Grasp', replaced_verb)
            replaced_verb = random.choice(['Put', 'Set'])
            sentence.replace('Place', replaced_verb)
        if UNSEEN_NOUN == True:
            replaced_noun = random.choice(['box', 'dustbin'])
            sentence.replace('bin', replaced_noun)
        if UNSEEN_VN == True:
            replaced_verb = random.choice(['Grip', 'Catch'])
            sentence.replace('Grasp', replaced_verb)
            replaced_verb = random.choice(['Put', 'Set'])
            sentence.replace('Place', replaced_verb)
            replaced_noun = random.choice(['box', 'dustbin'])
            sentence.replace('bin', replaced_noun)

        output_id, output_sentence = planner.test_nlp_infer('image.jpg', sentence, tactile)
        if output_sentence == label:
            infer_success += 1
        else:
            print(sentence, tactile, output_sentence, label)
    inf.append(infer_success)
    c_list.append(case_id)
# %%
print(inf)
print(c_list)
print(np.mean(np.array(inf)))
