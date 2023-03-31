import stable_baselines3
from rl_env import Sim
from nlp_infer import Planner
from util import render_camera
from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np

# UNSEEN_VERB = True
# UNSEEN_NOUN = False
# UNSEEN_VN = False

# plan1 = np.array([0.55, 0.76, 0.97, 0.79])
# print('plan1:', np.mean(plan1))
# exec1 = np.array([0.35, 0.67, 0.94, 0.74])
# print('exec1:', np.mean(exec1))


# plan2 = np.array([0.7, 0.89, 0.33, 0.32])
# plan2 = np.array([0.89, 0.33, 0.32])
# print('plan2:', np.mean(plan2))
# exec2 = np.array([0.56, 0.9, 0.11, 0.25])
# exec2 = np.array([0.9, 0.11, 0.25])
# print('exec2:', np.mean(exec2))


# UNSEEN_VERB = False
# UNSEEN_NOUN = True
# UNSEEN_VN = False

# plan1 = np.array([0.7, 0.8, 0.89, 0.8])
# print('plan1:', np.mean(plan1))
# exec1 = np.array([0.53, 0.74, 0.85, 0.77])
# print('exec1:', np.mean(exec1))

# plan2 = np.array([0, 0.95, 0.52, 0.17])
# plan2 = np.array([0.95, 0.52, 0.17])
# print('plan2:', np.mean(plan2))
# exec2 = np.array([0, 0.88, 0.19, 0.1])
# exec2 = np.array([0.88, 0.19, 0.1])
# print('exec2:', np.mean(exec2))

# UNSEEN_VERB = False
# UNSEEN_NOUN = False
# UNSEEN_VN = True

# plan1 = np.array([0.5, 0.5, 0.34, 0.74])
# print('plan1:', np.mean(plan1))
# exec1 = np.array([0.35, 0.52, 0.33, 0.69])
# print('exec1:', np.mean(exec1))

# plan2 = np.array([0, 0.67, 0.26, 0.07])
# plan2 = np.array([0.67, 0.26, 0.07])
# print('plan2:', np.mean(plan2))
# exec2 = np.array([0, 0.6, 0.1, 0.04])
# exec2 = np.array([0.6, 0.1, 0.04])
# print('exec2:', np.mean(exec2))

UNSEEN_VERB = False
UNSEEN_NOUN = False
UNSEEN_VN = False

def env_pick_sth_into_drawer(env):
    case_id = env.case_id
    object_name = env.object_name
    tactile = env.tactile
    sim_d, sim_float, mask = render_camera()

    image = Image.fromarray(sim_d).convert('L')
    image.save('image.jpg')

    high_level_command = ['Please put the ' + object_name + ' into the drawer']

    if UNSEEN_VERB == True:
        replaced_verb = random.choice(['place'])
        high_level_command = ['Please ' + replaced_verb + ' the ' + object_name + ' into the drawer']
    if UNSEEN_NOUN == True:
        if object_name == 'can':
            object_name = random.choice(['jar', 'cola'])
        if object_name == 'cosmetic':
            object_name = random.choice(['makeup'])
        high_level_command = ['Please put the ' + object_name + ' into the drawer']
    if UNSEEN_VN == True:
        replaced_verb = random.choice(['place'])
        if object_name == 'can':
            object_name = random.choice(['jar', 'cola'])
        if object_name == 'cosmetic':
            object_name = random.choice(['makeup'])
        high_level_command = ['Please ' + replaced_verb + ' the ' + object_name + ' into the drawer']

    low_level_command = ['Open the drawer',
                         'Grasp the ' + object_name,
                         'Place the ' + object_name + ' into the drawer',
                         'Close the drawer',
                         'Done']
    sentence = high_level_command[0]
    if case_id == 1:
        label = low_level_command[0]
    elif case_id == 2:
        label = low_level_command[1]
    elif case_id == 3:
        label = low_level_command[2]
    elif case_id == 4:
        label = low_level_command[3]
    elif case_id == 5:
        label = low_level_command[4]
    return sentence, tactile, label


def env_cut(env):
    case_id = env.case_id
    object_name = env.object_name
    tactile = env.tactile
    sim_d, sim_float, mask = render_camera()

    image = Image.fromarray(sim_d).convert('L')
    image.save('image.jpg')

    high_level_command = ['Please cut the ' + object_name]

    if UNSEEN_VERB == True:
        replaced_verb = random.choice(['chop', 'slice'])
        high_level_command = ['Please ' + replaced_verb + ' the ' + object_name]
    if UNSEEN_NOUN == True:
        if object_name == 'apple':
            object_name = random.choice(['fruit'])
        if object_name == 'banana':
            object_name = random.choice(['fruit'])
        high_level_command = ['Please cut the ' + object_name]
    if UNSEEN_VN == True:
        replaced_verb = random.choice(['chop', 'slice'])
        if object_name == 'apple':
            object_name = random.choice(['fruit'])
        if object_name == 'banana':
            object_name = random.choice(['fruit'])
        high_level_command = ['Please ' + replaced_verb + ' the ' + object_name]

    low_level_command = ['Grasp the knife', 'Cut the ' + object_name, 'Done']

    sentence = high_level_command[0]
    if case_id == 6:
        label = low_level_command[0]
    elif case_id == 7:
        label = low_level_command[1]
    elif case_id == 8:
        label = low_level_command[2]
    return sentence, tactile, label


def env_clean(env):
    case_id = env.case_id
    object_name = env.object_name
    tactile = env.tactile
    sim_d, sim_float, mask = render_camera()

    image = Image.fromarray(sim_d).convert('L')
    image.save('image.jpg')

    high_level_command = ['Please clean the table']

    if UNSEEN_VERB == True:
        replaced_verb = random.choice(['empty', 'clear'])
        high_level_command = ['Please ' + replaced_verb + ' the table']
    if UNSEEN_NOUN == True:
        replaced_noun = random.choice(['tableland', 'stage'])
        high_level_command = ['Please clean the ' + replaced_noun]
    if UNSEEN_VN == True:
        replaced_noun = random.choice(['tableland', 'stage'])
        replaced_verb = random.choice(['empty', 'clear'])
        high_level_command = ['Please ' + replaced_verb + ' the ' + replaced_noun]

    low_level_command = ['Grasp an object',
                         'Place the object into the bin',
                         'Done']

    sentence = high_level_command[0]
    if case_id == 9:
        label = low_level_command[0]
    elif case_id == 10:
        label = low_level_command[1]
    elif case_id == 11:
        label = low_level_command[2]
    return sentence, tactile, label


def env_round(env):
    case_id = env.case_id
    object_name = env.object_name
    tactile = env.tactile
    sim_d, sim_float, mask = render_camera()

    image = Image.fromarray(sim_d).convert('L')
    image.save('image.jpg')

    high_level_command = ['Please put all the round objects from the table to the box']

    if UNSEEN_VERB == True:
        replaced_verb = random.choice(['move', 'place', 'pick'])
        high_level_command = ['Please ' + replaced_verb + ' all the round objects from the table to the box']
    if UNSEEN_NOUN == True:
        replaced_noun = random.choice(['tableland', 'desk'])
        high_level_command = ['Please put all the round objects from the ' + replaced_noun + ' to the box']
    if UNSEEN_VN == True:
        replaced_verb = random.choice(['move', 'place', 'pick'])
        replaced_noun = random.choice(['tableland', 'desk'])
        high_level_command = ['Please ' + replaced_verb + ' all the round objects from the ' + replaced_noun + ' to the box']

    low_level_command = ['Grasp an round object',
                         'Place the round object into the bin',
                         'Done']

    sentence = high_level_command[0]
    if case_id == 30:
        label = low_level_command[0]
    elif case_id == 31:
        label = low_level_command[1]
    elif case_id == 32:
        label = low_level_command[2]
    return sentence, tactile, label


def env_multi1(env):
    case_id = env.case_id
    object_name = env.object_name
    tactile = env.tactile
    sim_d, sim_float, mask = render_camera()

    image = Image.fromarray(sim_d).convert('L')
    image.save('image.jpg')

    high_level_command = ['Please close the drawer and then grasp the apple']

    if UNSEEN_VERB == True:
        if random.random() > 0.5:
            replaced_verb1 = random.choice(['close'])
            replaced_verb2 = random.choice(['grip', 'catch'])
        else:
            replaced_verb1 = random.choice(['shut'])
            replaced_verb2 = random.choice(['grasp'])
        high_level_command = ['Please ' + replaced_verb1 + ' the drawer and then ' + replaced_verb2 + ' the apple']
    if UNSEEN_NOUN == True:
        replaced_noun = random.choice(['fruit', 'object', 'thing'])
        high_level_command = ['Please close the drawer and then grasp the ' + replaced_noun]
    if UNSEEN_VN == True:
        if random.random() > 0.5:
            replaced_verb1 = random.choice(['close'])
            replaced_verb2 = random.choice(['grip', 'catch'])
        else:
            replaced_verb1 = random.choice(['shut'])
            replaced_verb2 = random.choice(['grasp'])
        replaced_noun = random.choice(['fruit'])
        high_level_command = ['Please ' + replaced_verb1 + ' the drawer and then ' + replaced_verb2 + ' the ' + replaced_noun]

    low_level_command = ['Close the drawer',
                         'Grasp the apple',
                         'Done']

    sentence = high_level_command[0]
    if case_id == 12:
        label = low_level_command[0]
    elif case_id == 13:
        label = low_level_command[1]
    elif case_id == 14:
        label = low_level_command[2]
    return sentence, tactile, label


def env_multi2(env):
    case_id = env.case_id
    object_name = env.object_name
    tactile = env.tactile
    sim_d, sim_float, mask = render_camera()

    image = Image.fromarray(sim_d).convert('L')
    image.save('image.jpg')

    high_level_command = ['Please close the drawer and then grasp the knife']

    if UNSEEN_VERB == True:
        if random.random() > 0.5:
            replaced_verb1 = random.choice(['close'])
            replaced_verb2 = random.choice(['grip', 'catch'])
        else:
            replaced_verb1 = random.choice(['shut'])
            replaced_verb2 = random.choice(['grasp'])
        high_level_command = ['Please ' + replaced_verb1 + ' the drawer and then ' + replaced_verb2 + ' the knife']
    if UNSEEN_NOUN == True:
        replaced_noun = random.choice(['blade'])
        high_level_command = ['Please close the drawer and then grasp the ' + replaced_noun]
    if UNSEEN_VN == True:
        if random.random() > 0.5:
            replaced_verb1 = random.choice(['close'])
            replaced_verb2 = random.choice(['grip', 'catch'])
        else:
            replaced_verb1 = random.choice(['shut'])
            replaced_verb2 = random.choice(['grasp'])
        replaced_noun = random.choice(['blade'])
        high_level_command = ['Please ' + replaced_verb1 + ' the drawer and then ' + replaced_verb2 + ' the ' + replaced_noun]

    low_level_command = ['Close the drawer',
                         'Grasp the knife',
                         'Done']

    sentence = high_level_command[0]
    if case_id == 15:
        label = low_level_command[0]
    elif case_id == 16:
        label = low_level_command[1]
    elif case_id == 17:
        label = low_level_command[2]
    return sentence, tactile, label


def env_multi3(env):
    case_id = env.case_id
    object_name = env.object_name
    tactile = env.tactile
    sim_d, sim_float, mask = render_camera()

    image = Image.fromarray(sim_d).convert('L')
    image.save('image.jpg')

    high_level_command = ['Please put the cosmetic into the drawer and then clean the table.']

    if UNSEEN_VERB == True:
        if random.random() > 0.5:
            replaced_verb1 = random.choice(['put'])
            replaced_verb2 = random.choice(['empty', 'clear'])
        else:
            replaced_verb1 = random.choice(['place'])
            replaced_verb2 = random.choice(['clean'])

        high_level_command = ['Please ' + replaced_verb1 + ' the cosmetic into the drawer and then ' + replaced_verb2 + ' the table.']
    if UNSEEN_NOUN == True:
        replaced_noun = random.choice(['cosmetic'])
        high_level_command = ['Please put the ' + replaced_noun + ' into the drawer and then clean the table.']
    if UNSEEN_VN == True:
        if random.random() > 0.5:
            replaced_verb1 = random.choice(['put'])
            replaced_verb2 = random.choice(['empty', 'clear'])
        else:
            replaced_verb1 = random.choice(['place'])
            replaced_verb2 = random.choice(['clean'])
        replaced_noun = random.choice(['cosmetic'])
        high_level_command = ['Please ' + replaced_verb1 + ' the ' + replaced_noun + ' into the drawer and then ' + replaced_verb2 + ' the table.']

    low_level_command = ['Open the drawer',
                         'Grasp the cosmetic',
                         'Put the cosmetic into drawer',
                         'Close the drawer',
                         'Grasp an object',
                         'Put the object into the bin',
                         'Done']

    sentence = high_level_command[0]
    if case_id == 18:
        label = low_level_command[0]
    elif case_id == 19:
        label = low_level_command[1]
    elif case_id == 20:
        label = low_level_command[2]
    elif case_id == 21:
        label = low_level_command[3]
    elif case_id == 22:
        label = low_level_command[4]
    elif case_id == 23:
        label = low_level_command[5]
    elif case_id == 24:
        label = low_level_command[6]
    return sentence, tactile, label


def env_multi4(env):
    case_id = env.case_id
    object_name = env.object_name
    tactile = env.tactile
    sim_d, sim_float, mask = render_camera()

    image = Image.fromarray(sim_d).convert('L')
    image.save('image.jpg')

    high_level_command = ['Please clean the table and then cut the ' + object_name]

    if UNSEEN_VERB == True:
        if random.random() > 0.5:
            replaced_verb1 = random.choice(['empty', 'clear'])
            replaced_verb2 = random.choice(['cut']) 
        else:
            replaced_verb1 = random.choice(['clean'])
            replaced_verb2 = random.choice(['chop', 'slice'])
        high_level_command = ['Please ' + replaced_verb1 + ' the table and then ' + replaced_verb2 + ' the ' + object_name]
    if UNSEEN_NOUN == True:
        if object_name == 'apple':
            object_name = random.choice(['fruit'])
        if object_name == 'banana':
            object_name = random.choice(['fruit'])
        high_level_command = ['Please clean the table and then cut the ' + object_name]
    if UNSEEN_VN == True:
        if random.random() > 0.5:
            replaced_verb1 = random.choice(['empty', 'clear'])
            replaced_verb2 = random.choice(['cut']) 
        else:
            replaced_verb1 = random.choice(['clean'])
            replaced_verb2 = random.choice(['chop', 'slice'])
        if object_name == 'apple':
            object_name = random.choice(['fruit'])
        if object_name == 'banana':
            object_name = random.choice(['fruit'])
        high_level_command = ['Please ' + replaced_verb1 + ' the table and then ' + replaced_verb2 + ' the ' + object_name]

    low_level_command = ['Grasp an object',
                         'Put the object into the bin',
                         'Grasp the knife',
                         'Cut the ' + object_name,
                         'Done']

    sentence = high_level_command[0]
    if case_id == 25:
        label = low_level_command[0]
    elif case_id == 26:
        label = low_level_command[1]
    elif case_id == 27:
        label = low_level_command[2]
    elif case_id == 28:
        label = low_level_command[3]
    elif case_id == 29:
        label = low_level_command[4]
    return sentence, tactile, label
