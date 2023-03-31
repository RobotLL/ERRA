import numpy as np
import random
import pybullet
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
import math
from util import render_camera
from util import generate_roll_obj, generate_td_obj


def build_pick_sth_into_drawer_env(obj_random=True, rl=False, saycan=False, case_id=None):

    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=50)

    # p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetDebugVisualizerCamera(
        cameraDistance=.2, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0, 0, 0.2])
    # Define world
    p.setGravity(0, 0, -9.8)
    p.loadURDF('plane.urdf', [0, 0, 0], useFixedBase=True)

    # random case:
    # 1: start state, drawer closed, has object
    # 2: start state, drawer open, has object
    # 3: middle state, drawer opened, object is grasped
    # 4: final state, drawer opened, object in drawer
    if rl is True:
        case_list = [1, 2, 3, 4]
    else:
        case_list = [1, 2, 3, 4, 5]

    if saycan is True:
        case_list = [1, 2]

    if case_id is None:
        case_id = random.choice(case_list)
    else:
        case_id = case_id

    object_name = None
    knife_id = None
    obj_id = None
    obj_id_list = []
    tactile = None

    object_list = ['can', 'cosmetic', 'clip']
    object_name = random.choice(object_list)

    # case_id = 5
    if case_id == 1:  # Low level command:
        # tactile infomation
        tactile = 0
        # build drawer
        if obj_random is False:
            drawer_x = 0.1
            drawer_y = -0.1
            drawer_z = 0
        else:
            drawer_x = 0.1 + (random.random()*2-1)*0.005
            drawer_y = -0.1 + (random.random()*2-1)*0.005
            drawer_z = 0
            scale = random.uniform(0.85, 1)

        drawer_down = p.loadURDF('./model/task_drawer/obj/drawer/drawer_down.urdf', [
                                 drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=False)
        drawer_up = p.loadURDF('./model/task_drawer/obj/drawer/drawer_up.urdf', [
                               drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=True)

        # load object
        obj_path = './model/task_drawer/obj/' + object_name + '/' + object_name + '.urdf'
        if obj_random is False:
            obj_x = -0.1
            obj_y = 0
            obj_z = 0.01
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0
        else:
            obj_x = -0.1 + (random.random()*2-1)*0.025
            obj_y = 0 + (random.random()*2-1)*0.025
            obj_z = 0.01
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0 + (random.random()*2-1)*3.14/4
            scale = random.uniform(0.85, 1.1)

        obj_id = p.loadURDF(obj_path, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
            [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False, globalScaling=scale)
        obj_id_list.append(obj_id)

    elif case_id == 2:  # Low level command:
        # tactile infomation
        tactile = 0
        # build drawer
        if obj_random is False:
            drawer_x = 0.1
            drawer_y = -0.1
            drawer_z = 0
        else:
            drawer_x = 0.1 + (random.random()*2-1)*0.005
            drawer_y = -0.1 + (random.random()*2-1)*0.005
            drawer_z = 0
            scale = random.uniform(0.85, 1)

        drawer_down = p.loadURDF('./model/task_drawer/obj/drawer/drawer_down.urdf', [
                                 drawer_x, drawer_y+0.1, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=False)
        drawer_up = p.loadURDF('./model/task_drawer/obj/drawer/drawer_up.urdf', [
                               drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=True)

        # load object
        obj_path = './model/task_drawer/obj/' + object_name + '/' + object_name + '.urdf'
        if obj_random is False:
            obj_x = -0.1
            obj_y = 0
            obj_z = 0.01
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0
        else:
            obj_x = -0.1 + (random.random()*2-1)*0.025
            obj_y = 0 + (random.random()*2-1)*0.025
            obj_z = 0.01
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0 + (random.random()*2-1)*3.14/4
            scale = random.uniform(0.85, 1.1)

        obj_id = p.loadURDF(obj_path, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
            [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False, globalScaling=scale)
        obj_id_list.append(obj_id)

    elif case_id == 3:  # Low level command:
        # tactile infomation
        tactile = 1
        # build drawer
        if obj_random is False:
            drawer_x = 0.1
            drawer_y = -0.1
            drawer_z = 0
        else:
            drawer_x = 0.1 + (random.random()*2-1)*0.005
            drawer_y = -0.1 + (random.random()*2-1)*0.005
            drawer_z = 0
            scale = random.uniform(0.85, 1)

        drawer_down = p.loadURDF('./model/task_drawer/obj/drawer/drawer_down.urdf', [
                                 drawer_x, drawer_y+0.1, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=False)
        drawer_up = p.loadURDF('./model/task_drawer/obj/drawer/drawer_up.urdf', [
                               drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=True)

    elif case_id == 4:  # Low level command:
        # tactile infomation
        tactile = 0
        # build drawer
        if obj_random is False:
            drawer_x = 0.1
            drawer_y = -0.1
            drawer_z = 0
            scale = 1
        else:
            drawer_x = 0.1 + (random.random()*2-1)*0.005
            drawer_y = -0.1 + (random.random()*2-1)*0.005
            drawer_z = 0
            scale = random.uniform(0.85, 1.1)

        drawer_down = p.loadURDF('./model/task_drawer/obj/drawer/drawer_down.urdf', [
                                 drawer_x, drawer_y+0.1, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=False)
        drawer_up = p.loadURDF('./model/task_drawer/obj/drawer/drawer_up.urdf', [
                               drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=True)

        # load object
        obj_path = './model/task_drawer/obj/' + object_name + '/' + object_name + '.urdf'
        if obj_random is False:
            obj_x = 0.1
            obj_y = 0.05
            obj_z = 0.006
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0
        else:
            obj_x = 0.1 + (random.random()*2-1)*0.02
            obj_y = 0.05 + (random.random()*2-1)*0.005
            obj_z = 0.006
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0 + (random.random()*2-1)*3.14/4
            scale = random.uniform(0.85, 1.1)

        obj_id = p.loadURDF(obj_path, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
            [obj_roll, obj_pitch, obj_yaw]), useFixedBase=True, globalScaling=scale)
        obj_id_list.append(obj_id)

    elif case_id == 5:  # Low level command: Done
        # tactile infomation
        tactile = 0
        # build drawer
        if obj_random is False:
            drawer_x = 0.1
            drawer_y = -0.1
            drawer_z = 0
        else:
            drawer_x = 0.1 + (random.random()*2-1)*0.005
            drawer_y = -0.1 + (random.random()*2-1)*0.005
            drawer_z = 0
            scale = random.uniform(0.85, 1)

        drawer_down = p.loadURDF('./model/task_drawer/obj/drawer/drawer_down.urdf', [
                                 drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=False)
        drawer_up = p.loadURDF('./model/task_drawer/obj/drawer/drawer_up.urdf', [
                               drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=True)

    for _ in range(240):
        p.stepSimulation()

    return case_id, object_name, obj_id_list, tactile, knife_id

# p.connect(p.DIRECT)
# # p.connect(p.GUI)
# build_pick_sth_into_drawer_env(obj_random=True, rl=False)
# sim_d, sim_float, mask = render_camera()
# plt.imshow(sim_d)
# %%


def build_cut_sth_env(obj_random=True, rl=False, saycan=False, case_id=None):

    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=50)

    # p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetDebugVisualizerCamera(
        cameraDistance=.2, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0, 0, 0.2])
    # Define world
    p.setGravity(0, 0, -9.8)
    p.loadURDF('plane.urdf', [0, 0, 0], useFixedBase=True)

    # random case:
    # 1: start state, knife not grasp, has object
    # 2: middle state, knife in hand, has object
    # 3: final state, cut object, done, objrct in two parts, not in rl

    if rl is True:
        case_list = [6, 7]
    else:
        case_list = [6, 7, 8]

    if saycan is True:
        case_list = [6]

    if case_id is None:
        case_id = random.choice(case_list)
    else:
        case_id = case_id
    object_list = ['apple', 'banana', 'eggplant']
    # object_list = ['apple']
    object_name = random.choice(object_list)
    knife_id = None
    obj_id = None
    obj_id_list = []
    tactile = None

    if case_id == 6:  # Low level command: 'grasp the knife'

        # build knife
        if obj_random is False:
            knife_x = -0.075
            knife_y = 0
            knife_z = 0.01
        else:
            knife_x = -0.075 + (random.random()*2-1)*0.01
            knife_y = 0 + (random.random()*2-1)*0.01
            knife_z = 0.01
            scale = random.uniform(0.85, 1.1)
        # load knife
        knife_id = p.loadURDF('./model/task_cut/obj/knife/knife.urdf', [
                              knife_x, knife_y, knife_z], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=False, globalScaling=scale)

        # load object
        obj_path = './model/task_cut/obj/' + object_name + '/' + object_name + '.urdf'
        if obj_random is False:
            obj_x = 0.1
            obj_y = 0
            obj_z = 0.01
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0
        else:
            obj_x = 0.1 + (random.random()*2-1)*0.025
            obj_y = 0 + (random.random()*2-1)*0.025
            obj_z = 0.01
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0 + (random.random()*2-1)*3.14/4
            scale = random.uniform(0.85, 1.1)

        obj_id = p.loadURDF(obj_path, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
            [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False, globalScaling=scale)
        obj_id_list.append(obj_id)
        tactile = 0

    elif case_id == 7:  # Low level command: 'cut the object_name'
        # load object
        obj_path = './model/task_cut/obj/' + object_name + '/' + object_name + '.urdf'
        if obj_random is False:
            obj_x = 0.1
            obj_y = 0
            obj_z = 0.01
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0
        else:
            obj_x = 0.1 + (random.random()*2-1)*0.025
            obj_y = 0 + (random.random()*2-1)*0.025
            obj_z = 0.01
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0 + (random.random()*2-1)*3.14/4
            scale = random.uniform(0.85, 1.1)
        obj_id = p.loadURDF(obj_path, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
            [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False,  globalScaling=scale)
        obj_id_list.append(obj_id)
        tactile = 1

    elif case_id == 8:  # Low level command: 'Done'
        # load object
        obj_path_left = './model/task_cut/obj/' + object_name + '/' + object_name + '_left.urdf'
        obj_path_right = './model/task_cut/obj/' + object_name + '/' + object_name + '_right.urdf'
        if obj_random is False:
            obj_x = 0.1
            obj_y = 0
            obj_z = 0.05
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0
        else:
            obj_x = 0.1 + (random.random()*2-1)*0.025
            obj_y = 0 + (random.random()*2-1)*0.025
            obj_z = 0.035 + (random.random()*2-1)*0.025
            obj_roll = 0 + (random.random()*2-1)*1.57
            obj_pitch = 0 + (random.random()*2-1)*3.14/20
            obj_yaw = 0 + (random.random()*2-1)*3.14/4
            scale = random.uniform(0.85, 1.1)

        obj_l = p.loadURDF(obj_path_left, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
            [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False,  globalScaling=scale)
        obj_r = p.loadURDF(obj_path_right, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
            [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False,  globalScaling=scale)
        tactile = 0

    for _ in range(240):
        p.stepSimulation()

    return case_id, object_name, obj_id_list,  tactile, knife_id

# p.connect(p.DIRECT)
# # # p.connect(p.GUI)
# build_cut_sth_env(obj_random=True, rl=True)
# sim_d, sim_float, mask = render_camera(rl=True)
# plt.imshow(sim_d)
# %%


def build_clean_table_env(obj_random=True, rl=False, saycan=False, case_id=None):

    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=50)

    # p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetDebugVisualizerCamera(
        cameraDistance=.2, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0, 0, 0.2])
    # Define world
    p.setGravity(0, 0, -9.8)
    p.loadURDF('plane.urdf', [0, 0, 0], useFixedBase=True)

    # random case:
    # 1: start state, random objects drop, has objects, tactile = 0 , need to grasp one
    # 2: middle state, has objects, tactile = 1, need to put the object into trash can
    # 3: final state, no object in the image and tactile = 0
    if rl is True:
        case_list = [9, 10]
        # case_list = [9]
    else:
        case_list = [9, 10, 11]

    if saycan is True:
        case_list = [9]

    if case_id is None:
        case_id = random.choice(case_list)
    else:
        case_id = case_id
    object_name = None
    knife_id = None
    obj_id = None
    obj_id_list = []
    tactile = None

    object_drop_position_lists = [[-0.05, 0], [-0.125, -0.05], [-0.125, 0.05]]
    object_list = ['bottle', 'can', 'chips']

    if case_id == 9:  # Low level command:
        tactile = 0
        drop_x_objects = random.randint(1, 3)
        object_names = random.choices(object_list, k=drop_x_objects)
        object_drop_positions = random.sample(object_drop_position_lists, k=drop_x_objects)
        for i in range(drop_x_objects):
            # load object
            obj_path = './model/task_clean/obj/' + object_names[i] + '/' + object_names[i] + '.urdf'
            if obj_random is False:
                print('something wrong')
                pass
            else:
                obj_x = object_drop_positions[i][0] + (random.random()*2-1)*0.025
                obj_y = object_drop_positions[i][1] + (random.random()*2-1)*0.025
                obj_z = 0.01
                obj_roll = 0
                obj_pitch = 0
                obj_yaw = 0 + (random.random()*2-1)*3.14/4
                scale = random.uniform(0.85, 1.1)

            obj_id = p.loadURDF(obj_path, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
                [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False, globalScaling=scale)
            obj_id_list.append(obj_id)
            for _ in range(40):
                p.stepSimulation()

    elif case_id == 10:  # Low level command:
        tactile = 1
        drop_x_objects = random.randint(0, 2)
        if drop_x_objects != 0:
            object_names = random.choices(object_list, k=drop_x_objects)
            object_drop_positions = random.sample(object_drop_position_lists, k=drop_x_objects)
            for i in range(drop_x_objects):
                # load object
                obj_path = './model/task_clean/obj/' + \
                    object_names[i] + '/' + object_names[i] + '.urdf'
                if obj_random is False:
                    print('something wrong')
                    pass
                else:
                    obj_x = object_drop_positions[i][0] + (random.random()*2-1)*0.025
                    obj_y = object_drop_positions[i][1] + (random.random()*2-1)*0.025
                    obj_z = 0.01
                    obj_roll = 0
                    obj_pitch = 0
                    obj_yaw = 0 + (random.random()*2-1)*3.14/4
                    scale = random.uniform(0.85, 1.1)

                obj_id = p.loadURDF(obj_path, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
                    [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False, globalScaling=scale)
                obj_id_list.append(obj_id)
                for _ in range(40):
                    p.stepSimulation()

    elif case_id == 11:
        tactile = 0
        for _ in range(10):
            p.stepSimulation()

    return case_id, object_name, obj_id_list, tactile, knife_id


# p.connect(p.GUI)
# p.connect(p.DIRECT)
# build_clean_table_env(obj_random=True,rl=False)
# sim_d, sim_float, mask = render_camera()
# plt.imshow(sim_d)
# %%
# close_drawer_grasp_apple_env
def multi1(obj_random=True, rl=False, saycan=False, case_id=None):

    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=50)

    # p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetDebugVisualizerCamera(
        cameraDistance=.2, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0, 0, 0.2])
    # Define world
    p.setGravity(0, 0, -9.8)
    p.loadURDF('plane.urdf', [0, 0, 0], useFixedBase=True)

    # random case:
    # 1: drawer open, apple on the table.
    # 2: drawer closed, apple on the table.
    # 3: drawer closed, apple not at the table. Done
    if rl is True:
        case_list = [12, 13]
    else:
        case_list = [12, 13, 14]

    if saycan is True:
        case_list = [12]

    if case_id is None:
        case_id = random.choice(case_list)
    else:
        case_id = case_id
    object_name = None
    knife_id = None
    obj_id = None
    obj_id_list = []
    tactile = None

    object_drop_position_lists = [[-0.05, 0], [-0.125, -0.05], [-0.125, 0.05]]
    object_list = ['banana', 'can']

    if case_id == 12:  # Low level command:
        # tactile infomation
        tactile = 0
        # build drawer
        if obj_random is False:
            drawer_x = 0.1
            drawer_y = -0.1
            drawer_z = 0
        else:
            drawer_x = 0.1 + (random.random()*2-1)*0.005
            drawer_y = -0.1 + (random.random()*2-1)*0.005
            drawer_z = 0
            scale = random.uniform(0.85, 1)

        drawer_down = p.loadURDF('./model/obj/drawer/drawer_down.urdf', [
                                 drawer_x, drawer_y+0.1, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=False)
        drawer_up = p.loadURDF('./model/obj/drawer/drawer_up.urdf', [
                               drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=True)

        # load object
        drop_x_objects = random.randint(1, 2)
        if drop_x_objects != 0:
            object_names = random.choices(object_list, k=drop_x_objects)
            object_names.append('apple')
            object_drop_positions = random.sample(object_drop_position_lists, k=drop_x_objects+1)
            for i in range(drop_x_objects+1):
                # load object
                obj_path = './model/obj/' + \
                    object_names[i] + '/' + object_names[i] + '.urdf'
                if obj_random is False:
                    print('something wrong')
                    pass
                else:
                    obj_x = object_drop_positions[i][0] + (random.random()*2-1)*0.025
                    obj_y = object_drop_positions[i][1] + (random.random()*2-1)*0.025
                    obj_z = 0.01
                    obj_roll = 0
                    obj_pitch = 0
                    obj_yaw = 0 + (random.random()*2-1)*3.14/4
                    scale = random.uniform(0.85, 1.1)

                obj_id = p.loadURDF(obj_path, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
                    [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False, globalScaling=scale)
                if object_names[i] == 'apple':
                    obj_id_list.append(obj_id)
                for _ in range(40):
                    p.stepSimulation()

    elif case_id == 13:
        # tactile infomation
        tactile = 0
        # build drawer
        if obj_random is False:
            drawer_x = 0.1
            drawer_y = -0.1
            drawer_z = 0
        else:
            drawer_x = 0.1 + (random.random()*2-1)*0.005
            drawer_y = -0.1 + (random.random()*2-1)*0.005
            drawer_z = 0
            scale = random.uniform(0.85, 1)

        drawer_down = p.loadURDF('./model/obj/drawer/drawer_down.urdf', [
                                 drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=False)
        drawer_up = p.loadURDF('./model/obj/drawer/drawer_up.urdf', [
                               drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=True)

        # load object
        drop_x_objects = random.randint(1, 2)
        if drop_x_objects != 0:
            object_names = random.choices(object_list, k=drop_x_objects)
            object_names.append('apple')
            object_drop_positions = random.sample(object_drop_position_lists, k=drop_x_objects+1)
            for i in range(drop_x_objects+1):
                # load object
                obj_path = './model/obj/' + \
                    object_names[i] + '/' + object_names[i] + '.urdf'
                if obj_random is False:
                    print('something wrong')
                    pass
                else:
                    obj_x = object_drop_positions[i][0] + (random.random()*2-1)*0.025
                    obj_y = object_drop_positions[i][1] + (random.random()*2-1)*0.025
                    obj_z = 0.01
                    obj_roll = 0
                    obj_pitch = 0
                    obj_yaw = 0 + (random.random()*2-1)*3.14/4
                    scale = random.uniform(0.85, 1.1)

                obj_id = p.loadURDF(obj_path, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
                    [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False, globalScaling=scale)
                if object_names[i] == 'apple':
                    obj_id_list.append(obj_id)
                for _ in range(40):
                    p.stepSimulation()

    elif case_id == 14:  # Low level command: Done
        # tactile infomation
        tactile = 1
        # build drawer
        if obj_random is False:
            drawer_x = 0.1
            drawer_y = -0.1
            drawer_z = 0
        else:
            drawer_x = 0.1 + (random.random()*2-1)*0.005
            drawer_y = -0.1 + (random.random()*2-1)*0.005
            drawer_z = 0
            scale = random.uniform(0.85, 1)

        drawer_down = p.loadURDF('./model/obj/drawer/drawer_down.urdf', [
                                 drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=False)
        drawer_up = p.loadURDF('./model/obj/drawer/drawer_up.urdf', [
                               drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=True)

        # load object
        drop_x_objects = random.randint(1, 2)
        if drop_x_objects != 0:
            object_names = random.choices(object_list, k=drop_x_objects)
            object_drop_positions = random.sample(object_drop_position_lists, k=drop_x_objects)
            for i in range(drop_x_objects):
                # load object
                obj_path = './model/obj/' + \
                    object_names[i] + '/' + object_names[i] + '.urdf'
                if obj_random is False:
                    print('something wrong')
                    pass
                else:
                    obj_x = object_drop_positions[i][0] + (random.random()*2-1)*0.025
                    obj_y = object_drop_positions[i][1] + (random.random()*2-1)*0.025
                    obj_z = 0.01
                    obj_roll = 0
                    obj_pitch = 0
                    obj_yaw = 0 + (random.random()*2-1)*3.14/4
                    scale = random.uniform(0.85, 1.1)

                obj_id = p.loadURDF(obj_path, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
                    [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False, globalScaling=scale)
                obj_id_list.append(obj_id)
                for _ in range(40):
                    p.stepSimulation()
    return case_id, object_name, obj_id_list, tactile, knife_id


# p.connect(p.DIRECT)
# p.connect(p.GUI)
# multi1(obj_random=True, rl=False)
# sim_d, sim_float, mask = render_camera()
# plt.imshow(sim_d)
# %%
# close_drawer_grasp_knife_env
def multi2(obj_random=True, rl=False, saycan=False, case_id=None):

    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=50)

    # p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetDebugVisualizerCamera(
        cameraDistance=.2, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0, 0, 0.2])
    # Define world
    p.setGravity(0, 0, -9.8)
    p.loadURDF('plane.urdf', [0, 0, 0], useFixedBase=True)

    # random case:
    # 1: drawer open, knife on the table.
    # 2: drawer closed, knife on the table.
    # 3: drawer closed, knife not at the table. Done
    if rl is True:
        case_list = [15, 16]
    else:
        case_list = [15, 16, 17]

    if saycan is True:
        case_list = [15]

    if case_id == None:
        case_id = random.choice(case_list)
    else:
        case_id = case_id
    object_name = None
    knife_id = None
    obj_id = None
    obj_id_list = []
    tactile = None

    if case_id == 15:  # Low level command:
        # tactile infomation
        tactile = 0
        # build drawer
        if obj_random is False:
            drawer_x = 0.1
            drawer_y = -0.1
            drawer_z = 0
        else:
            drawer_x = 0.1 + (random.random()*2-1)*0.005
            drawer_y = -0.1 + (random.random()*2-1)*0.005
            drawer_z = 0
            scale = random.uniform(0.85, 1)

        drawer_down = p.loadURDF('./model/obj/drawer/drawer_down.urdf', [
                                 drawer_x, drawer_y+0.1, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=False)
        drawer_up = p.loadURDF('./model/obj/drawer/drawer_up.urdf', [
                               drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=True)

        # build knife
        if obj_random is False:
            knife_x = -0.075
            knife_y = 0
            knife_z = 0.01
        else:
            knife_x = -0.075 + (random.random()*2-1)*0.01
            knife_y = 0 + (random.random()*2-1)*0.01
            knife_z = 0.01
            scale = random.uniform(0.85, 1.1)
        # load knife
        knife_id = p.loadURDF('./model/task_cut/obj/knife/knife.urdf', [
                              knife_x, knife_y, knife_z], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=False, globalScaling=scale)
        for _ in range(10):
            p.stepSimulation()

    elif case_id == 16:
        # tactile infomation
        tactile = 0
        # build drawer
        if obj_random is False:
            drawer_x = 0.1
            drawer_y = -0.1
            drawer_z = 0
        else:
            drawer_x = 0.1 + (random.random()*2-1)*0.005
            drawer_y = -0.1 + (random.random()*2-1)*0.005
            drawer_z = 0
            scale = random.uniform(0.85, 1)

        drawer_down = p.loadURDF('./model/obj/drawer/drawer_down.urdf', [
                                 drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=False)
        drawer_up = p.loadURDF('./model/obj/drawer/drawer_up.urdf', [
                               drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=True)

        for _ in range(10):
            p.stepSimulation()

        # build knife
        if obj_random is False:
            knife_x = -0.075
            knife_y = 0
            knife_z = 0.01
        else:
            knife_x = -0.075 + (random.random()*2-1)*0.01
            knife_y = 0 + (random.random()*2-1)*0.01
            knife_z = 0.01
            scale = random.uniform(0.85, 1.1)
        # load knife
        knife_id = p.loadURDF('./model/task_cut/obj/knife/knife.urdf', [
                              knife_x, knife_y, knife_z], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=False, globalScaling=scale)
        for _ in range(10):
            p.stepSimulation()

    elif case_id == 17:  # Low level command: Done
        # tactile infomation
        tactile = 1
        # build drawer
        if obj_random is False:
            drawer_x = 0.1
            drawer_y = -0.1
            drawer_z = 0
        else:
            drawer_x = 0.1 + (random.random()*2-1)*0.005
            drawer_y = -0.1 + (random.random()*2-1)*0.005
            drawer_z = 0
            scale = random.uniform(0.85, 1)

        drawer_down = p.loadURDF('./model/obj/drawer/drawer_down.urdf', [
                                 drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=False)
        drawer_up = p.loadURDF('./model/obj/drawer/drawer_up.urdf', [
                               drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=True)

        for _ in range(240):
            p.stepSimulation()
    return case_id, object_name, obj_id_list, tactile, knife_id


# p.connect(p.DIRECT)
# p.connect(p.GUI)
# multi2(obj_random=True, rl=False)
# sim_d, sim_float, mask = render_camera()
# plt.imshow(sim_d)
# %% Multi-task3: put the cosmetic into the drawer and clean the table.
def multi3(obj_random=True, rl=False, saycan=False, case_id=None):

    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=50)

    # p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetDebugVisualizerCamera(
        cameraDistance=.2, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0, 0, 0.2])
    # Define world
    p.setGravity(0, 0, -9.8)
    p.loadURDF('plane.urdf', [0, 0, 0], useFixedBase=True)

    # random case:
    # random case:
    # 18: drawer closed, has object
    # 19: drawer open, has object
    # 20: drawer opened, object is grasped
    # 21: drawer opened, object in drawer
    # 22: has objects for bin, tactile = 0 , need to grasp one
    # 23: has objects, tactile = 1, need to put the object into trash can
    # 24: no object in the image and tactile = 0
    if rl is True:
        case_list = [18, 19, 20, 21, 22, 23]
    else:
        case_list = [18, 19, 20, 21, 22, 23, 24]

    if saycan is True:
        case_list = [18]

    if case_id == None:
        case_id = random.choice(case_list)
    else:
        case_id = case_id
    object_name = None
    knife_id = None
    obj_id = None
    obj_id_list = []
    tactile = None
    clean_task_object_number = None

    object_list = ['cosmetic']
    object_name = random.choice(object_list)

    if case_id == 18:  # Low level command:
        # tactile infomation
        tactile = 0
        # build drawer
        if obj_random is False:
            drawer_x = -0.025
            drawer_y = -0.1
            drawer_z = 0
        else:
            drawer_x = -0.025 + (random.random()*2-1)*0.005
            drawer_y = -0.1 + (random.random()*2-1)*0.005
            drawer_z = 0
            scale = random.uniform(0.85, 1)

        drawer_down = p.loadURDF('./model/task_drawer/obj/drawer/drawer_down.urdf', [
                                 drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=False)
        drawer_up = p.loadURDF('./model/task_drawer/obj/drawer/drawer_up.urdf', [
                               drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=True)

        # load object
        obj_path = './model/task_drawer/obj/' + object_name + '/' + object_name + '.urdf'
        if obj_random is False:
            obj_x = -0.175
            obj_y = 0
            obj_z = 0.01
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0
        else:
            obj_x = -0.175 - random.random()*0.02
            obj_y = 0 + (random.random()*2-1)*0.075
            obj_z = 0.01
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0 + (random.random()*2-1)*3.14/4
            scale = random.uniform(0.8, 1)

        obj_id = p.loadURDF(obj_path, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
            [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False, globalScaling=scale)
        obj_id_list.append(obj_id)

        for _ in range(40):
            p.stepSimulation()

        object_drop_position_lists = [[0.15, 0], [0.15, -0.1], [0.15, 0.1]]
        object_list = ['bottle', 'can', 'chips']
        drop_x_objects = random.randint(1, 3)
        object_names = random.choices(object_list, k=drop_x_objects)
        clean_task_object_number = drop_x_objects
        object_drop_positions = random.sample(object_drop_position_lists, k=drop_x_objects)
        for i in range(drop_x_objects):
            # load object
            obj_path = './model/task_clean/obj/' + object_names[i] + '/' + object_names[i] + '.urdf'
            if obj_random is False:
                print('something wrong')
                pass
            else:
                obj_x = object_drop_positions[i][0] + (random.random()*2-1)*0.025
                obj_y = object_drop_positions[i][1] + (random.random()*2-1)*0.025
                obj_z = 0.01
                obj_roll = 0
                obj_pitch = 0
                obj_yaw = 0 + (random.random()*2-1)*3.14/4
                scale = random.uniform(0.85, 1.1)
            obj_id = p.loadURDF(obj_path, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
                [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False, globalScaling=scale)
            for _ in range(40):
                p.stepSimulation()

    elif case_id == 19:  # Low level command:
        # tactile infomation
        tactile = 0
        # build drawer
        if obj_random is False:
            drawer_x = -0.025
            drawer_y = -0.1
            drawer_z = 0
        else:
            drawer_x = -0.025 + (random.random()*2-1)*0.005
            drawer_y = -0.1 + (random.random()*2-1)*0.005
            drawer_z = 0
            scale = random.uniform(0.85, 1)

        drawer_down = p.loadURDF('./model/task_drawer/obj/drawer/drawer_down.urdf', [
                                 drawer_x, drawer_y+0.1, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=False)
        drawer_up = p.loadURDF('./model/task_drawer/obj/drawer/drawer_up.urdf', [
                               drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=True)

        # load object
        obj_path = './model/task_drawer/obj/' + object_name + '/' + object_name + '.urdf'
        if obj_random is False:
            obj_x = -0.175
            obj_y = 0
            obj_z = 0.01
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0
        else:
            obj_x = -0.175 - random.random()*0.02
            obj_y = 0 + (random.random()*2-1)*0.075
            obj_z = 0.01
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0 + (random.random()*2-1)*3.14/4
            scale = random.uniform(0.8, 1)

        obj_id = p.loadURDF(obj_path, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
            [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False, globalScaling=scale)
        obj_id_list.append(obj_id)

        for _ in range(40):
            p.stepSimulation()

        object_drop_position_lists = [[0.15, 0], [0.15, -0.1], [0.15, 0.1]]
        object_list = ['bottle', 'can', 'chips']
        drop_x_objects = random.randint(1, 3)
        object_names = random.choices(object_list, k=drop_x_objects)
        object_drop_positions = random.sample(object_drop_position_lists, k=drop_x_objects)
        for i in range(drop_x_objects):
            # load object
            obj_path = './model/task_clean/obj/' + object_names[i] + '/' + object_names[i] + '.urdf'
            if obj_random is False:
                print('something wrong')
                pass
            else:
                obj_x = object_drop_positions[i][0] + (random.random()*2-1)*0.025
                obj_y = object_drop_positions[i][1] + (random.random()*2-1)*0.025
                obj_z = 0.01
                obj_roll = 0
                obj_pitch = 0
                obj_yaw = 0 + (random.random()*2-1)*3.14/4
                scale = random.uniform(0.85, 1.1)
            obj_id = p.loadURDF(obj_path, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
                [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False, globalScaling=scale)
            for _ in range(40):
                p.stepSimulation()

    elif case_id == 20:  # Low level command:
        # tactile infomation
        tactile = 1
        # build drawer
        if obj_random is False:
            drawer_x = -0.025
            drawer_y = -0.1
            drawer_z = 0
        else:
            drawer_x = -0.025 + (random.random()*2-1)*0.005
            drawer_y = -0.1 + (random.random()*2-1)*0.005
            drawer_z = 0
            scale = random.uniform(0.85, 1)

        drawer_down = p.loadURDF('./model/task_drawer/obj/drawer/drawer_down.urdf', [
                                 drawer_x, drawer_y+0.1, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=False)
        drawer_up = p.loadURDF('./model/task_drawer/obj/drawer/drawer_up.urdf', [
                               drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=True)

        for _ in range(40):
            p.stepSimulation()

        object_drop_position_lists = [[0.15, 0], [0.15, -0.1], [0.15, 0.1]]
        object_list = ['bottle', 'can', 'chips']
        drop_x_objects = random.randint(1, 3)
        object_names = random.choices(object_list, k=drop_x_objects)
        object_drop_positions = random.sample(object_drop_position_lists, k=drop_x_objects)
        for i in range(drop_x_objects):
            # load object
            obj_path = './model/task_clean/obj/' + object_names[i] + '/' + object_names[i] + '.urdf'
            if obj_random is False:
                print('something wrong')
                pass
            else:
                obj_x = object_drop_positions[i][0] + (random.random()*2-1)*0.025
                obj_y = object_drop_positions[i][1] + (random.random()*2-1)*0.025
                obj_z = 0.01
                obj_roll = 0
                obj_pitch = 0
                obj_yaw = 0 + (random.random()*2-1)*3.14/4
                scale = random.uniform(0.85, 1.1)
            obj_id = p.loadURDF(obj_path, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
                [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False, globalScaling=scale)
            for _ in range(40):
                p.stepSimulation()

    elif case_id == 21:  # Low level command:
        # tactile infomation
        tactile = 0
        # build drawer
        if obj_random is False:
            drawer_x = -0.025
            drawer_y = -0.1
            drawer_z = 0
        else:
            drawer_x = -0.025 + (random.random()*2-1)*0.005
            drawer_y = -0.1 + (random.random()*2-1)*0.005
            drawer_z = 0
            scale = random.uniform(0.8, 1)

        drawer_down = p.loadURDF('./model/task_drawer/obj/drawer/drawer_down.urdf', [
                                 drawer_x, drawer_y+0.1, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=False)
        drawer_up = p.loadURDF('./model/task_drawer/obj/drawer/drawer_up.urdf', [
                               drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=True)

        # load object
        obj_path = './model/task_drawer/obj/' + object_name + '/' + object_name + '.urdf'
        if obj_random is False:
            obj_x = 0.1
            obj_y = 0.05
            obj_z = 0.006
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0
        else:
            obj_x = -0.025 + (random.random()*2-1)*0.02
            obj_y = 0.05 + (random.random()*2-1)*0.02
            obj_z = 0.006
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0 + (random.random()*2-1)*3.14/4
            scale = random.uniform(0.85, 1)

        obj_id = p.loadURDF(obj_path, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
            [obj_roll, obj_pitch, obj_yaw]), useFixedBase=True, globalScaling=scale)
        obj_id_list.append(obj_id)

        object_drop_position_lists = [[0.15, 0], [0.15, -0.1], [0.15, 0.1]]
        object_list = ['bottle', 'can', 'chips']
        drop_x_objects = random.randint(1, 3)
        object_names = random.choices(object_list, k=drop_x_objects)
        object_drop_positions = random.sample(object_drop_position_lists, k=drop_x_objects)
        for i in range(drop_x_objects):
            # load object
            obj_path = './model/task_clean/obj/' + object_names[i] + '/' + object_names[i] + '.urdf'
            if obj_random is False:
                print('something wrong')
                pass
            else:
                obj_x = object_drop_positions[i][0] + (random.random()*2-1)*0.025
                obj_y = object_drop_positions[i][1] + (random.random()*2-1)*0.025
                obj_z = 0.01
                obj_roll = 0
                obj_pitch = 0
                obj_yaw = 0 + (random.random()*2-1)*3.14/4
                scale = random.uniform(0.85, 1.1)
            obj_id = p.loadURDF(obj_path, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
                [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False, globalScaling=scale)
            for _ in range(40):
                p.stepSimulation()

    elif case_id == 22:  # Low level command: Done
        # tactile infomation
        tactile = 0
        # build drawer
        if obj_random is False:
            drawer_x = -0.025
            drawer_y = -0.1
            drawer_z = 0
        else:
            drawer_x = -0.025 + (random.random()*2-1)*0.005
            drawer_y = -0.1 + (random.random()*2-1)*0.005
            drawer_z = 0
            scale = random.uniform(0.85, 1)

        drawer_down = p.loadURDF('./model/task_drawer/obj/drawer/drawer_down.urdf', [
                                 drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=False)
        drawer_up = p.loadURDF('./model/task_drawer/obj/drawer/drawer_up.urdf', [
                               drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=True)

        object_drop_position_lists = [[0.15, 0], [0.15, -0.1], [0.15, 0.1]]
        object_list = ['bottle', 'can', 'chips']
        drop_x_objects = random.randint(1, 3)
        object_names = random.choices(object_list, k=drop_x_objects)
        object_drop_positions = random.sample(object_drop_position_lists, k=drop_x_objects)
        for i in range(drop_x_objects):
            # load object
            obj_path = './model/task_clean/obj/' + object_names[i] + '/' + object_names[i] + '.urdf'
            if obj_random is False:
                print('something wrong')
                pass
            else:
                obj_x = object_drop_positions[i][0] + (random.random()*2-1)*0.025
                obj_y = object_drop_positions[i][1] + (random.random()*2-1)*0.025
                obj_z = 0.01
                obj_roll = 0
                obj_pitch = 0
                obj_yaw = 0 + (random.random()*2-1)*3.14/4
                scale = random.uniform(0.85, 1.1)
            obj_id = p.loadURDF(obj_path, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
                [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False, globalScaling=scale)
            obj_id_list.append(obj_id)
            for _ in range(40):
                p.stepSimulation()

    elif case_id == 23:  # Low level command: Done
        # tactile infomation
        tactile = 1
        # build drawer
        if obj_random is False:
            drawer_x = -0.025
            drawer_y = -0.1
            drawer_z = 0
        else:
            drawer_x = -0.025 + (random.random()*2-1)*0.005
            drawer_y = -0.1 + (random.random()*2-1)*0.005
            drawer_z = 0
            scale = random.uniform(0.85, 1)

        drawer_down = p.loadURDF('./model/task_drawer/obj/drawer/drawer_down.urdf', [
                                 drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=False)
        drawer_up = p.loadURDF('./model/task_drawer/obj/drawer/drawer_up.urdf', [
                               drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=True)

        object_drop_position_lists = [[0.15, 0], [0.15, -0.1], [0.15, 0.1]]
        object_list = ['bottle', 'can', 'chips']
        drop_x_objects = random.randint(1, 3)
        object_names = random.choices(object_list, k=drop_x_objects)
        object_drop_positions = random.sample(object_drop_position_lists, k=drop_x_objects)
        for i in range(drop_x_objects):
            # load object
            obj_path = './model/task_clean/obj/' + object_names[i] + '/' + object_names[i] + '.urdf'
            if obj_random is False:
                print('something wrong')
                pass
            else:
                obj_x = object_drop_positions[i][0] + (random.random()*2-1)*0.025
                obj_y = object_drop_positions[i][1] + (random.random()*2-1)*0.025
                obj_z = 0.01
                obj_roll = 0
                obj_pitch = 0
                obj_yaw = 0 + (random.random()*2-1)*3.14/4
                scale = random.uniform(0.85, 1.1)
            obj_id = p.loadURDF(obj_path, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
                [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False, globalScaling=scale)
            obj_id_list.append(obj_id)
            for _ in range(40):
                p.stepSimulation()

    elif case_id == 24:  # Low level command: Done
        # tactile infomation
        tactile = 0
        # build drawer
        if obj_random is False:
            drawer_x = -0.025
            drawer_y = -0.1
            drawer_z = 0
        else:
            drawer_x = -0.025 + (random.random()*2-1)*0.005
            drawer_y = -0.1 + (random.random()*2-1)*0.005
            drawer_z = 0
            scale = random.uniform(0.85, 1)

        drawer_down = p.loadURDF('./model/task_drawer/obj/drawer/drawer_down.urdf', [
                                 drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=False)
        drawer_up = p.loadURDF('./model/task_drawer/obj/drawer/drawer_up.urdf', [
                               drawer_x, drawer_y, drawer_z], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=True)

        for _ in range(40):
            p.stepSimulation()

    if rl is False:
        return case_id, object_name, obj_id_list, tactile, knife_id
    elif rl is True:
        return case_id, object_name, obj_id_list, tactile, knife_id


# p.connect(p.DIRECT)
# p.connect(p.GUI)
# multi3(obj_random=True, rl=False)
# sim_d, sim_float, mask = render_camera()
# plt.imshow(sim_d)
# %% Multi-task4: clean the table and cur the banana.
def multi4(obj_random=True, rl=False, saycan=False, case_id=None):

    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=50)

    # p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetDebugVisualizerCamera(
        cameraDistance=.2, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0, 0, 0.2])
    # Define world
    p.setGravity(0, 0, -9.8)
    p.loadURDF('plane.urdf', [0, 0, 0], useFixedBase=True)

    # random case:
    # random case:
    # 25: has objects for bin, tactile = 0 , need to grasp one
    # 26: has objects, tactile = 1, need to put the object into trash can
    # 27: no object in the image and tactile = 0, knife with object
    # 28: knife is grasped, object on the board
    # 29: object into two parts
    if rl is True:
        case_list = [25, 26, 27, 28]
    else:
        case_list = [25, 26, 27, 28, 29]

    if saycan is True:
        case_list = [25]

    if case_id is None:
        case_id = random.choice(case_list)
    else:
        case_id = case_id
    object_name = None
    knife_id = None
    obj_id = None
    obj_id_list = []
    tactile = None

    object_list = ['apple', 'banana', 'eggplant']
    object_name = random.choice(object_list)

    if case_id == 25:  # Low level command: Done
        # tactile infomation
        tactile = 0

        object_drop_position_lists = [[0.15, 0], [0.15, -0.1], [0.15, 0.1]]
        object_list = ['bottle', 'can', 'chips']
        drop_x_objects = random.randint(1, 3)
        object_names = random.choices(object_list, k=drop_x_objects)
        object_drop_positions = random.sample(object_drop_position_lists, k=drop_x_objects)
        for i in range(drop_x_objects):
            # load object
            obj_path = './model/task_clean/obj/' + object_names[i] + '/' + object_names[i] + '.urdf'
            if obj_random is False:
                print('something wrong')
                pass
            else:
                obj_x = object_drop_positions[i][0] + (random.random()*2-1)*0.025
                obj_y = object_drop_positions[i][1] + (random.random()*2-1)*0.025
                obj_z = 0.01
                obj_roll = 0
                obj_pitch = 0
                obj_yaw = 0 + (random.random()*2-1)*3.14/4
                scale = random.uniform(0.85, 1.1)
            obj_id = p.loadURDF(obj_path, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
                [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False, globalScaling=scale)
            obj_id_list.append(obj_id)
            for _ in range(40):
                p.stepSimulation()

        if obj_random is False:
            knife_x = -0.175
            knife_y = 0
            knife_z = 0.01
        else:
            knife_x = -0.175 + (random.random()*2-1)*0.01
            knife_y = 0 + (random.random()*2-1)*0.01
            knife_z = 0.01
            scale = random.uniform(0.85, 1.1)
        # load knife
        knife_id = p.loadURDF('./model/task_cut/obj/knife/knife.urdf', [
                              knife_x, knife_y, knife_z], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=False, globalScaling=scale)

        # load object
        obj_path = './model/task_cut/obj/' + object_name + '/' + object_name + '.urdf'
        if obj_random is False:
            obj_x = 0
            obj_y = 0
            obj_z = 0.01
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0
        else:
            obj_x = 0 + (random.random()*2-1)*0.025
            obj_y = 0 + (random.random()*2-1)*0.025
            obj_z = 0.01
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0 + (random.random()*2-1)*3.14/4
            scale = random.uniform(0.85, 1.1)

        obj_id = p.loadURDF(obj_path, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
            [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False, globalScaling=scale)

        for _ in range(40):
            p.stepSimulation()

    elif case_id == 26:  # Low level command: Done
        # tactile infomation
        tactile = 1
        object_drop_position_lists = [[0.15, 0], [0.15, -0.1], [0.15, 0.1]]
        object_list = ['bottle', 'can', 'chips']
        drop_x_objects = random.randint(1, 3)
        object_names = random.choices(object_list, k=drop_x_objects)
        object_drop_positions = random.sample(object_drop_position_lists, k=drop_x_objects)
        for i in range(drop_x_objects):
            # load object
            obj_path = './model/task_clean/obj/' + object_names[i] + '/' + object_names[i] + '.urdf'
            if obj_random is False:
                print('something wrong')
                pass
            else:
                obj_x = object_drop_positions[i][0] + (random.random()*2-1)*0.025
                obj_y = object_drop_positions[i][1] + (random.random()*2-1)*0.025
                obj_z = 0.01
                obj_roll = 0
                obj_pitch = 0
                obj_yaw = 0 + (random.random()*2-1)*3.14/4
                scale = random.uniform(0.85, 1.1)
            obj_id = p.loadURDF(obj_path, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
                [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False, globalScaling=scale)
            obj_id_list.append(obj_id)
            for _ in range(40):
                p.stepSimulation()

        if obj_random is False:
            knife_x = -0.175
            knife_y = 0
            knife_z = 0.01
        else:
            knife_x = -0.175 + (random.random()*2-1)*0.01
            knife_y = 0 + (random.random()*2-1)*0.01
            knife_z = 0.01
            scale = random.uniform(0.85, 1.1)
        # load knife
        knife_id = p.loadURDF('./model/task_cut/obj/knife/knife.urdf', [
                              knife_x, knife_y, knife_z], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=False, globalScaling=scale)

        # load object
        obj_path = './model/task_cut/obj/' + object_name + '/' + object_name + '.urdf'
        if obj_random is False:
            obj_x = 0
            obj_y = 0
            obj_z = 0.01
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0
        else:
            obj_x = 0 + (random.random()*2-1)*0.025
            obj_y = 0 + (random.random()*2-1)*0.025
            obj_z = 0.01
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0 + (random.random()*2-1)*3.14/4
            scale = random.uniform(0.85, 1.1)

        obj_id = p.loadURDF(obj_path, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
            [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False, globalScaling=scale)
        for _ in range(40):
            p.stepSimulation()
    elif case_id == 27:  # Low level command: Done
        # tactile infomation
        tactile = 0
        object_list = ['apple', 'banana', 'eggplant']
        object_name = random.choice(object_list)
        if obj_random is False:
            knife_x = -0.175
            knife_y = 0
            knife_z = 0.01
        else:
            knife_x = -0.175 + (random.random()*2-1)*0.01
            knife_y = 0 + (random.random()*2-1)*0.01
            knife_z = 0.01
            scale = random.uniform(0.85, 1.1)
        # load knife
        knife_id = p.loadURDF('./model/task_cut/obj/knife/knife.urdf', [
                              knife_x, knife_y, knife_z], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=False, globalScaling=scale)

        # load object
        obj_path = './model/task_cut/obj/' + object_name + '/' + object_name + '.urdf'
        if obj_random is False:
            obj_x = 0
            obj_y = 0
            obj_z = 0.01
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0
        else:
            obj_x = 0 + (random.random()*2-1)*0.025
            obj_y = 0 + (random.random()*2-1)*0.025
            obj_z = 0.01
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0 + (random.random()*2-1)*3.14/4
            scale = random.uniform(0.85, 1.1)

        obj_id = p.loadURDF(obj_path, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
            [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False, globalScaling=scale)
        obj_id_list.append(obj_id)
        for _ in range(40):
            p.stepSimulation()

    elif case_id == 28:  # Low level command: Done
        # tactile infomation
        tactile = 1

        # load object
        obj_path = './model/task_cut/obj/' + object_name + '/' + object_name + '.urdf'
        if obj_random is False:
            obj_x = 0
            obj_y = 0
            obj_z = 0.01
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0
        else:
            obj_x = 0 + (random.random()*2-1)*0.025
            obj_y = 0 + (random.random()*2-1)*0.025
            obj_z = 0.01
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0 + (random.random()*2-1)*3.14/4
            scale = random.uniform(0.85, 1.1)

        obj_id = p.loadURDF(obj_path, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
            [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False, globalScaling=scale)
        obj_id_list.append(obj_id)
        for _ in range(40):
            p.stepSimulation()

    elif case_id == 29:  # Low level command: Done
        # tactile infomation
        tactile = 0

        # load object
        obj_path_left = './model/task_cut/obj/' + object_name + '/' + object_name + '_left.urdf'
        obj_path_right = './model/task_cut/obj/' + object_name + '/' + object_name + '_right.urdf'
        if obj_random is False:
            obj_x = 0
            obj_y = 0
            obj_z = 0.05
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0
        else:
            obj_x = 0 + (random.random()*2-1)*0.025
            obj_y = 0 + (random.random()*2-1)*0.025
            obj_z = 0.035 + (random.random()*2-1)*0.025
            obj_roll = 0 + (random.random()*2-1)*1.57
            obj_pitch = 0 + (random.random()*2-1)*3.14/20
            obj_yaw = 0 + (random.random()*2-1)*3.14/4
            scale = random.uniform(0.85, 1.1)

        obj_l = p.loadURDF(obj_path_left, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
            [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False,  globalScaling=scale)
        obj_r = p.loadURDF(obj_path_right, [obj_x, obj_y, obj_z], p.getQuaternionFromEuler(
            [obj_roll, obj_pitch, obj_yaw]), useFixedBase=False,  globalScaling=scale)

        for _ in range(40):
            p.stepSimulation()

    return case_id, object_name, obj_id_list, tactile, knife_id


# p.connect(p.DIRECT)
# p.connect(p.GUI)
# multi4(obj_random=True, rl=False)
# sim_d, sim_float, mask = render_camera()
# plt.imshow(sim_d)
# %%
def build_pick_round_env(obj_random=True, rl=False, saycan=False, case_id=None):

    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=50)

    # p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetDebugVisualizerCamera(
        cameraDistance=.2, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0, 0, 0.2])
    # Define world
    p.setGravity(0, 0, -9.8)
    p.loadURDF('plane.urdf', [0, 0, 0], useFixedBase=True)

    # random case:
    # 1: start state, random objects drop, has objects, tactile = 0 , need to grasp round
    # 2: middle state, tactile = 1, need to put the object into trash can
    # 3: final state, no round object in the image and tactile = 0
    if rl is True:
        case_list = [30, 31]
        # case_list = [9]
    else:
        case_list = [30, 31, 32]

    if saycan is True:
        case_list = [30, 31]

    if case_id is None:
        case_id = random.choice(case_list)
    else:
        case_id = case_id
    object_name = None
    knife_id = None
    obj_id = None
    obj_id_list = []
    tactile = None
    # todo
    object_drop_position_lists = [[0, 0.075], [0, -0.075], [-0.075, 0.075], [-0.075, -0.075], [-0.15, 0.075], [-0.15, -0.075]]
    if case_id == 30:  # Low level command:
        tactile = 0
        object_drop_positions = random.sample(object_drop_position_lists, k=5)
        for i in range(5):  # 0 1 2 3 4
            # load object
            obj_x = object_drop_positions[i][0] + (random.random()*2-1)*0.025
            obj_y = object_drop_positions[i][1] + (random.random()*2-1)*0.025
            obj_z = 0.01
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0 + (random.random()*2-1)*3.14/4
            scale = random.uniform(0.85, 1.1)
            if i <= 2:
                obj_id = generate_roll_obj(obj_position=[obj_x, obj_y, obj_z], obj_orientation=p.getQuaternionFromEuler([0, 0, obj_yaw]))
                obj_id_list.append(obj_id)
            else:
                obj_id = generate_td_obj(obj_position=[obj_x, obj_y, obj_z], obj_orientation=p.getQuaternionFromEuler([0, 0, obj_yaw]))
            for _ in range(40):
                p.stepSimulation()

    elif case_id == 31:  # Low level command:
        tactile = 1
        obj_num = random.randint(2, 4)
        object_drop_positions = random.sample(object_drop_position_lists, k=obj_num)
        for i in range(obj_num):  # 0 1 2 3
            # load object
            obj_x = object_drop_positions[i][0] + (random.random()*2-1)*0.025
            obj_y = object_drop_positions[i][1] + (random.random()*2-1)*0.025
            obj_z = 0.01
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0 + (random.random()*2-1)*3.14/4
            scale = random.uniform(0.85, 1.1)
            if i <= 1:
                obj_id = generate_td_obj(obj_position=[obj_x, obj_y, obj_z], obj_orientation=p.getQuaternionFromEuler([0, 0, obj_yaw]))
            else:
                obj_id = generate_roll_obj(obj_position=[obj_x, obj_y, obj_z], obj_orientation=p.getQuaternionFromEuler([0, 0, obj_yaw]))
            for _ in range(40):
                p.stepSimulation()

    elif case_id == 32:
        tactile = 0
        obj_num = 2
        object_drop_positions = random.sample(object_drop_position_lists, k=obj_num)
        for i in range(obj_num):  # 0 1 2 3
            # load object
            obj_x = object_drop_positions[i][0] + (random.random()*2-1)*0.025
            obj_y = object_drop_positions[i][1] + (random.random()*2-1)*0.025
            obj_z = 0.01
            obj_roll = 0
            obj_pitch = 0
            obj_yaw = 0 + (random.random()*2-1)*3.14/4
            scale = random.uniform(0.85, 1.1)

            obj_id = generate_td_obj(obj_position=[obj_x, obj_y, obj_z], obj_orientation=p.getQuaternionFromEuler([0, 0, obj_yaw]))
            for _ in range(40):
                p.stepSimulation()

    return case_id, object_name, obj_id_list, tactile, knife_id


# p.connect(p.DIRECT)
# # p.connect(p.GUI)
# build_pick_round_env(obj_random=True, rl=False)
# sim_d, sim_float, mask = render_camera()
# plt.imshow(sim_d)
