import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from build_env import build_cut_sth_env
from util import render_camera
import pybullet as p
import random
import os
# %%
use_gui = False
if use_gui:
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.3, cameraYaw=50.8, cameraPitch=-44.2, cameraTargetPosition=[-0.56, 0.47, -0.52])
else:
    p.connect(p.DIRECT)


def collect_high_level_task_data(collect_number, output_path, c_id=None, generate_test=False):
    os.makedirs(output_path+'task_cut/', exist_ok=True)
    language_input_list = []
    image_input_list = []
    tactile_input_list = []
    label_language_list = []

    for d_id in range(collect_number):
        case_id, object_name, obj_id_list,  tactile_info, _ = build_cut_sth_env(
            obj_random=True, rl=False, case_id=c_id)
        sim_d, sim_float, mask = render_camera()
        # plt.imshow(sim_d)
        if c_id is not None:
            img_save_path = output_path +'task_cut/' + str(int(d_id)) + '_' + str(int(case_id)) + '_high.jpg'
        elif generate_test is False:
            img_save_path = output_path +'task_cut/' + str(int(d_id)) + '_high.jpg'
        elif generate_test is True:
            img_save_path = output_path +'task_cut/' + str(int(d_id)) + 'test_high.jpg'
        depth_image = Image.fromarray(sim_d).convert('L')
        depth_image.save(img_save_path)

        high_level_command = ['Please cut the ' + object_name]
        low_level_command = ['Grasp the knife', 'Cut the ' + object_name, 'Done']
        language_input_list.append(high_level_command[0])
        image_input_list.append(img_save_path)
        if case_id == 6:
            label_language_list.append(low_level_command[0])
            tactile_input_list.append(tactile_info)
        elif case_id == 7:
            label_language_list.append(low_level_command[1])
            tactile_input_list.append(tactile_info)
        elif case_id == 8:
            label_language_list.append(low_level_command[2])
            tactile_input_list.append(tactile_info)

    return language_input_list, image_input_list, tactile_input_list, label_language_list


# %%


def collect_low_level_task_data(collect_number, output_path, c_id=None, generate_test=False):
    os.makedirs(output_path+'task_cut/', exist_ok=True)
    language_input_list = []
    image_input_list = []
    tactile_input_list = []
    label_language_list = []
    for d_id in range(collect_number):
        case_id, object_name, obj_id_list,  tactile_info, _ = build_cut_sth_env(
            obj_random=True, rl=False, case_id=c_id)
        sim_d, sim_float, mask = render_camera()
        # plt.imshow(sim_d)
        if c_id is not None:
            img_save_path = output_path +'task_cut/' + str(int(d_id)) + '_' + str(int(case_id)) + '_low.jpg'
        elif generate_test is False:
            img_save_path = output_path +'task_cut/' + str(int(d_id)) + '_low.jpg'
        elif generate_test is True:
            img_save_path = output_path +'task_cut/' + str(int(d_id)) + '_test_low.jpg'
        depth_image = Image.fromarray(sim_d).convert('L')
        depth_image.save(img_save_path)

        low_level_command = ['Grasp the knife',
                             'Cut the ' + object_name,
                             'Done']

        if case_id == 6:
            image_input_list.append(img_save_path)
            language_input_list.append(low_level_command[0])
            label_language_list.append(low_level_command[0])
            tactile_input_list.append(tactile_info)
        elif case_id == 7:
            if random.random() > 0.5:
                image_input_list.append(img_save_path)
                language_input_list.append(low_level_command[1])
                label_language_list.append(low_level_command[1])
                tactile_input_list.append(tactile_info)
            else:
                image_input_list.append(img_save_path)
                language_input_list.append(low_level_command[0])
                label_language_list.append(low_level_command[2])
                tactile_input_list.append(tactile_info)
        elif case_id == 8:
            image_input_list.append(img_save_path)
            language_input_list.append(low_level_command[1])
            label_language_list.append(low_level_command[2])
            tactile_input_list.append(tactile_info)

    return language_input_list, image_input_list, tactile_input_list, label_language_list


# %%
if __name__ == "__main__":
    high_level_collect_number = 1
    language_input_list1, image_input_list1, tactile_input_list1, label_list1, = collect_high_level_task_data(
        high_level_collect_number)
    low_level_collect_number = 1
    language_input_list2, image_input_list2, tactile_input_list2,  label_list2, = collect_low_level_task_data(
        low_level_collect_number)
    merge_list = [language_input_list1+language_input_list2,
                  image_input_list1+image_input_list2, tactile_input_list1+tactile_input_list2, label_list1+label_list2]
    np.savetxt("task_cut.csv", merge_list, delimiter=", ", fmt='% s')
