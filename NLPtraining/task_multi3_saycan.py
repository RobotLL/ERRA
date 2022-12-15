import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from build_env import multi3
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
    os.makedirs(output_path+'task_multi3_saycan/', exist_ok=True)
    language_input_list = []
    image_input_list = []
    tactile_input_list = []
    label_language_list = []

    for d_id in range(collect_number):
        try:
            case_id, object_name, obj_id_list,  tactile_info, _ = multi3(
                obj_random=True, rl=False, case_id=c_id)
        except ValueError:
            import pdb; pdb.set_trace()
        
        sim_d, sim_float, mask = render_camera()
        if c_id is not None:
            img_save_path = output_path+'task_multi3_saycan/' + str(int(d_id)) + '_' + str(int(case_id)) + '_high.jpg'
        elif generate_test is False:
            img_save_path = output_path+'task_multi3_saycan/' + str(int(d_id)) + '_high.jpg'
        elif generate_test is True:
            img_save_path = output_path+'task_multi3_saycan/' + str(int(d_id)) + 'test_high.jpg'
        depth_image = Image.fromarray(sim_d).convert('L')
        depth_image.save(img_save_path)

        high_level_command = ['Please put the cosmetic into the drawer and then clean the table.']
        low_level_command = ['Open the drawer',
                             'Grasp the cosmetic',
                             'Put the cosmetic into drawer',
                             'Close the drawer',
                             'Grasp an object',
                             'Put the object into the bin',
                             'Done']

        language_input_list.append(high_level_command[0])
        image_input_list.append(img_save_path)

        if case_id == 18:
            if len(obj_id_list) == 1:
                label_language_list.append([low_level_command[0], low_level_command[1],
                                           low_level_command[2], low_level_command[3],
                                           low_level_command[4], low_level_command[5],
                                           low_level_command[6]])
                tactile_input_list.append(tactile_info)
            if len(obj_id_list) == 2:
                label_language_list.append([low_level_command[0], low_level_command[1],
                                           low_level_command[2], low_level_command[3],
                                           low_level_command[4], low_level_command[5],
                                           low_level_command[4], low_level_command[5],
                                           low_level_command[6]])
                tactile_input_list.append(tactile_info)
            if len(obj_id_list) == 3:
                label_language_list.append([low_level_command[0], low_level_command[1],
                                           low_level_command[2], low_level_command[3],
                                           low_level_command[4], low_level_command[5],
                                           low_level_command[4], low_level_command[5],
                                           low_level_command[4], low_level_command[5],
                                           low_level_command[6]])
                tactile_input_list.append(tactile_info)
    return language_input_list, image_input_list, tactile_input_list, label_language_list


# %%
if __name__ == "__main__":
    high_level_collect_number = 1
    language_input_list1, image_input_list1, tactile_input_list1, label_list1, = collect_high_level_task_data(
        high_level_collect_number)
    merge_list = [language_input_list1, image_input_list1, tactile_input_list1, label_list1]
    np.savetxt("task_multi3.csv", merge_list, delimiter=", ", fmt='% s')
