import numpy as np

# high level
from task_clean_main import collect_high_level_task_data as t1h  # case id = 9 or 10 or 11
from task_cut_main import collect_high_level_task_data as t2h  # case id =6 or 7 or 8
from task_drawer_main import collect_high_level_task_data as t3h  # case id =1 or 2 or 3 or 4 or 5
from task_pick_round_main import collect_high_level_task_data as t4h  # case id =30 or 31 or 32

# low level
from task_clean_main import collect_low_level_task_data as t1l  # case id = 9 or 10 or 11
from task_cut_main import collect_low_level_task_data as t2l  # case id =6 or 7 or 8
from task_drawer_main import collect_low_level_task_data as t3l  # case id =1 or 2 or 3 or 4 or 5
from task_pick_round_main import collect_low_level_task_data as t4l  # case id =30 or 31 or 32

# multi tasks
from task_multi1_main import collect_high_level_task_data as t5h  # case id =12 or 13 or 14
from task_multi2_main import collect_high_level_task_data as t6h  # case id =15 or 16 or 17
from task_multi3_main import collect_high_level_task_data as t7h  # case id =18 or 19 or 20 or 21 or 22 or 23 or 24
from task_multi4_main import collect_high_level_task_data as t8h  # case id =25 or 26 or 27 or 28 or 29

# %%high-level case id
for c_id in [9, 10, 11]:
    collect_number_for_each_task = 200
    language_input, image_input, tactile_input, label, = t1h(collect_number_for_each_task, c_id=c_id)
    merge_list = [language_input, image_input, tactile_input, label]
    np.savetxt('high_level_case_id'+str(int(c_id))+'.csv', merge_list, delimiter=", ", fmt='% s')

for c_id in [6, 7, 8]:
    collect_number_for_each_task = 200
    language_input, image_input, tactile_input, label, = t2h(collect_number_for_each_task, c_id=c_id)
    merge_list = [language_input, image_input, tactile_input, label]
    np.savetxt('high_level_case_id'+str(int(c_id))+'.csv', merge_list, delimiter=", ", fmt='% s')

for c_id in [1, 2, 3, 4, 5]:
    collect_number_for_each_task = 200
    language_input, image_input, tactile_input, label, = t3h(collect_number_for_each_task, c_id=c_id)
    merge_list = [language_input, image_input, tactile_input, label]
    np.savetxt('high_level_case_id'+str(int(c_id))+'.csv', merge_list, delimiter=", ", fmt='% s')

for c_id in [30, 31, 32]:
    collect_number_for_each_task = 200
    language_input, image_input, tactile_input, label, = t4h(collect_number_for_each_task, c_id=c_id)
    merge_list = [language_input, image_input, tactile_input, label]
    np.savetxt('high_level_case_id'+str(int(c_id))+'.csv', merge_list, delimiter=", ", fmt='% s')


# %% low-level case id
for c_id in [9, 10, 11]:
    collect_number_for_each_task = 200
    language_input, image_input, tactile_input, label, = t1l(collect_number_for_each_task, c_id=c_id)
    merge_list = [language_input, image_input, tactile_input, label]
    np.savetxt('low_level_case_id'+str(int(c_id))+'.csv', merge_list, delimiter=", ", fmt='% s')

for c_id in [6, 7, 8]:
    collect_number_for_each_task = 200
    language_input, image_input, tactile_input, label, = t2l(collect_number_for_each_task, c_id=c_id)
    merge_list = [language_input, image_input, tactile_input, label]
    np.savetxt('low_level_case_id'+str(int(c_id))+'.csv', merge_list, delimiter=", ", fmt='% s')

for c_id in [1, 2, 3, 4, 5]:
    collect_number_for_each_task = 200
    language_input, image_input, tactile_input, label, = t3l(collect_number_for_each_task, c_id=c_id)
    merge_list = [language_input, image_input, tactile_input, label]
    np.savetxt('low_level_case_id'+str(int(c_id))+'.csv', merge_list, delimiter=", ", fmt='% s')

for c_id in [30, 31, 32]:
    collect_number_for_each_task = 200
    language_input, image_input, tactile_input, label, = t4l(collect_number_for_each_task, c_id=c_id)
    merge_list = [language_input, image_input, tactile_input, label]
    np.savetxt('low_level_case_id'+str(int(c_id))+'.csv', merge_list, delimiter=", ", fmt='% s')

# %% multi-task case id
for c_id in [12, 13, 14]:
    collect_number_for_each_task = 200
    language_input, image_input, tactile_input, label, = t5h(collect_number_for_each_task, c_id=c_id)
    merge_list = [language_input, image_input, tactile_input, label]
    np.savetxt('multi_case_id'+str(int(c_id))+'.csv', merge_list, delimiter=", ", fmt='% s')

for c_id in [15, 16, 17]:
    collect_number_for_each_task = 200
    language_input, image_input, tactile_input, label, = t6h(collect_number_for_each_task, c_id=c_id)
    merge_list = [language_input, image_input, tactile_input, label]
    np.savetxt('multi_case_id'+str(int(c_id))+'.csv', merge_list, delimiter=", ", fmt='% s')

for c_id in [18, 19, 20, 21, 22, 23, 24]:
    collect_number_for_each_task = 200
    language_input, image_input, tactile_input, label, = t7h(collect_number_for_each_task, c_id=c_id)
    merge_list = [language_input, image_input, tactile_input, label]
    np.savetxt('multi_case_id'+str(int(c_id))+'.csv', merge_list, delimiter=", ", fmt='% s')

for c_id in [25, 26, 27, 28, 29]:
    collect_number_for_each_task = 200
    language_input, image_input, tactile_input, label, = t8h(collect_number_for_each_task, c_id=c_id)
    merge_list = [language_input, image_input, tactile_input, label]
    np.savetxt('multi_case_id'+str(int(c_id))+'.csv', merge_list, delimiter=", ", fmt='% s')
