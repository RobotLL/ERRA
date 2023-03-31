import numpy as np
import torch
import clip
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from build_env import build_cut_sth_env, build_clean_table_env, build_pick_sth_into_drawer_env
from torchvision import datasets, models, transforms
import torch.optim as optim
from util import render_camera
import os
import pybullet as p
from random import shuffle
from tqdm import tqdm

# high level
from task_clean_main import collect_high_level_task_data as t1h
from task_cut_main import collect_high_level_task_data as t2h
from task_drawer_main import collect_high_level_task_data as t3h
from task_pick_round_main import collect_high_level_task_data as t4h

# low level
from task_clean_main import collect_low_level_task_data as t1l
from task_cut_main import collect_low_level_task_data as t2l
from task_drawer_main import collect_low_level_task_data as t3l
#from task_pick_round_main import collect_low_level_task_data as t4l

# multi tasks
from task_multi1_main import collect_high_level_task_data as t5h
from task_multi2_main import collect_high_level_task_data as t6h
from task_multi3_main import collect_high_level_task_data as t7h
from task_multi4_main import collect_high_level_task_data as t8h


collect_number_for_each_task = 500


def Image2embedding(image_input_list):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load("ViT-B/32", device=device)
    # model, preprocess = clip.load("RN50", device=device)
    image_features = list()

    with torch.no_grad():
        for image in image_input_list:
            _image = preprocess(Image.open(image)).unsqueeze(0).to(device)
            image_features.append(model.encode_image(_image).squeeze().cpu().numpy()) #size [1,512]   

    return image_features 

def main():

    output_path = f"./data/training-{collect_number_for_each_task}/"
    os.makedirs(output_path, exist_ok=True)

    language_input_list = []
    image_input_list = []
    tactile_input_list = []
    label_list = []
    # %%
    language_input, image_input, tactile_input, label, = t1h(collect_number_for_each_task,output_path)
    language_input_list.extend(language_input)
    image_input_list.extend(Image2embedding(image_input))
    tactile_input_list.extend(tactile_input)
    label_list.extend(label)

    language_input, image_input, tactile_input, label, = t2h(collect_number_for_each_task,output_path)
    language_input_list.extend(language_input)
    image_input_list.extend(Image2embedding(image_input))
    tactile_input_list.extend(tactile_input)
    label_list.extend(label)

    language_input, image_input, tactile_input, label, = t3h(collect_number_for_each_task,output_path)
    language_input_list.extend(language_input)
    image_input_list.extend(Image2embedding(image_input))
    tactile_input_list.extend(tactile_input)
    label_list.extend(label)

    language_input, image_input, tactile_input, label, = t4h(collect_number_for_each_task,output_path)
    language_input_list.extend(language_input)
    image_input_list.extend(Image2embedding(image_input))
    tactile_input_list.extend(tactile_input)
    label_list.extend(label)

    language_input, image_input, tactile_input, label, = t5h(collect_number_for_each_task,output_path)
    language_input_list.extend(language_input)
    image_input_list.extend(Image2embedding(image_input))
    tactile_input_list.extend(tactile_input)
    label_list.extend(label)

    language_input, image_input, tactile_input, label, = t6h(collect_number_for_each_task,output_path)
    language_input_list.extend(language_input)
    image_input_list.extend(Image2embedding(image_input))
    tactile_input_list.extend(tactile_input)
    label_list.extend(label)

    language_input, image_input, tactile_input, label, = t7h(collect_number_for_each_task,output_path)
    language_input_list.extend(language_input)
    image_input_list.extend(Image2embedding(image_input))
    tactile_input_list.extend(tactile_input)
    label_list.extend(label)

    language_input, image_input, tactile_input, label, = t8h(collect_number_for_each_task,output_path)
    language_input_list.extend(language_input)
    image_input_list.extend(Image2embedding(image_input))
    tactile_input_list.extend(tactile_input)
    label_list.extend(label)

    language_input, image_input, tactile_input, label, = t1l(collect_number_for_each_task,output_path)
    language_input_list.extend(language_input)
    image_input_list.extend(Image2embedding(image_input))
    tactile_input_list.extend(tactile_input)
    label_list.extend(label)

    language_input, image_input, tactile_input, label, = t2l(collect_number_for_each_task,output_path)
    language_input_list.extend(language_input)
    image_input_list.extend(Image2embedding(image_input))
    tactile_input_list.extend(tactile_input)
    label_list.extend(label)

    language_input, image_input, tactile_input, label, = t3l(collect_number_for_each_task,output_path)
    language_input_list.extend(language_input)
    image_input_list.extend(Image2embedding(image_input))
    tactile_input_list.extend(tactile_input)
    label_list.extend(label)

#     language_input, image_input, tactile_input, label, = t4l(collect_number_for_each_task,output_path)
#     language_input_list.extend(language_input)
#     image_input_list.extend(Image2embedding(image_input))
#     tactile_input_list.extend(tactile_input)
#     label_list.extend(label)

    # %%

    results = [language_input_list,image_input_list,tactile_input_list,label_list]
    files = ["txt.csv","img.csv","tact.csv","label.csv"]
    for file,n in zip(files, [0,1,2,3]):
        with open(os.path.join(output_path,file), 'w') as f:
            writer=csv.writer(f)
            if n==2:
                writer.writerows(map(lambda y:str(y),results[n]))
            elif n==0 or n==3:
                writer.writerows(map(lambda y:[y],results[n]))
            else:
                writer.writerows(results[n])

    
if __name__ == "__main__":
    main()
