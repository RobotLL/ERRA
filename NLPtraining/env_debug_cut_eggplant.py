import numpy as np
import random
import pybullet 
import pybullet as p
import pybullet_data
from PIL import Image
import matplotlib.pyplot as plt
import math
from util import render_camera

# p.connect(p.DIRECT)
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetDebugVisualizerCamera(cameraDistance=.2, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0,0,0.2])
# Define world
p.setGravity(0, 0, -9.8)
p.loadURDF('plane.urdf', [0, 0, 0], useFixedBase=True)
      
# obj= p.loadURDF('./model/task_cut/obj/eggplant/eggplant.urdf', [0.1, 0, 0.01], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=False) 

knife = p.loadURDF('./model/task_cut/obj/knife/knife.urdf', [-0.075, 0, 0.05], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=False) 

obj_l= p.loadURDF('./model/task_cut/obj/eggplant/eggplant_left.urdf', [0.1, 0, 0.05], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=False) 
obj_r= p.loadURDF('./model/task_cut/obj/eggplant/eggplant_right.urdf', [0.1, 0, 0.05], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=False) 

gripper_urdf = './model/robot/gripper.urdf'
robot_id = p.loadURDF(gripper_urdf, [0.05,0.05,0.1], [0, 0, 0, 1], flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)

for _ in range(240):
    p.stepSimulation()
#%%
process_depth, sim_float, mask = render_camera()
plt.imshow(sim_float)
#%%
p.disconnect()



