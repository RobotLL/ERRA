import numpy as np
import random
import pybullet 
import pybullet as p
import pybullet_data
from PIL import Image
import matplotlib.pyplot as plt
import math
from util import render_camera

p.connect(p.DIRECT)
# p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetDebugVisualizerCamera(cameraDistance=.2, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0,0,0.2])
# Define world
p.setGravity(0, 0, -9.8)
p.loadURDF('plane.urdf', [0, 0, 0], useFixedBase=True)


drawer_down= p.loadURDF('./model/task_drawer/obj/drawer/drawer_down.urdf', [0.1, -0.1+0.1, 0], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=False) 
drawer_up= p.loadURDF('./model/task_drawer/obj/drawer/drawer_up.urdf', [0.1, -0.1, 0], p.getQuaternionFromEuler([math.radians(90), 0, math.radians(180)]), useFixedBase=True) 
gripper_urdf = './model/robot/gripper.urdf'
# robot_id = p.loadURDF(gripper_urdf, [0.05,0.05,0.2], [0, 0, 0, 1], flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)

for _ in range(240):
    p.stepSimulation()
#%%
process_depth, sim_float, mask = render_camera()
plt.imshow(sim_float)
#%%
p.disconnect()



