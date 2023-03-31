import numpy as np
import gym
import pybullet as p
import random
import math
import time
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict
import torch
from util import normalize_01 as normalize
from util import normalize_01, objects_limits, gripper_orn_to_world, gripper_limits, discretize, normalize_m11
from util import render_camera as camera
from build_env import build_pick_sth_into_drawer_env, build_cut_sth_env, build_clean_table_env,  build_pick_round_env, multi1, multi2, multi3, multi4

ACTION_SIZE = 4
VECTOR_SIZE = 3


class Sim(gym.Env):

    def __init__(self, gui=False, discrete=True, number_of_objects=1, reset_interval=1, classification_model=None, test=False):
        self.benchmark = False
        self.force_environment_id = None
        self.force_case_id = None

        #robot and objects
        self.robot_position = [0, 0, 0.4]
        self.reset_every = reset_interval
        self.gripper_urdf = './model/robot/gripper.urdf'
        self.reward_height = 0.05
        self.test = test

        # var
        self.joints = {}
        self.objects = []
        self.gripper_orn_obs = [0, 0, 0]
        self.if_open = 0
        # rl relative
        self.steps = 0
        self.episode = 0

        # camera
        self.depth_image = None
        self.use_gui = gui

        if self.use_gui:
            p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.3, cameraYaw=50.8, cameraPitch=-44.2, cameraTargetPosition=[-0.56, 0.47, -0.52])
        else:
            p.connect(p.DIRECT)

        self.observation_space = self.get_observation_space()
        # x y yaw if_open
        self.action_space = gym.spaces.MultiDiscrete([21, 21, 2, 2])

    def set_environment_properties(self, target_number_of_objects, workspace_length):
        self.object_number = target_number_of_objects

    def get_observation_space(self):
        return gym.spaces.Dict({
            'vector': gym.spaces.Box(low=-1, high=1, shape=(VECTOR_SIZE,), dtype=np.float32),
            'image': gym.spaces.Box(low=0, high=255, shape=(1, 120, 160), dtype=np.uint8)
        })

    def get_action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(ACTION_SIZE,), dtype=np.float32)

    def _delay(self, duration):
        for _ in range(duration):
            if self.use_gui == True:
                time.sleep(0.005)
            p.stepSimulation()

    def render_camera(self):
        process_depth, sim_float, mask = camera(rl=True)
        process_depth = process_depth[np.newaxis, :, :]
        process_depth = np.array(process_depth).astype(np.uint8)
        self.depth_image = process_depth

    def get_observation(self, x, y, yaw):

        current_x = normalize(x, gripper_limits['joint_x'])
        current_y = normalize(y, gripper_limits['joint_y'])
        current_yaw = normalize(yaw, gripper_limits['joint_yaw'])
        current_case_id = normalize(self.sentence_id, gripper_limits['rl_id'])
        current_tactile = normalize(self.tactile, gripper_limits['joint_open'])
        current_step = normalize(self.steps, gripper_limits['steps'])
        gripper_pose = [current_tactile, current_case_id, current_step]

        return {
            'vector': np.array(
                gripper_pose,
                dtype=np.float32
            ),
            'image': self.depth_image
        }

    def get_real_vector(self, x, y, yaw, case_id, tactile, steps):
        # for real test
        current_x = normalize(x, gripper_limits['joint_x'])
        current_y = normalize(y, gripper_limits['joint_y'])
        current_yaw = normalize(yaw, gripper_limits['joint_yaw'])

        current_case_id = normalize(case_id, gripper_limits['case_id'])
        current_if_open = normalize(tactile, gripper_limits['joint_open'])
        current_step = normalize(steps, gripper_limits['steps'])

        gripper_pose = [current_x, current_y, current_yaw,
                        current_case_id, current_if_open, current_step]

        return np.array(gripper_pose, dtype=np.float32)

    def build_environment(self):
        if self.selected_env == 1:
            self.case_id, self.object_name, self.objects, self.tactile, _, self.drawer_down = build_pick_sth_into_drawer_env(
                obj_random=not self.test, rl=True, force_case_id=self.force_case_id)
        elif self.selected_env == 2:
            self.case_id, self.object_name, self.objects, self.tactile, self.knife_id = build_cut_sth_env(
                obj_random=not self.test, rl=True, force_case_id=self.force_case_id)
        elif self.selected_env == 3:
            self.case_id, self.object_name, self.objects, self.tactile, _ = build_clean_table_env(
                obj_random=not self.test, rl=True, force_case_id=self.force_case_id)
        elif self.selected_env == 4:
            self.case_id, self.object_name, self.objects, self.tactile, _ = build_pick_round_env(
                obj_random=not self.test, rl=True, force_case_id=self.force_case_id)

        elif self.selected_env == 5:
            self.case_id, self.object_name, self.objects, self.tactile, _, self.drawer_down = multi1(
                obj_random=not self.test, rl=True, force_case_id=self.force_case_id)
        elif self.selected_env == 6:
            self.case_id, self.object_name, self.objects, self.tactile, self.knife_id, self.drawer_down = multi2(
                obj_random=not self.test, rl=True, force_case_id=self.force_case_id)
        elif self.selected_env == 7:
            self.case_id, self.object_name, self.objects, self.tactile, self.knife_id, self.drawer_down = multi3(
                obj_random=not self.test, rl=True, force_case_id=self.force_case_id)
        elif self.selected_env == 8:
            self.case_id, self.object_name, self.objects, self.tactile, self.knife_id = multi4(
                obj_random=not self.test, rl=True, force_case_id=self.force_case_id)

        if self.benchmark == True:
            # Define the robot
            self.robot_id = p.loadURDF(self.gripper_urdf, self.robot_position, [
                                       0, 0, 0, 1], flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)

            # Store joint information
            jointTypeList = ["REVOLUTE", "PRISMATIC",
                             "SPHERICAL", "PLANAR", "FIXED"]
            self.number_of_joints = p.getNumJoints(self.robot_id)
            jointInfo = namedtuple("jointInfo",
                                   ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity"])

            self.joints = AttrDict()
            self.control_link_index = 0
            # get jointInfo and index of dummy_center_indicator_link
            for i in range(self.number_of_joints):
                info = p.getJointInfo(self.robot_id, i)
                jointID = info[0]
                jointName = info[1].decode("utf-8")
                jointType = jointTypeList[info[2]]
                jointLowerLimit = info[8]
                jointUpperLimit = info[9]
                jointMaxForce = info[10]
                jointMaxVelocity = info[11]
                singleInfo = jointInfo(jointID, jointName, jointType, jointLowerLimit,
                                       jointUpperLimit, jointMaxForce, jointMaxVelocity)
                self.joints[singleInfo.name] = singleInfo
                # register index of dummy center link
                if jointName == "joint_yaw":
                    self.control_link_index = i
            self.position_control_joint_name = [
                "joint_x", "joint_y", "joint_z", "joint_roll", "joint_pitch", "joint_yaw"]
            self.left_finger_joint_name = 'joint_left_finger'
            self.right_finger_joint_name = 'joint_right_finger'

    def reset(self, output_id=None):
        self.steps = 0
        self.episode += 1

        self.objects.clear()
        self.if_open = 0

        self.selected_env = random.choice([1, 2, 3, 4, 5, 6, 7, 7, 8])
        # self.selected_env = random.choice([5, 6, 7, 8])
        if self.force_environment_id is not None:
            self.selected_env = self.force_environment_id

        self.build_environment()
        self.render_camera()

        RL_ID_MAP = {
            1:  44818,
            2: random.choice([37018, 45566, 42480]),
            3: random.choice([46034, 54582, 51496]),
            4: 51518,

            6: 47785,
            7: random.choice([47923, 52610, 69874]),

            9: 40737,
            10: 42090,

            30: 42488,
            31: 43841,

            12: 51518,
            13: 45911,

            15: 51518,
            16: 47785,

            18: 44818,
            19: 45566,
            20: 56481,
            21: 51518,
            22: 40737,
            23: 43997,

            25: 40737,
            26: 43997,
            27: 47785,
            28: random.choice([47923, 52610, 69874]),

            5: 1,
            8: 2,
            11: 3,
            32: 4,
            14: 5,
            17: 6,
            24: 7,
            29: 8,
        }
        self.sentence_id = RL_ID_MAP[self.case_id]

        return self.get_observation(0, 0, 0.1)

    def step(self, action):

        done = False
        self.steps += 1
        reward = 0
        observation = None

        new_x = (action[0]*2-20)*0.01
        new_y = (action[1]*2-20)*0.01
        new_z = 0.03
        roll = 0
        pitch = 0
        yaw = action[2]*90

        self.if_open = action[3]

        self.gripper_distance = 8
        self.gripper_orn_obs = gripper_orn_to_world(math.radians(pitch), math.radians(roll), math.radians(yaw))
        observation = self.get_observation(new_x, new_y, yaw)
        # %%build_pick_sth_into_drawer_env: caseid 1 2 3 4
        if self.case_id == 1:
            # two steps to open the drawer
            if self.steps == 1:
                done = False
                if self.test:
                    new_x = 0.1
                    new_y = 0.04
                    self.gripper_orn_obs = (0, 0, 0, 1)
                if self.benchmark == True:
                    if self.if_open > 0:
                        self.open_gripper(3.5)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(440)
                        self.control_gripper([new_x, new_y, 0.03], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                else:
                    # tx=0.1, ty=0.04: drawer handle is around this point
                    # reward = self.new_scale_reward(px=new_x, py=new_y, tx=0.1, ty=0.04) if abs(yaw) == 0 and self.if_open > 0 else -1
                    reward = 1 if self.check_in_circle(px=new_x, py=new_y, tx=0.1, ty=0.04, dis=0.013) and abs(yaw) == 0 and self.if_open > 0 else 0
                    if self.if_open > 0:
                        self.open_gripper(3.5)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(440)
                        self.control_gripper([new_x, new_y, 0.03], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)

            elif self.steps == 2:
                if self.test:
                    new_x = 0.1
                    new_y = 0.04 + 0.15
                    self.gripper_orn_obs = (0, 0, 0, 1)

                if self.benchmark == True:
                    if self.if_open > 0:
                        self.close_gripper()
                        self._delay(260)
                        self.control_gripper([new_x, new_y + 0.15, 0.03], self.gripper_orn_obs, velocity=.2, j_force=200)
                        self._delay(260)

                        drawer_position = p.getBasePositionAndOrientation(self.drawer_down)[0]
                        reward = 1 if drawer_position[1] > 0.02 else 0
                else:
                    # reward = self.new_scale_reward(px=new_x, py=new_y, tx=0.1, ty=0.18) if abs(yaw) == 0 and self.if_open > 0 else -1
                    reward = 1 if self.check_in_circle(px=new_x, py=new_y, tx=0.1, ty=0.18, dis=0.013) and abs(yaw) == 0 and self.if_open > 0 else 0
                    if self.if_open > 0:
                        self.close_gripper()
                        self._delay(260)
                        self.control_gripper([new_x, new_y + 0.15, 0.03], self.gripper_orn_obs, velocity=.2, j_force=200)
                        self._delay(260)

                        drawer_position = p.getBasePositionAndOrientation(self.drawer_down)[0]
                        if drawer_position[1] > 0.02:
                            reward = 1
                done = True

        if self.case_id == 2:
            # one step to grasp the object, success if it lift the object over 0.1m
            if self.steps == 1:
                if self.test:
                    new_x = -0.1
                    new_y = 0.0
                    self.gripper_orn_obs = (0, 0, 0, 1)
                if self.benchmark == True:
                    if self.if_open > 0:
                        self.open_gripper(8)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(440)
                        self.control_gripper([new_x, new_y, 0.01], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.close_gripper()
                        self._delay(300)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        objected_lifted = p.getBasePositionAndOrientation(self.objects[0])[0][2] > 0.1
                        reward = 1 if objected_lifted else 0
                else:
                    reward = self.get_scale_reward(new_x, new_y) if self.if_open > 0 else -1
                    if self.if_open > 0:
                        self.open_gripper(8)  # We need at least this wide to handle the cylinder.
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(440)
                        self.control_gripper([new_x, new_y, 0.01], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.close_gripper()
                        self._delay(300)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        objected_lifted = p.getBasePositionAndOrientation(self.objects[0])[0][2] > 0.1
                        if objected_lifted:
                            reward = 1
                done = True

        if self.case_id == 3:
            # place the object into the drawer.
            if self.steps == 1:
                if self.test:
                    new_x = 0.1
                    new_y = 0.04
                    self.gripper_orn_obs = (0, 0, 0, 1)
                if self.benchmark == True:
                    if self.if_open < 1:
                        self.close_gripper()
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(440)
                        self.control_gripper([new_x, new_y, 0.03], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.open_gripper(6.0)
                        self._delay(240)
                        reward = 1 if self.check_in_circle(px=new_x, py=new_y, tx=0.1, ty=-0.1, dis=0.013) and self.if_open < 1 else 0
                else:
                    # reward = self.new_scale_reward(px=new_x, py=new_y, tx=0.1, ty=-0.1) if self.if_open < 1 else -1
                    reward = 1 if self.check_in_circle(px=new_x, py=new_y, tx=0.1, ty=-0.1, dis=0.013) and self.if_open < 1 else 0
                    if self.if_open < 1:
                        self.close_gripper()
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(440)
                        self.control_gripper([new_x, new_y, 0.03], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.open_gripper(6.0)
                        self._delay(240)
                        reward = 1 if self.check_in_circle(px=new_x, py=new_y, tx=0.1, ty=-0.1, dis=0.013) and self.if_open < 1 else 0
                done = True

        if self.case_id == 4:
            # close the drawer
            if self.steps == 1:
                if self.test:
                    new_x = 0.1
                    new_y = 0.0 + 0.15
                    self.gripper_orn_obs = (0, 0, 0, 1)
                if self.benchmark == True:
                    if self.if_open > 0:
                        self.open_gripper(3.5)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.03], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                else:
                    # reward = self.new_scale_reward(px=new_x, py=new_y, tx=0.1, ty=0.15) if abs(yaw) == 0 and self.if_open > 0 else -1
                    reward = 1 if self.check_in_circle(px=new_x, py=new_y, tx=0.1, ty=0.15, dis=0.013) and abs(yaw) == 0 and self.if_open > 0 else 0
                    if self.if_open > 0:
                        self.open_gripper(3.5)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.03], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
            elif self.steps == 2:
                if self.test:
                    new_x = 0.1
                    new_y = 0.0
                    self.gripper_orn_obs = (0, 0, 0, 1)
                if self.benchmark == True:
                    if self.if_open > 0:
                        self.close_gripper()
                        self.control_gripper([new_x, new_y, 0.03], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        drawer_position = p.getBasePositionAndOrientation(self.drawer_down)[0]
                        reward = 1 if drawer_position[1] < -0.03 else 0
                else:
                    # reward = self.new_scale_reward(px=new_x, py=new_y, tx=0.1, ty=0) if abs(yaw) == 0 and self.if_open > 0 else -1
                    reward = 1 if self.check_in_circle(px=new_x, py=new_y, tx=0.1, ty=0, dis=0.013) and abs(yaw) == 0 and self.if_open > 0 else 0
                    if self.if_open > 0:
                        self.close_gripper()
                        self.control_gripper([new_x, new_y, 0.03], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        drawer_position = p.getBasePositionAndOrientation(self.drawer_down)[0]
                        if drawer_position[1] < -0.03:
                            reward = 1
                done = True

        # %%build_cut_sth_env: caseid 6, 7
        if self.case_id == 6:
            # grasp the knife
            if self.steps == 1:
                if self.test:
                    new_x = -0.08
                    new_y = -0.04
                    self.gripper_orn_obs = p.getQuaternionFromEuler([0, 0, math.pi/2])

                if self.benchmark == True:
                    if self.if_open > 0:
                        self.open_gripper(3.5)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.0], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.close_gripper()
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)

                        knife_lifted = p.getBasePositionAndOrientation(self.knife_id)[0][2] > 0.1
                        reward = 1 if knife_lifted else 0
                else:
                    knife_x = p.getBasePositionAndOrientation(self.knife_id)[0][0]
                    knife_y = p.getBasePositionAndOrientation(self.knife_id)[0][1] - 0.04
                    reward = 1 if self.check_in_circle(px=new_x, py=new_y, tx=knife_x, ty=knife_y,
                                                       dis=0.013) and abs(yaw) == 90 and self.if_open > 0 else 0
                    if self.if_open > 0:
                        self.open_gripper(3.5)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.0], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.close_gripper()
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)

                        knife_lifted = p.getBasePositionAndOrientation(self.knife_id)[0][2] > 0.1
                        if knife_lifted:
                            reward = 1
                done = True

        if self.case_id == 7:
            # cut the object
            if self.steps == 1:
                if self.test:
                    new_x = 0.08
                    new_y = -0.04
                    self.gripper_orn_obs = p.getQuaternionFromEuler([0, 0, math.pi/2])
                if self.benchmark == True:
                    if self.if_open > 0:
                        self.knife_id = p.loadURDF('./model/task_cut/obj/knife/knife_ivs.urdf',
                                                   [-0.075, 0, 0.01], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=False)
                        pick_knife_orientation = p.getQuaternionFromEuler([0, 0, math.pi/2])
                        pick_knife_position = [-0.08, -0.04]
                        self.open_gripper(3.5)
                        self.control_gripper([pick_knife_position[0], pick_knife_position[1], 0.2], pick_knife_orientation, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([pick_knife_position[0], pick_knife_position[1], 0.0], pick_knife_orientation, velocity=.2, j_force=100)
                        self._delay(240)
                        self.close_gripper()
                        self._delay(240)
                        self.control_gripper([pick_knife_position[0], pick_knife_position[1], 0.2], pick_knife_orientation, velocity=.2, j_force=100)
                        self._delay(240)

                        # Cut the object
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.0], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)

                        contact_points = p.getContactPoints(self.knife_id, self.objects[0])
                        reward = 1 if len(contact_points) > 0 else 0
                else:
                    reward = self.get_scale_reward(new_x, new_y+0.04) if abs(yaw) == 90 and self.if_open > 0 else -1
                    if self.if_open > 0:
                        self.knife_id = p.loadURDF('./model/task_cut/obj/knife/knife_ivs.urdf',
                                                   [-0.075, 0, 0.01], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=False)
                        pick_knife_orientation = p.getQuaternionFromEuler([0, 0, math.pi/2])
                        pick_knife_position = [-0.08, -0.04]
                        self.open_gripper(3.5)
                        self.control_gripper([pick_knife_position[0], pick_knife_position[1], 0.2], pick_knife_orientation, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([pick_knife_position[0], pick_knife_position[1], 0.0], pick_knife_orientation, velocity=.2, j_force=100)
                        self._delay(240)
                        self.close_gripper()
                        self._delay(240)
                        self.control_gripper([pick_knife_position[0], pick_knife_position[1], 0.2], pick_knife_orientation, velocity=.2, j_force=100)
                        self._delay(240)

                        # Cut the object
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.0], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)

                        contact_points = p.getContactPoints(self.knife_id, self.objects[0])
                        if len(contact_points) > 0:
                            reward = 1
                done = True

        # %%build_clean_table_env: caseid 9, 10
        if self.case_id == 9:
            # grasp an object from table
            if self.steps == 1:
                if self.test:
                    new_x = -0.05
                    new_y = 0.0
                    self.gripper_orn_obs = p.getQuaternionFromEuler([0, 0, math.pi/2])
                if self.benchmark == True:
                    if self.if_open > 0:
                        self.open_gripper(8)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.0], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.close_gripper()
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)

                        objects_lifted = [p.getBasePositionAndOrientation(object)[0][2] > 0.1 for object in self.objects]
                        reward = 1 if any(objects_lifted) else 0
                else:
                    reward = self.get_scale_reward(new_x, new_y) if self.if_open > 0 else -1
                    if self.if_open > 0:
                        self.open_gripper(8)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.0], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.close_gripper()
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)

                        objects_lifted = [p.getBasePositionAndOrientation(object)[0][2] > 0.1 for object in self.objects]
                        if any(objects_lifted):
                            reward = 1
                done = True

        if self.case_id == 10:
            # put the object into the bin
            if self.steps == 1:
                reward = 1 if self.check_in_circle(px=new_x, py=new_y, tx=-0.1, ty=0, dis=0.03) and self.if_open < 1 else -1
                done = True
        # %%build_pick_round_env: case: 30 31
        if self.case_id == 30:
            # grasp an round object
            if self.steps == 1:
                if self.test:
                    new_x = 0
                    new_y = 0.08
                    self.gripper_orn_obs = p.getQuaternionFromEuler([0, 0, math.pi/2])
                if self.benchmark == True:
                    if self.if_open > 0:
                        self.open_gripper(8)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.0], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.close_gripper()
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(440)

                        objects_lifted = [p.getBasePositionAndOrientation(object)[0][2] > 0.1 for object in self.objects]
                        reward = 1 if any(objects_lifted) else 0
                else:
                    reward = self.get_scale_reward(new_x, new_y) if self.if_open > 0 else -1
                    if self.if_open > 0:
                        self.open_gripper(8)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.0], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.close_gripper()
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(440)

                        objects_lifted = [p.getBasePositionAndOrientation(object)[0][2] > 0.1 for object in self.objects]
                        if any(objects_lifted):
                            reward = 1
                done = True

        if self.case_id == 31:
            # put the round object into the box
            if self.steps == 1:
                reward = 1 if self.check_in_circle(px=new_x, py=new_y, tx=-0.1, ty=0, dis=0.03) and self.if_open < 1 else -1
                done = True
        # %%multi1 env: case_id 12, 13
        if self.case_id == 12:
            # close the drawer
            if self.steps == 1:
                if self.test:
                    new_x = 0.1
                    new_y = 0.0 + 0.15
                    self.gripper_orn_obs = (0, 0, 0, 1)
                if self.benchmark == True:
                    if self.if_open > 0:
                        self.open_gripper(3.5)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.03], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                else:
                    # reward = self.new_scale_reward(px=new_x, py=new_y, tx=0.1, ty=0.15) if abs(yaw) == 0 and self.if_open > 0 else -1
                    reward = 1 if self.check_in_circle(px=new_x, py=new_y, tx=0.1, ty=0.15, dis=0.013) and abs(yaw) == 0 and self.if_open > 0 else 0
                    if self.if_open > 0:
                        self.open_gripper(3.5)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.03], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)

            elif self.steps == 2:
                if self.test:
                    new_x = 0.1
                    new_y = 0.0
                    self.gripper_orn_obs = (0, 0, 0, 1)
                if self.benchmark == True:
                    if self.if_open > 0:
                        self.close_gripper()
                        self.control_gripper([new_x, new_y, 0.03], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        drawer_position = p.getBasePositionAndOrientation(self.drawer_down)[0]
                        reward = 1 if drawer_position[1] < -0.03 else 0
                else:
                    # reward = self.new_scale_reward(px=new_x, py=new_y, tx=0.1, ty=0) if abs(yaw) == 0 and self.if_open > 0 else -1
                    reward = 1 if self.check_in_circle(px=new_x, py=new_y, tx=0.1, ty=0, dis=0.013) and abs(yaw) == 0 and self.if_open > 0 else 0
                    if self.if_open > 0:
                        self.close_gripper()
                        self.control_gripper([new_x, new_y, 0.03], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        drawer_position = p.getBasePositionAndOrientation(self.drawer_down)[0]
                        if drawer_position[1] < -0.03:
                            reward = 1
                done = True

        if self.case_id == 13:
            # grasp the apple
            if self.steps == 1:
                if self.test:
                    new_x = -0.05
                    new_y = 0.0
                    self.gripper_orn_obs = p.getQuaternionFromEuler([0, 0, math.pi/2])
                if self.benchmark == True:
                    if self.if_open > 0:
                        self.open_gripper(8)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.0], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.close_gripper()
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)

                        objects_lifted = [p.getBasePositionAndOrientation(object)[0][2] > 0.1 for object in self.objects]
                        reward = 1 if any(objects_lifted) else 0
                else:
                    reward = self.get_scale_reward(new_x, new_y) if self.if_open > 0 else -1
                    if self.if_open > 0:
                        self.open_gripper(8)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.0], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.close_gripper()
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)

                        objects_lifted = [p.getBasePositionAndOrientation(object)[0][2] > 0.1 for object in self.objects]
                        if any(objects_lifted):
                            reward = 1
                done = True
        # %%multi2 env: case_id 15, 16
        if self.case_id == 15:
            # close the drawer
            if self.steps == 1:
                if self.test:
                    new_x = 0.1
                    new_y = 0.0 + 0.15
                    self.gripper_orn_obs = (0, 0, 0, 1)
                if self.benchmark == True:
                    if self.if_open > 0:
                        self.open_gripper(3.5)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.03], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                else:
                    # reward = self.new_scale_reward(px=new_x, py=new_y, tx=0.1, ty=0.15) if abs(yaw) == 0 and self.if_open > 0 else -1
                    reward = 1 if self.check_in_circle(px=new_x, py=new_y, tx=0.1, ty=0.15, dis=0.013) and abs(yaw) == 0 and self.if_open > 0 else 0
                    if self.if_open > 0:
                        self.open_gripper(3.5)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.03], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)

            elif self.steps == 2:
                if self.test:
                    new_x = 0.1
                    new_y = 0.0
                    self.gripper_orn_obs = (0, 0, 0, 1)
                if self.benchmark == True:
                    if self.if_open > 0:
                        self.close_gripper()
                        self.control_gripper([new_x, new_y, 0.03], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        drawer_position = p.getBasePositionAndOrientation(self.drawer_down)[0]
                        reward = 1 if drawer_position[1] < -0.03 else 0
                else:
                    # reward = self.new_scale_reward(px=new_x, py=new_y, tx=0.1, ty=0) if abs(yaw) == 0 and self.if_open > 0 else -1
                    reward = 1 if self.check_in_circle(px=new_x, py=new_y, tx=0.1, ty=0, dis=0.013) and abs(yaw) == 0 and self.if_open > 0 else 0
                    if self.if_open > 0:
                        self.close_gripper()
                        self.control_gripper([new_x, new_y, 0.03], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        drawer_position = p.getBasePositionAndOrientation(self.drawer_down)[0]
                        if drawer_position[1] < -0.03:
                            reward = 1
                done = True

        if self.case_id == 16:
            # grasp the knife
            if self.steps == 1:
                if self.test:
                    new_x = -0.08
                    new_y = -0.04
                    self.gripper_orn_obs = p.getQuaternionFromEuler([0, 0, math.pi/2])

                if self.benchmark == True:
                    if self.if_open > 0:
                        self.open_gripper(3.5)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.0], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.close_gripper()
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)

                        knife_lifted = p.getBasePositionAndOrientation(self.knife_id)[0][2] > 0.1
                        reward = 1 if knife_lifted else 0
                else:
                    knife_x = p.getBasePositionAndOrientation(self.knife_id)[0][0]
                    knife_y = p.getBasePositionAndOrientation(self.knife_id)[0][1] - 0.04
                    reward = 1 if self.check_in_circle(px=new_x, py=new_y, tx=knife_x, ty=knife_y,
                                                       dis=0.013) and abs(yaw) == 90 and self.if_open > 0 else 0
                    if self.if_open > 0:
                        self.open_gripper(3.5)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.0], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.close_gripper()
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        knife_lifted = p.getBasePositionAndOrientation(self.knife_id)[0][2] > 0.1
                        if knife_lifted:
                            reward = 1

                done = True

        # %%multi3 env: case_id 18, 19, 20, 21, 22, 23
        if self.case_id == 18:
            # two steps to open the drawer
            if self.steps == 1:
                done = False
                if self.test:
                    new_x = 0
                    new_y = 0.04
                    self.gripper_orn_obs = (0, 0, 0, 1)
                if self.benchmark == True:
                    if self.if_open > 0:
                        self.open_gripper(3.5)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(440)
                        self.control_gripper([new_x, new_y, 0.03], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                else:
                    if self.if_open > 0:
                        self.open_gripper(3.5)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(440)
                        self.control_gripper([new_x, new_y, 0.03], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                    reward = 1 if self.check_in_circle(px=new_x, py=new_y, tx=0, ty=0.04, dis=0.013) and abs(yaw) == 0 and self.if_open > 0 else 0

            elif self.steps == 2:
                if self.test:
                    new_x = 0
                    new_y = 0.04 + 0.15
                    self.gripper_orn_obs = (0, 0, 0, 1)

                if self.benchmark == True:
                    if self.if_open > 0:
                        self.close_gripper()
                        self._delay(260)
                        self.control_gripper([new_x, new_y + 0.15, 0.03], self.gripper_orn_obs, velocity=.2, j_force=200)
                        self._delay(260)

                        drawer_position = p.getBasePositionAndOrientation(self.drawer_down)[0]
                        reward = 1 if drawer_position[1] > 0.02 else 0
                else:
                    reward = 1 if self.check_in_circle(px=new_x, py=new_y, tx=0, ty=0.18, dis=0.013) and abs(yaw) == 0 and self.if_open > 0 else 0
                    if self.if_open > 0:
                        self.close_gripper()
                        self._delay(260)
                        self.control_gripper([new_x, new_y + 0.15, 0.03], self.gripper_orn_obs, velocity=.2, j_force=200)
                        self._delay(260)

                        drawer_position = p.getBasePositionAndOrientation(self.drawer_down)[0]
                        if drawer_position[1] > 0.02:
                            reward = 1
                done = True

        if self.case_id == 19:
            # one step to grasp the object, success if it lift the object over 0.1m
            if self.steps == 1:
                if self.test:
                    new_x = -0.1
                    new_y = 0.0
                    self.gripper_orn_obs = (0, 0, 0, 1)
                if self.benchmark == True:
                    if self.if_open > 0:
                        self.open_gripper(8)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(440)
                        self.control_gripper([new_x, new_y, 0.01], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.close_gripper()
                        self._delay(300)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        objected_lifted = p.getBasePositionAndOrientation(self.objects[0])[0][2] > 0.1
                        reward = 1 if objected_lifted else 0
                else:
                    reward = self.get_scale_reward(new_x, new_y) if self.if_open > 0 else -1
                    if self.if_open > 0:
                        self.open_gripper(8)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(440)
                        self.control_gripper([new_x, new_y, 0.01], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.close_gripper()
                        self._delay(300)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        objected_lifted = p.getBasePositionAndOrientation(self.objects[0])[0][2] > 0.1
                        if objected_lifted:
                            reward = 1
                done = True

        if self.case_id == 20:
            # place the object into the drawer.
            if self.steps == 1:
                if self.test:
                    new_x = 0
                    new_y = 0.04
                    self.gripper_orn_obs = (0, 0, 0, 1)
                if self.benchmark == True:
                    if self.if_open < 1:
                        self.close_gripper()
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(440)
                        self.control_gripper([new_x, new_y, 0.03], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.open_gripper(6.0)
                        self._delay(240)
                    reward = 1 if self.check_in_circle(px=new_x, py=new_y, tx=0, ty=-0.1, dis=0.013) and self.if_open < 1 else 0
                else:
                    if self.if_open < 1:
                        self.close_gripper()
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(440)
                        self.control_gripper([new_x, new_y, 0.03], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.open_gripper(6.0)
                        self._delay(240)
                    reward = 1 if self.check_in_circle(px=new_x, py=new_y, tx=0, ty=-0.1, dis=0.013) and self.if_open < 1 else 0
                done = True

        if self.case_id == 21:
            # close the drawer
            if self.steps == 1:
                if self.test:
                    new_x = 0
                    new_y = 0.0 + 0.15
                    self.gripper_orn_obs = (0, 0, 0, 1)
                if self.benchmark == True:
                    if self.if_open > 0:
                        self.open_gripper(3.5)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.03], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                else:
                    if self.if_open > 0:
                        self.open_gripper(3.5)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.03], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                    reward = 1 if self.check_in_circle(px=new_x, py=new_y, tx=0, ty=0.15, dis=0.013) and abs(yaw) == 0 and self.if_open > 0 else 0

            elif self.steps == 2:
                if self.test:
                    new_x = 0
                    new_y = 0.0
                    self.gripper_orn_obs = (0, 0, 0, 1)
                if self.benchmark == True:
                    if self.if_open > 0:
                        self.close_gripper()
                        self.control_gripper([new_x, new_y, 0.03], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        drawer_position = p.getBasePositionAndOrientation(self.drawer_down)[0]
                        reward = 1 if drawer_position[1] < -0.03 else 0
                else:
                    reward = 1 if self.check_in_circle(px=new_x, py=new_y, tx=0, ty=0, dis=0.013) and abs(yaw) == 0 and self.if_open > 0 else 0
                    if self.if_open > 0:
                        self.close_gripper()
                        self.control_gripper([new_x, new_y, 0.03], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        drawer_position = p.getBasePositionAndOrientation(self.drawer_down)[0]
                        if drawer_position[1] < -0.03:
                            reward = 1
                done = True

        if self.case_id == 22:
            # grasp an object from table
            if self.steps == 1:
                if self.test:
                    new_x = -0.05
                    new_y = 0.0
                    self.gripper_orn_obs = p.getQuaternionFromEuler([0, 0, math.pi/2])
                if self.benchmark == True:
                    if self.if_open > 0:
                        self.open_gripper(8)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.0], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.close_gripper()
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)

                        objects_lifted = [p.getBasePositionAndOrientation(object)[0][2] > 0.1 for object in self.objects]
                        reward = 1 if any(objects_lifted) else 0
                else:
                    reward = self.get_scale_reward(new_x, new_y) if self.if_open > 0 else -1
                    if self.if_open > 0:
                        self.open_gripper(8)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.0], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.close_gripper()
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)

                        objects_lifted = [p.getBasePositionAndOrientation(object)[0][2] > 0.1 for object in self.objects]
                        if any(objects_lifted):
                            reward = 1
                done = True

        if self.case_id == 23:
            # put the object into the bin
            if self.steps == 1:
                reward = 1 if self.check_in_circle(px=new_x, py=new_y, tx=-0.1, ty=0, dis=0.03) and self.if_open < 1 else -1
                done = True
        # %%multi4 env: case_id 25, 26, 27, 28
        if self.case_id == 25:
            # grasp an object from table
            if self.steps == 1:
                if self.test:
                    new_x = -0.05
                    new_y = 0.0
                    self.gripper_orn_obs = p.getQuaternionFromEuler([0, 0, math.pi/2])
                if self.benchmark == True:
                    if self.if_open > 0:
                        self.open_gripper(8)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.0], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.close_gripper()
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)

                        objects_lifted = [p.getBasePositionAndOrientation(object)[0][2] > 0.1 for object in self.objects]
                        reward = 1 if any(objects_lifted) else 0
                else:
                    reward = self.get_scale_reward(new_x, new_y) if self.if_open > 0 else -1
                    if self.if_open > 0:
                        self.open_gripper(8)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.0], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.close_gripper()
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)

                        objects_lifted = [p.getBasePositionAndOrientation(object)[0][2] > 0.1 for object in self.objects]
                        if any(objects_lifted):
                            reward = 1
                done = True

        if self.case_id == 26:
            # put the object into the bin
            if self.steps == 1:
                reward = 1 if self.check_in_circle(px=new_x, py=new_y, tx=-0.1, ty=0, dis=0.03) and self.if_open < 1 else -1
                done = True
        if self.case_id == 27:
            # grasp the knife
            if self.steps == 1:
                if self.test:
                    new_x = -0.08
                    new_y = -0.04
                    self.gripper_orn_obs = p.getQuaternionFromEuler([0, 0, math.pi/2])

                if self.benchmark == True:
                    if self.if_open > 0:
                        self.open_gripper(3.5)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.0], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.close_gripper()
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)

                        knife_lifted = p.getBasePositionAndOrientation(self.knife_id)[0][2] > 0.1
                        reward = 1 if knife_lifted else 0
                else:
                    knife_x = p.getBasePositionAndOrientation(self.knife_id)[0][0]
                    knife_y = p.getBasePositionAndOrientation(self.knife_id)[0][1] - 0.04
                    reward = 1 if self.check_in_circle(px=new_x, py=new_y, tx=knife_x, ty=knife_y,
                                                       dis=0.013) and abs(yaw) == 90 and self.if_open > 0 else 0
                    if self.if_open > 0:
                        self.open_gripper(3.5)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.0], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.close_gripper()
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)

                        knife_lifted = p.getBasePositionAndOrientation(self.knife_id)[0][2] > 0.1
                        if knife_lifted is True:
                            reward = 1
                done = True

        if self.case_id == 28:
            # cut the object
            if self.steps == 1:
                if self.test:
                    new_x = 0.08
                    new_y = -0.04
                    self.gripper_orn_obs = p.getQuaternionFromEuler([0, 0, math.pi/2])
                if self.benchmark == True:
                    if self.if_open > 0:
                        self.knife_id = p.loadURDF('./model/task_cut/obj/knife/knife_ivs.urdf',
                                                   [-0.075, 0, 0.01], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=False)
                        pick_knife_orientation = p.getQuaternionFromEuler([0, 0, math.pi/2])
                        pick_knife_position = [-0.08, -0.04]
                        self.open_gripper(3.5)
                        self.control_gripper([pick_knife_position[0], pick_knife_position[1], 0.2], pick_knife_orientation, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([pick_knife_position[0], pick_knife_position[1], 0.0], pick_knife_orientation, velocity=.2, j_force=100)
                        self._delay(240)
                        self.close_gripper()
                        self._delay(240)
                        self.control_gripper([pick_knife_position[0], pick_knife_position[1], 0.2], pick_knife_orientation, velocity=.2, j_force=100)
                        self._delay(240)

                        # Cut the object
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.0], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)

                        contact_points = p.getContactPoints(self.knife_id, self.objects[0])
                        reward = 1 if len(contact_points) > 0 else 0
                else:
                    reward = self.get_scale_reward(new_x, new_y+0.04) if abs(yaw) == 90 and self.if_open > 0 else -1
                    if self.if_open > 0:
                        self.knife_id = p.loadURDF('./model/task_cut/obj/knife/knife_ivs.urdf',
                                                   [-0.075, 0, 0.01], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=False)
                        pick_knife_orientation = p.getQuaternionFromEuler([0, 0, math.pi/2])
                        pick_knife_position = [-0.08, -0.04]
                        self.open_gripper(3.5)
                        self.control_gripper([pick_knife_position[0], pick_knife_position[1], 0.2], pick_knife_orientation, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([pick_knife_position[0], pick_knife_position[1], 0.0], pick_knife_orientation, velocity=.2, j_force=100)
                        self._delay(240)
                        self.close_gripper()
                        self._delay(240)
                        self.control_gripper([pick_knife_position[0], pick_knife_position[1], 0.2], pick_knife_orientation, velocity=.2, j_force=100)
                        self._delay(240)

                        # Cut the object
                        self.control_gripper([new_x, new_y, 0.2], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)
                        self.control_gripper([new_x, new_y, 0.0], self.gripper_orn_obs, velocity=.2, j_force=100)
                        self._delay(240)

                        contact_points = p.getContactPoints(self.knife_id, self.objects[0])
                        if len(contact_points) > 0:
                            reward = 1
                done = True

        return observation, reward, done, {}
# %%

    def check_in_circle(self, px, py, tx, ty, dis):
        in_circle = False
        if (tx-px)*(tx-px)+(ty-py)*(ty-py) < dis*dis:
            in_circle = True
        return in_circle

    def new_scale_reward(self, px, py, tx, ty):
        reward = normalize_m11(min(max(math.sqrt((px-tx)*(px-tx)+(py-ty)*(py-ty)), 0), 0.06), gripper_limits['o_g_dis'])
        return reward

    def get_scale_reward(self, new_x, new_y):
        reward = -10
        for object_id in self.objects:
            object_position, _ = p.getBasePositionAndOrientation(object_id)
            if reward < normalize_m11(min(max(math.sqrt((object_position[0]-new_x)*(object_position[0]-new_x)+(object_position[1]-new_y)*(object_position[1]-new_y)), 0), 0.06), gripper_limits['o_g_dis']):
                reward = normalize_m11(min(max(math.sqrt((object_position[0]-new_x)*(object_position[0]-new_x)+(
                    object_position[1]-new_y)*(object_position[1]-new_y)), 0), 0.06), gripper_limits['o_g_dis'])

        total_picked = 0
        for object_id in self.objects:
            object_position, _ = p.getBasePositionAndOrientation(object_id)
            if self.check_in_circle(px=new_x, py=new_y, tx=object_position[0], ty=object_position[1], dis=0.0125) == True:
                self.objects.remove(object_id)
                p.removeBody(object_id)
                total_picked += 1
        # any object is picked is fine
        if total_picked >= 1:
            reward = 1

        return reward

    def get_reward(self):
        reward = 0
        total_picked = 0
        for object_id in self.objects:
            object_position, _ = p.getBasePositionAndOrientation(object_id)
            if object_position[2] > self.reward_height:
                self.objects.remove(object_id)
                p.removeBody(object_id)
                total_picked += 1
        # any object is picked is fine
        if total_picked == 1:
            reward += 10
        return reward

    def get_robotid(self):
        return self.robot_id, self.gripper_urdf, self.table_id

    def lift_gripper(self, distance=0.15, initial_orientation=None):
        state = p.getLinkState(self.robot_id, self.joints['joint_yaw'].id)
        robot_pos = state[0]
        robot_orn = state[1] if initial_orientation is None else initial_orientation
        jointPose = p.calculateInverseKinematics(self.robot_id,
                                                 self.control_link_index,
                                                 [robot_pos[0], robot_pos[1],
                                                     robot_pos[2]+distance],
                                                 robot_orn)
        for jointName in self.joints:
            if jointName in self.position_control_joint_name:
                joint = self.joints[jointName]
                p.setJointMotorControl2(self.robot_id, joint.id, p.POSITION_CONTROL,
                                        targetPosition=jointPose[joint.id], force=joint.maxForce,
                                        maxVelocity=0.15)

    def get_center_pose(self):
        state = p.getLinkState(self.robot_id, self.joints['joint_yaw'].id)
        return state[0], state[1]

    def control_gripper(self, world_position, world_orientation, velocity=None, j_force=None):
        jointPose = p.calculateInverseKinematics(self.robot_id,
                                                 self.control_link_index,
                                                 world_position,
                                                 world_orientation)
        for jointName in self.joints:
            if jointName in self.position_control_joint_name:
                joint = self.joints[jointName]
                joint_velocity = joint.maxVelocity if velocity is None else velocity
                joint_force = joint.maxForce if j_force is None else j_force
                if jointName in ['joint_yaw', 'joint_pitch', 'joint_roll']:
                    joint_velocity = joint.maxVelocity
                    joint_force = joint.maxForce
                p.setJointMotorControl2(self.robot_id, joint.id, p.POSITION_CONTROL,
                                        targetPosition=jointPose[joint.id], force=joint_force,
                                        maxVelocity=joint_velocity)

    def close_gripper(self):
        # 1mm distace
        p.setJointMotorControl2(self.robot_id, self.joints[self.right_finger_joint_name].id, p.POSITION_CONTROL,
                                targetPosition=-0.001,
                                force=70,
                                maxVelocity=0.025)
        p.setJointMotorControl2(self.robot_id, self.joints[self.left_finger_joint_name].id, p.POSITION_CONTROL,
                                targetPosition=0.001,
                                force=70,
                                maxVelocity=0.025)

    def open_gripper(self, ditance_between_fingers):
        # ditance_between_fingers unit is cm
        ditance_between_fingers = ditance_between_fingers/100/2
        p.setJointMotorControl2(self.robot_id, self.joints[self.right_finger_joint_name].id, p.POSITION_CONTROL,
                                targetPosition=-ditance_between_fingers,
                                force=500,
                                maxVelocity=.1)
        p.setJointMotorControl2(self.robot_id, self.joints[self.left_finger_joint_name].id, p.POSITION_CONTROL,
                                targetPosition=ditance_between_fingers,
                                force=500,
                                maxVelocity=.1)
