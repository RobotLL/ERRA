import numpy as np
import random
import pybullet
import pybullet as p
import math3d as m3d
from scipy.spatial.transform import Rotation as R
import math


def render_camera(rl=False):

    camera_position = [0, 0, 0.4]
    camera_target = [0, 0, 0]
    camera_fov = 55.371784673413806
    camera_position[0] = camera_position[0] + random.uniform(-1, 1)*0.005
    camera_position[1] = camera_position[1] + random.uniform(-1, 1)*0.005
    camera_position[2] = camera_position[2] + random.uniform(-1, 1)*0.0025
    camera_target[0] = camera_target[0] + random.uniform(-1, 1)*0.005
    camera_target[1] = camera_target[1] + random.uniform(-1, 1)*0.005
    camera_fov = 55.371784673413806 + random.uniform(-0.025, 0.025)

    if rl == False:
        IMAGE_WIDTH = 320
        IMAGE_HEIGHT = 240
    else:
        IMAGE_WIDTH = 160
        IMAGE_HEIGHT = 120
    CAMERA_FAR = 1
    CAMERA_NEAR = 0.19
    HFOV_VFOV = 320/240
    UpVector = [0, 1, 0]

    view_matrix = p.computeViewMatrix(cameraEyePosition=camera_position,
                                      cameraTargetPosition=camera_target,
                                      cameraUpVector=UpVector)
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=camera_fov, aspect=HFOV_VFOV, nearVal=CAMERA_NEAR, farVal=CAMERA_FAR)
    _, _, _, depth_image, mask = p.getCameraImage(
        width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    # Depth image
    depth_image = np.array(depth_image).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
    mask = np.array(mask).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
    depth_image = CAMERA_FAR * CAMERA_NEAR / \
        (CAMERA_FAR - (CAMERA_FAR - CAMERA_NEAR) * depth_image)
    depth_image = np.array(depth_image).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
    sim_float = depth_image.copy()

    min_in_d = 0.2
    max_in_d = 0.4
    process_depth = depth_image.copy()
    process_depth = (process_depth-min_in_d)/(max_in_d-min_in_d)*255
    process_depth[np.where(sim_float > 0.39)] = 255
    process_depth = np.array(process_depth).astype(np.uint8)

    return process_depth, sim_float, mask


# limit
gripper_limits = {'joint_x': [-0.25, 0.25],
                  'joint_y': [-0.25, 0.25],
                  'joint_z': [0, 0.1],
                  'joint_roll': [-45, 45],
                  'joint_pitch': [-45, 45],
                  'joint_yaw': [-90, 90],
                  'joint_open': [0, 1],
                  'joint_distance': [0, 0.3],
                  'steps': [0, 4],
                  'case_id': [0, 50],
                  'o_g_dis': [0.05, 0.01]}

objects_limits = {'o_x': [-0.07, 0.07],
                  'o_y': [-0.07, 0.07],
                  'o_z': [0, 0.1],
                  'o_roll': [-1.57, 1.57],
                  'o_pitch': [-1.57, 1.57],
                  'o_yaw': [-1.57, 1.57]}


def discretize(value, possibilities):
    closest_value = possibilities[0]
    for i in range(len(possibilities)):
        if abs(value - possibilities[i]) < abs(value - closest_value):
            closest_value = possibilities[i]
    return closest_value


def rescale(x, x_min, x_max, y_min, y_max):
    return (x - x_min) * (y_max - y_min) / (x_max - x_min) + y_min


def normalize_m11(x, old_range):
    return rescale(x, old_range[0], old_range[1], -1, 1)


def normalize_01(x, old_range):
    return rescale(x, old_range[0], old_range[1], 0, 1)


def unnormalize_01(x, new_range):
    return rescale(x, 0, 1, new_range[0], new_range[1])


def unnormalize(x, new_range):
    return rescale(x, -1, 1, new_range[0], new_range[1])


def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def T_rotatez(rx, ry, rz, yaw):
    grip_rot = m3d.Transform()
    grip_rot.pos = (0, 0, 0)
    grip_rot.orient.rotate_zb(yaw)  # yaw
    grip_matrix = grip_rot.get_matrix()
    gripper_base_start_pos = np.array([rx, ry, rz, 1]).reshape(4, 1)
    g_tool = np.matmul(grip_matrix, gripper_base_start_pos)
    return g_tool[0:3].tolist()


def gripper_orn_to_world(pitch, roll, yaw):
    grip_rot = m3d.Transform()
    grip_rot.pos = (0, 0, 0)
    grip_rot.orient.rotate_yb(roll)  # roll
    grip_rot.orient.rotate_xb(pitch)  # pitch
    grip_rot.orient.rotate_zb(yaw)  # yaw
    grip_matrix = grip_rot.get_matrix()
    robot_Orn = R.from_matrix(grip_matrix[:3, :3]).as_quat()
    return robot_Orn


def gfinger_T_gbase_next(long_finger_pos, long_finger_orn, finger_distance):
    finger_distance = finger_distance/2/100
    long_finger_start_pos = [0, finger_distance, 0]
    long_finger_start_pos_inverse = [0, -finger_distance, 0]
    long_finger_start_orn = np.array([0, 0, 0, 1])

    long_finger_pos = np.array(long_finger_pos)
    long_finger_orn = np.array(long_finger_orn)

    diff_pos = long_finger_pos - long_finger_start_pos
    gripper_base_start_pos = np.array([0, 0, 0, 1]).reshape(4, 1)

    T_gbase_lnext_pos, T_gbase_lnext_orn = p.multiplyTransforms(
        long_finger_start_pos, long_finger_start_orn, diff_pos, long_finger_orn)

    gTgn_pos, gTgn_orn = p.multiplyTransforms(
        T_gbase_lnext_pos, T_gbase_lnext_orn, long_finger_start_pos_inverse, long_finger_start_orn)

    SO3_gTgn_orn = np.array(p.getMatrixFromQuaternion(gTgn_orn)).reshape(3, 3)
    SE3_gTgn = np.hstack([SO3_gTgn_orn, np.array(gTgn_pos).reshape(3, 1)])
    SE3_gTgn = np.vstack([SE3_gTgn, np.array([0, 0, 0, 1]).reshape(1, 4)])
    gbase_next = np.matmul(SE3_gTgn, gripper_base_start_pos)
    return gbase_next[0:3].tolist()


def rotate_point(point, theta):
    return np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]
    ]) @ point


def is_point_below_line(point, line_point, line_angle, n_offset=0):
    k = math.tan(line_angle)
    n = line_point[1] - k * line_point[0] + n_offset
    return point[1] < k * point[0] + n


def is_point_inside_rectangle(point, center, width, length, orientation):
    point = np.array(point)
    center = np.array(center)
    orientation = np.array(orientation)
    point_rot = rotate_point(point - center, -orientation)
    return point_rot[0] > -length/2 and point_rot[0] < length/2 and point_rot[1] > -width/2 and point_rot[1] < width/2


def normalize_angle(theta):
    if theta > math.pi:
        theta -= 2 * math.pi
    elif theta < -math.pi:
        theta += 2 * math.pi
    return theta


def are_angles_close(theta1, theta2, threshold):
    return abs(normalize_angle(theta1) - normalize_angle(theta2)) < threshold


def generate_td_obj(obj_position, obj_orientation):
    # yellow_color = [0.949, 0.878, 0.0392, 1.0]

    obj_vw = .06/2 * random.uniform(0.5, 1)
    obj_vh = .06/2 * random.uniform(0.5, 1)
    obj_vd = .1/2 * random.uniform(0.2, 1)
    obj_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[obj_vw, obj_vh, obj_vd])

    obj_cw = .06/2 * random.uniform(0.5, 1)
    obj_ch = .06/2 * random.uniform(0.5, 1)
    obj_cd = .1/2 * random.uniform(0.2, 1)
    obj_c = p.createCollisionShape(p.GEOM_BOX, halfExtents=[obj_cw, obj_ch, obj_cd])

    mass = 0.01
    obj_id = p.createMultiBody(mass, obj_c, obj_v, obj_position, obj_orientation)
    # p.changeVisualShape (obj_id, -1, rgbaColor=yellow_color,specularColor=[1.,1.,1.])

    obj_friction_ceof = 1
    p.changeDynamics(obj_id, -1, lateralFriction=obj_friction_ceof)
    # p.changeDynamics(obj_id, -1, rollingFriction=obj_friction_ceof)
    # p.changeDynamics(obj_id, -1, spinningFriction=obj_friction_ceof)
    return obj_id


def generate_roll_obj(obj_position, obj_orientation):
    # yellow_color = [0.949, 0.878, 0.0392, 1.0]

    radius = 0.03 * random.uniform(0.8, 1)
    height = 0.1 * random.uniform(0.4, 1)
    obj_v = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height)
    obj_c = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)

    mass = 0.01
    obj_id = p.createMultiBody(mass, obj_c, obj_v, obj_position, obj_orientation)
    # p.changeVisualShape (obj_id, -1, rgbaColor=yellow_color,specularColor=[1.,1.,1.])

    obj_friction_ceof = 1
    p.changeDynamics(obj_id, -1, lateralFriction=obj_friction_ceof)
    # p.changeDynamics(obj_id, -1, rollingFriction=obj_friction_ceof)
    # p.changeDynamics(obj_id, -1, spinningFriction=obj_friction_ceof)
    return obj_id
