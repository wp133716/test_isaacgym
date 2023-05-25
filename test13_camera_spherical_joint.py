"""
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Spherical Joint
------------
- Demonstrates usage of spherical joints.
"""

import math
import numpy as np
from isaacgym import gymapi, gymutil
from isaacgym import gymtorch
import torch

from scipy.spatial.transform import Rotation as R
import time

# 设置torch打印格式，保留5位小数，不用科学计数法
torch.set_printoptions(precision=3, sci_mode=False)
np.set_printoptions(precision=3, suppress=True)

def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

# simple asset descriptor for selecting from a list


class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments


asset_descriptors = [
    AssetDesc("urdf/dof_spherical_joint_test.urdf", False),
    # AssetDesc("mjcf/spherical_joint.xml", False),
]

# parse arguments
args = gymutil.parse_arguments(
    description="Spherical Joint: Show example of controlling a spherical joint robot.",
)

# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
sim_params.gravity = gymapi.Vec3(0, 0, 0)
sim_params.up_axis = gymapi.UP_AXIS_Z

if args.physics_engine == gymapi.SIM_FLEX:
    pass
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0 # Ground plane distance from origin
# plane_params.static_friction = 1 # Coefficient of dynamic friction
# plane_params.dynamic_friction = 1 # Coefficient of static friction
# plane_params.restitution = 0 # Coefficient of restitution
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load asset
asset_root = "./assets"
asset_file = asset_descriptors[0].file_name

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = asset_descriptors[0].flip_visual_attachments
asset_options.use_mesh_materials = True

print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# get array of DOF names
dof_names = gym.get_asset_dof_names(asset)

# get array of DOF properties
dof_props = gym.get_asset_dof_properties(asset)

# create an array of DOF states that will be used to update the actors
num_dofs = gym.get_asset_dof_count(asset)
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

# get list of DOF types
dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]

# get the position slice of the DOF state array
dof_positions = dof_states['pos']

# get the limit-related slices of the DOF properties array
stiffnesses = dof_props['stiffness']
dampings = dof_props['damping']
armatures = dof_props['armature']
has_limits = dof_props['hasLimits']
lower_limits = dof_props['lower']
upper_limits = dof_props['upper']

# initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
defaults = np.zeros(num_dofs)
for i in range(num_dofs):
    if has_limits[i]:
        if dof_types[i] == gymapi.DOF_ROTATION:
            lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
            upper_limits[i] = clamp(upper_limits[i], -math.pi, math.pi)
        # make sure our default position is in range
        if lower_limits[i] > 0.0:
            defaults[i] = lower_limits[i]
        elif upper_limits[i] < 0.0:
            defaults[i] = upper_limits[i]
    else:
        # set reasonable animation limits for unlimited joints
        if dof_types[i] == gymapi.DOF_ROTATION:
            # unlimited revolute joint
            lower_limits[i] = -math.pi
            upper_limits[i] = math.pi
        elif dof_types[i] == gymapi.DOF_TRANSLATION:
            # unlimited prismatic joint
            lower_limits[i] = -1.0
            upper_limits[i] = 1.0
        else:
            print("Unknown DOF type!")
            exit()
    # set DOF position to default
    dof_positions[i] = defaults[i]

# Print DOF properties
for i in range(num_dofs):
    print("DOF %d" % i)
    print("  Name:     '%s'" % dof_names[i])
    print("  Type:     %s" % gym.get_dof_type_string(dof_types[i]))
    print("  Stiffness:  %r" % stiffnesses[i])
    print("  Damping:  %r" % dampings[i])
    print("  Armature:  %r" % armatures[i])
    print("  Limited?  %r" % has_limits[i])
    if has_limits[i]:
        print("    Lower   %f" % lower_limits[i])
        print("    Upper   %f" % upper_limits[i])

# set up the env grid
num_envs = 1
num_per_row = 6
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(1.0, 1.0, 1)
cam_target = gymapi.Vec3(0, 0, 1)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# cache useful handles
envs = []
actor_handles = []

print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
    # pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

    props = gym.get_actor_dof_properties(env, actor_handle)
    props["driveMode"] = (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS)
    # props["driveMode"] = (gymapi.DOF_MODE_VEL, gymapi.DOF_MODE_VEL, gymapi.DOF_MODE_VEL, gymapi.DOF_MODE_VEL, gymapi.DOF_MODE_VEL, gymapi.DOF_MODE_VEL)
    props["stiffness"] = (50.0, 50.0, 50.0, 50.0, 50.0, 50.0)
    # props["damping"] = (10.0, 10.0, 10.0, 100.0, 100.0, 100.0)
    # props["damping"] = (10.0, 10.0, 10.0, 1000.0, 1000.0, 1000.0)
    props["damping"] = (5, 5, 5, 5, 5, 5)

    gym.set_actor_dof_properties(env, actor_handle, props)

    # set default DOF positions
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)


def random_quaternion():
    """Random quaternion of the form (x, y, z, w).

    Returns:
        np.ndarray: 4-element array.
    """
    r1, r2, r3 = np.random.random(3)

    q1 = math.sqrt(1.0 - r1) * (math.sin(2 * math.pi * r2))
    q2 = math.sqrt(1.0 - r1) * (math.cos(2 * math.pi * r2))
    q3 = math.sqrt(r1) * (math.sin(2 * math.pi * r3))
    q4 = math.sqrt(r1) * (math.cos(2 * math.pi * r3))

    quat_xyzw = np.array([q2, q3, q4, q1])

    if quat_xyzw[-1] < 0:
        quat_xyzw = -quat_xyzw

    return quat_xyzw

def quat2expcoord(q):
    """Converts quaternion to exponential coordinates.

    Args:
        q (np.ndarray): Quaternion as a 4-element array of the form [x, y, z, w].

    Returns:
        np.ndarray: Exponential coordinate as 3-element array.
    """
    if (q[-1] < 0):
        q = -q

    theta = 2. * math.atan2(np.linalg.norm(q[:-1]), q[-1])
    # w = (1. / (np.sin(theta/2.0))) * q[:-1]
    w = (1. / (np.sin(theta/2.0)+1e-7)) * q[:-1]

    return w * theta


# Helper visualization for goal orientation
axes_geom = gymutil.AxesGeometry(0.5)

dof_state_buffer = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
dof_force_buffer = gymtorch.wrap_tensor(gym.acquire_dof_force_tensor(sim))
gym.refresh_dof_state_tensor(sim)
gym.refresh_actor_root_state_tensor(sim)
gym.refresh_rigid_body_state_tensor(sim)
dof_pos_buffer = dof_state_buffer.view(num_envs, num_dofs, 2)[..., 0]
dof_vel_buffer = dof_state_buffer.view(num_envs, num_dofs, 2)[..., 1]

default_dof_pos = torch.zeros_like(dof_pos_buffer)

cnt = 0
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.refresh_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)

    # Set new goal orientation
    if cnt % 1 == 0:
        goal_quat = random_quaternion()

        roll = np.clip(cnt%180-90, -90, 90)
        pitch = np.clip(cnt%180, 0, 120)
        yaw = np.clip(cnt%360-180, -180, 180)
        # euler = [roll, 0, 0]
        # euler = [0, pitch, 0]
        euler = [0, 0, yaw]
        # euler = [roll, pitch, yaw]
        goal_quat = R.from_euler('xyz', euler, degrees=True).as_quat()

        print("New goal orientation:", goal_quat)

        gym.clear_lines(viewer)

        goal_viz_T = gymapi.Transform(r=gymapi.Quat(*goal_quat), p=gymapi.Vec3(0, 0, 1.0))
        gymutil.draw_lines(axes_geom, gym, viewer, env, goal_viz_T)

        dof_positions[:] = 0.0
        dof_positions[3:] = quat2expcoord(goal_quat)
        print("New goal DOF positions:", dof_positions)
        
        dof_velocities = dof_states['vel']
        # dof_velocities[3] = 10.0
        # dof_velocities[4] = 10.0
        # dof_velocities[5] = 10.0
        print("New goal DOF vel:", dof_states['vel'])

        default_dof_pos[0] = torch.tensor(dof_positions) 
        print("New goal DOF positions:", dof_pos_buffer[:, 5])
        print("New goal DOF positions:", dof_positions[5])
        print("New goal DOF positions:", dof_pos_buffer)
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(default_dof_pos))

        # for i in range(num_envs):
        #     if props["driveMode"][0]==gymapi.DOF_MODE_POS:
        #         gym.set_actor_dof_position_targets(envs[i], actor_handles[i], dof_positions)
        #     elif props["driveMode"][0]==gymapi.DOF_MODE_VEL:
        #         gym.set_actor_dof_velocity_targets(envs[i], actor_handles[i], dof_velocities)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)
    # time.sleep(0.1)

    cnt += 1

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
