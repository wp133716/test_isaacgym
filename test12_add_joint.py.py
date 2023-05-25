import math
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import torch

from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2

# 设置torch打印格式，保留5位小数，不用科学计数法
torch.set_printoptions(precision=3, sci_mode=False)
np.set_printoptions(precision=3, suppress=True)

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments()

# create a simulator
sim_params = gymapi.SimParams()
sim_params.substeps = 2
sim_params.dt = 1.0 / 60.0

sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0)

sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1

sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0 # Ground plane distance from origin
# plane_params.static_friction = 1 # Coefficient of dynamic friction
# plane_params.dynamic_friction = 1 # Coefficient of static friction
# plane_params.restitution = 0 # Coefficient of restitution
gym.add_ground(sim, plane_params)

# set up the env grid
num_envs = 4
spacing = 1.
env_lower = gymapi.Vec3(-spacing, -spacing, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# add urdf asset
asset_root = "./assets"
asset_file = "urdf/dof_test_camera.urdf"

# Load asset with default control type of position for all joints
asset_options = gymapi.AssetOptions()
print("asset_options 1: ", asset_options)
asset_options.fix_base_link = True
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
print("asset_options 2: ", asset_options)
cartpole_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# initial root pose for cartpole actors
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 2.0, 3.0)
# initial_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

# Create environment 1
# Cart held steady using position target mode.
# Pole rotating using velocity target mode.
env = gym.create_env(sim, env_lower, env_upper, 2)
cartpole1 = gym.create_actor(env, cartpole_asset, initial_pose, 'cartpole', 1, 1)
# Configure DOF properties
props = gym.get_actor_dof_properties(env, cartpole1)
print("DOF properties:\n", props, type(props), props.dtype)
# props["driveMode"] = (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS)
# props["stiffness"] = (5000.0, 1000.0)
# props["damping"] = (100.0, 200.0)
print("DOF properties:\n", props, type(props))
# assert False

gym.set_actor_dof_properties(env, cartpole1, props)
# Set DOF drive targets
cart_dof_handle = gym.find_actor_dof_handle(env, cartpole1, 'uav_to_camera')
print("cart_dof_handle: ", cart_dof_handle)

body = gym.get_actor_rigid_body_handle(env, cartpole1, 2)
print("body: ", body)
# create camera actor
cam_props = gymapi.CameraProperties()
camera_handle = gym.create_camera_sensor(env, cam_props)
# transform = gymapi.Transform()
# transform.p = gymapi.Vec3(0, 0, 0)
# transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(45.0))
# gym.set_camera_transform(camera_handle, env, transform)
local_transform = gymapi.Transform()
local_transform.p = gymapi.Vec3(0, 0, 0)
local_transform.r = gymapi.Quat.from_euler_zyx(0, 0, 0)
# local_transform.r = gymapi.Quat(*euler2quat([0., 0, pi/2]))
# local_transform.r = gymapi.Quat(*euler2quat([0., 0., 0.]))
# gym.set_camera_transform(camera_handle, env, local_transform)
gym.attach_camera_to_body(camera_handle, env, body, local_transform, gymapi.FOLLOW_TRANSFORM) # gymapi.FOLLOW_TRANSFORM

# gym.set_dof_target_velocity(env, cart_dof_handle, 0.0)
# gym.set_dof_target_velocity(env, pole_dof_handle, 0.25 * math.pi)

# Look at the first env
cam_pos = gymapi.Vec3(2, 3, 4)
cam_target = gymapi.Vec3(0, 2, 3)
gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)

actor_root_state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
rigid_body_state = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
dof_state = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
net_contact_forces = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))

gym.refresh_dof_state_tensor(sim)
gym.refresh_actor_root_state_tensor(sim)
gym.refresh_rigid_body_state_tensor(sim)

step = 0
dt = 1
# Simulate
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    gym.render_all_camera_sensors(sim)


    gym.refresh_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)

    # print("Actor root state:\n", actor_root_state, type(actor_root_state))
    print("Rigid body state:\n", rigid_body_state, type(rigid_body_state))
    print("camera state:\n", rigid_body_state[1])
    camera_rot_yaw = rigid_body_state[2][3:7] # rigid_body_state[3][3:7]
    camera_rot_pitch = rigid_body_state[1][3:7] #rigid_body_state[2][3:7]
    actor_rot = actor_root_state[0][3:7]
    camera_yaw1 = gymapi.Quat(camera_rot_yaw[0], camera_rot_yaw[1], camera_rot_yaw[2], camera_rot_yaw[3]).to_euler_zyx()
    camera_yaw = R.from_quat(camera_rot_yaw).as_euler('xyz', degrees=True)
    camera_pitch = R.from_quat(camera_rot_pitch).as_euler('xyz', degrees=True)
    actor_euler = R.from_quat(actor_rot).as_euler('xyz', degrees=True)
    # cart_roration = rigid_body_state[1][3:7]
    # cart_elur = R.from_quat(cart_roration).as_euler('xyz', degrees=True)
    # print("camera_rot_yaw:\n", camera_rot_yaw)
    print("actor_rotation:\n", actor_euler)
    # print("camera_yaw1:\n", np.rad2deg(camera_yaw1))
    print("camera_yaw:\n", camera_yaw)
    print("camera_pitch:\n", camera_pitch)
    print("step:\n", step)
    # camera_euler2[0] = step/1
    # camera_pitch[1] = np.clip(step/1, 0, 90)
    camera_yaw[2] = np.clip(-step/1, -180, 180)
    # actor_euler[2] = np.clip(step/1, -180, 180)
    

    # cart_elur[2] = step/10
    print("camera_yaw order:\n", camera_yaw)
    camera_rot_yaw = R.from_euler('xyz', camera_yaw, degrees=True).as_quat()
    camera_rot_pitch = R.from_euler('xyz', camera_pitch, degrees=True).as_quat()
    actor_quat = R.from_euler('xyz', actor_euler, degrees=True).as_quat()
    rigid_body_state[1][3:7] = torch.tensor(camera_rot_yaw)
    rigid_body_state[2][3:7] = torch.tensor(camera_rot_pitch)
    # gym.set_rigid_body_state_tensor(sim, gymtorch.unwrap_tensor(rigid_body_state))
    # gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(dof_positions))
    actor_root_state[0][3:7] = torch.tensor(actor_quat)
    # gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(actor_root_state))

    print("DOF state:\n", dof_state, type(dof_state))
    print("Net contact forces:\n", net_contact_forces, type(net_contact_forces))

    color_image = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_COLOR)
    color_image = color_image.reshape(cam_props.height, cam_props.width, 4)
    
    # r, g, b, _ = cv2.split(color_image)
    # color_image = cv2.merge([b, g, r]) # cv2 读取图片格式为BGR
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGBA2BGRA)

    # cv2.imshow('isaac gym', color_image)
    # cv2.imshow('isaac gym {}'.format(i), color_image[:,:,::-1])
    cv2.waitKey(100)

    gym.sync_frame_time(sim)
    step += dt
    if step >= 90: dt = -1
    if step <= -0: dt = 1

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)