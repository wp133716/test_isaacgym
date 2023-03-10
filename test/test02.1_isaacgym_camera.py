import random
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import torch
import cv2
import numpy as np
import time

def print_asset_info(asset, name):
    print("======== Asset info %s: ========" % (name))
    num_bodies = gym.get_asset_rigid_body_count(asset)
    num_joints = gym.get_asset_joint_count(asset)
    num_dofs = gym.get_asset_dof_count(asset)
    print("Got %d bodies, %d joints, and %d DOFs" %
          (num_bodies, num_joints, num_dofs))

    # Iterate through bodies
    print("Bodies:")
    for i in range(num_bodies):
        name = gym.get_asset_rigid_body_name(asset, i)
        print(" %2d: '%s'" % (i, name))

    # Iterate through joints
    print("Joints:")
    for i in range(num_joints):
        name = gym.get_asset_joint_name(asset, i)
        type = gym.get_asset_joint_type(asset, i)
        type_name = gym.get_joint_type_string(type)
        print(" %2d: '%s' (%s)" % (i, name, type_name))

    # iterate through degrees of freedom (DOFs)
    print("DOFs:")
    for i in range(num_dofs):
        name = gym.get_asset_dof_name(asset, i)
        type = gym.get_asset_dof_type(asset, i)
        type_name = gym.get_dof_type_string(type)
        print(" %2d: '%s' (%s)" % (i, name, type_name))

def print_actor_info(gym, env, actor_handle):

    name = gym.get_actor_name(env, actor_handle)

    body_names = gym.get_actor_rigid_body_names(env, actor_handle)
    body_dict = gym.get_actor_rigid_body_dict(env, actor_handle)

    joint_names = gym.get_actor_joint_names(env, actor_handle)
    joint_dict = gym.get_actor_joint_dict(env, actor_handle)

    dof_names = gym.get_actor_dof_names(env, actor_handle)
    dof_dict = gym.get_actor_dof_dict(env, actor_handle)

    print()
    print("===== Actor: %s =======================================" % name)

    print("\nBodies")
    print(body_names)
    print(body_dict)

    print("\nJoints")
    print(joint_names)
    print(joint_dict)

    print("\n Degrees Of Freedom (DOFs)")
    print(dof_names)
    print(dof_dict)
    print()

    # Get body state information
    body_states = gym.get_actor_rigid_body_states(
        env, actor_handle, gymapi.STATE_ALL)

    # Print some state slices
    print("Poses from Body State:")
    print(body_states['pose'])          # print just the poses

    print("\nVelocities from Body State:")
    print(body_states['vel'])          # print just the velocities
    print()

    # iterate through bodies and print name and position
    body_positions = body_states['pose']['p']
    for i in range(len(body_names)):
        print("Body '%s' has position" % body_names[i], body_positions[i])

    print("\nDOF states:")

    # get DOF states
    dof_states = gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)

    # print some state slices
    # Print all states for each degree of freedom
    print(dof_states)
    print()

    # iterate through DOFs and print name and position
    dof_positions = dof_states['pos']
    for i in range(len(dof_names)):
        print("DOF '%s' has position" % dof_names[i], dof_positions[i])

def init_sim_params():
    # get default set of parameters
    sim_params = gymapi.SimParams()

    # set common parameters
    sim_params.dt = 1 / 60
    sim_params.substeps = 2

    # Isaac Gym supports both y-up and z-up simulations.
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    # set PhysX-specific parameters
    sim_params.physx.use_gpu = True
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0

    # set Flex-specific parameters
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 20
    sim_params.flex.relaxation = 0.8
    sim_params.flex.warm_start = 0.5

    return sim_params

def update_camera(gym, sim, camera_handle):
    gym.destroy_camera_sensor(sim, camera_handle)

    return
gym = gymapi.acquire_gym()

sim_params = init_sim_params()

# create sim with these parameters
# sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
args = gymutil.parse_arguments(description="Asset and Environment Information")
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

# create the ground plane
gym.add_ground(sim, plane_params)

# Print out the working directory
# helpful in determining the relative location that assets will be loaded from
print("Working directory: %s" % os.getcwd())

# Path where assets are searched, relative to the current working directory
asset_root = "../assets"

# List of assets that will be loaded, both URDF and MJCF files are supported
asset_files = ["urdf/uav/urdf/rq-1-predator-mae-uav.urdf",
            #    "urdf/uav/urdf/rq-1-predator-mae-uav.urdf"]
               "urdf/uav/urdf/tpz-fuchs-apc.urdf"]
asset_names = ["predator", "fuchs-apc"]
loaded_assets = []


# Load the assets and ensure that we are successful
for asset in asset_files:
    print("Loading asset '%s' from '%s'" % (asset, asset_root))

    # current_asset = gym.load_asset(sim, asset_root, asset)

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.armature = 0.01
    current_asset = gym.load_asset(sim, asset_root, asset, asset_options)

    if current_asset is None:
        print("*** Failed to load asset '%s'" % (asset, asset_root))
        quit()
    loaded_assets.append(current_asset)

for i in range(len(loaded_assets)):
    print()
    print_asset_info(loaded_assets[i], asset_names[i])


# set up the env grid
num_envs = 1
envs_per_row = int(math.sqrt(num_envs))
env_spacing = 20.0
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# Create helper geometry used for visualization
# Create an wireframe axis
axes_geom = gymutil.AxesGeometry(scale=5)
# Create a wireframe sphere
sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
sphere_pose = gymapi.Transform(r=sphere_rot)
sphere_geom = gymutil.WireframeSphereGeometry(radius=0.5, num_lats=12, num_lons=12, pose=sphere_pose, color=(1, 1, 0))

# cache some common handles for later use
envs = []
uav_actor_handles = []
car_actor_handles = []
camera_handles = []

# Sensor camera properties
# cam_pos = gymapi.Vec3(0.0, 0.0, 0.0)
# cam_target = gymapi.Vec3(0.0, 0.0, -1.0)
cam_props = gymapi.CameraProperties()
print("cam_props width          : ", cam_props.width)
print("cam_props height         : ", cam_props.height)
print("horizontal_fov           : ", cam_props.horizontal_fov)
print("near_plane               : ", cam_props.near_plane)
print("far_plane                : ", cam_props.far_plane)
print("supersampling_horizontal : ", cam_props.supersampling_horizontal)
print("supersampling_vertical   : ", cam_props.supersampling_vertical)
print("use_collision_geometry   : ", cam_props.use_collision_geometry)
print("enable_tensors           : ", cam_props.enable_tensors)
'''
vertical_fov = height/width * horizontal_fov
??????FOV: horizontal_fov = 2 * atan(0.5*width(sensor width) / focal(mm))
??????FOV: vertical_fov = 2 * atan(0.5*height(sensor heght) / focal(mm))
??????: focal(mm) = width(sensor width) / (2*tan(horizontal_fov / 2))
'''
# cam_props.horizontal_fov = 90 # horizontal field of view in radians. The vertical field of view will be height/width * horizontal_fov
# cam_props.width = 360
# cam_props.height = 360

# create and populate the environments
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)

    # Draw axes and sphere at (0, 0, 0)
    gymutil.draw_lines(axes_geom, gym, viewer, envs[i], gymapi.Transform())
    gymutil.draw_lines(sphere_geom, gym, viewer, envs[i], gymapi.Transform())

    # create_actor
    # height = random.uniform(1.0, 2.5)
    height = 2
    uav_pose = gymapi.Transform()
    car_pose = gymapi.Transform()

    uav_pose.p = gymapi.Vec3(0., 0., height+100.)
    # pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
    uav_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * math.pi * 0)
    uav_actor_handle = gym.create_actor(env=env, asset=loaded_assets[0], pose=uav_pose, name=asset_names[0]+str(i), group=i, filter=-1)
    uav_actor_handles.append(uav_actor_handle)
    
    car_pose.p = gymapi.Vec3(0., 10., height)
    # pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
    car_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * math.pi * 0)
    car_actor_handle = gym.create_actor(env, loaded_assets[1], car_pose, asset_names[1]+str(i), i, -1)
    car_actor_handles.append(car_actor_handle)

    # create camera actor
    camera_handle = gym.create_camera_sensor(env, cam_props)
    camera_handles.append(camera_handle)
    body = gym.get_actor_rigid_body_handle(env, uav_actor_handle, 0)
    # print("-----------body : ", body)
    # transform = gymapi.Transform()
    # transform.p = gymapi.Vec3(0, 0, 0)
    # transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(45.0))
    # gym.set_camera_transform(camera_handle, env, transform)
    local_transform = gymapi.Transform()
    local_transform.p = gymapi.Vec3(0, -2.5, 0)
    local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(90.0))
    gym.attach_camera_to_body(camera_handle, env, body, local_transform, gymapi.FOLLOW_TRANSFORM)

# position viewer camera
# gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

print("=== Environment info: ================================================")

actor_count = gym.get_actor_count(env)
print("%d actors total" % actor_count)

# Iterate through all actors for the environment
for i in range(actor_count):
    actor_handle = gym.get_actor_handle(env, i)
    print_actor_info(gym, env, actor_handle)

num_bodies = gym.get_asset_rigid_body_count(loaded_assets[0])
# print('num_bodies', num_bodies)

# print("camera_handles : ", camera_handles)
# print("camera_handles : ", len(camera_handles))
# print("camera_handles : ", camera_handles[0].id==camera_handle)
num_image = 0
while not gym.query_viewer_has_closed(viewer):
    time.sleep(0.5)
    num_image += 1
    print(num_image)
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # render camera sensor
    gym.render_all_camera_sensors(sim)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)
    
    for i in range(num_envs):
        # randomize sensor camera position
        # y_offset = random.uniform(-1.0, 1.0)
        # z_offset = random.uniform(-1.0, 1.0)
        # cam_pos_new = cam_pos + gymapi.Vec3(0., y_offset, z_offset)
        # gym.set_camera_location(camera_handles[i], env, cam_pos_new, cam_target)

        # randomize light parameters
        l_color = gymapi.Vec3(random.uniform(1, 1), random.uniform(1, 1), random.uniform(1, 1))
        l_ambient = gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        l_direction = gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        # gym.set_light_parameters(sim, 0, l_color, l_ambient, l_direction)

        #### gym.set_sensor_properties(sim, 0, camera_handle, gymapi.CameraProperties(fov=gymapi.Deg2Rad(20))) ## chatGPT gengrate
        # gym.write_camera_image_to_file(sim, envs[i], camera_handles[i], gymapi.IMAGE_COLOR, "images/image_"+str(num_image)+".png")
        color_image = gym.get_camera_image(sim, envs[i], camera_handles[i], gymapi.IMAGE_COLOR)
        # print(color_image, color_image.shape)
        cv2.imshow('isaac gym {}'.format(i),color_image.reshape(cam_props.height, cam_props.width, 4))
        cv2.waitKey(1)
    update_camera(gym, sim, camera_handles[0])


gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
