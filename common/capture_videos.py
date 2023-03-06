import gym
import isaacgym
import isaacgymenvs
import torch

envs = isaacgymenvs.make(
    seed=0, 
    task="Ant", 
    num_envs=20, 
    sim_device="cuda:0",
    rl_device="cuda:0",
    graphics_device_id=0,
    headless=False,
    multi_gpu=False,
    virtual_screen_capture=True,
    force_render=True,
)
envs.is_vector_env = True
# envs = gym.wrappers.RecordVideo(
#     envs,
#     "./videos",
#     step_trigger=lambda step: step % 10000 == 0, # record the videos every 10000 steps
#     video_length=100  # for each video record up to 100 steps
# )
envs.reset()
print("the image of Isaac Gym viewer is an array of shape", envs.render(mode="rgb_array").shape)
for _ in range(1000000):
    envs.render(mode="rgb_array")
    envs.step(
        torch.rand((20,)+envs.action_space.shape, device="cuda:0")
    )
