import os
from tqdm import tqdm
import gymnasium as gym 
from models import (PolicyGradient,
                    Buffer,
                    PPOClip)
from torch.utils.tensorboard import SummaryWriter

SEED = 123 
ENV_NAME = "HalfCheetah-v5" # CartPole-v1, HalfCheetah-v5, highway-v0 might need an import 
EPISODES = 20000 # since we have gpus maybe increase this to like 500,000 to 1 million depending on how long it takes
GAMMA = 0.99
ALPHA_PI = 3e-04 # list 0.03 range
ALPHA_V = 1e-03 # list 0.03 range
MAX_TRAJECTORIES = 5
EPSILON = 0.2
POLICY = 'mlp'

env = gym.make(ENV_NAME)
ACTIONS_DIM = env.action_space.shape
STATE_DIM = env.observation_space.shape

buffer = Buffer(max_trajectories=MAX_TRAJECTORIES)

# Add a for loop where PG, PPO is trained 

model = PPOClip(state_dim=STATE_DIM,
                num_actions=ACTIONS_DIM[0],
                gamma=GAMMA,
                alpha_pi = ALPHA_PI,
                alpha_v =ALPHA_V,
                epsilon=EPSILON,
                policy=POLICY)

writer = SummaryWriter(os.path.join(os.path.dirname(__file__), 'runs'))

prev_cum_rew = float('-inf')
for episode in tqdm(range(EPISODES)):
    buffer.reset()
    state, _ = env.reset(seed=SEED)
    done = False
    cum_rew = 0
    buffer.sample_state(state)
    while not done:
        action = model.policy(state)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if not done:
            buffer.sample_state(state)
        buffer.sample_action(action)
        buffer.sample_reward(reward)
        cum_rew += reward
    buffer.add_trajectory()
    if episode >= MAX_TRAJECTORIES - 1:
        model.update(buffer)
    if cum_rew >= prev_cum_rew:
        prev_cum_rew = cum_rew
        model.save_model(ENV_NAME)
        print(f"saved model at episode: {episode} with cum_rew {prev_cum_rew}")
    writer.add_scalar('total_reward_per_episode', cum_rew, episode)
env.close() 