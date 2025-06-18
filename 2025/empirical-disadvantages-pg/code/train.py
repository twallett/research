import os
<<<<<<< Updated upstream
import numpy as np
=======
>>>>>>> Stashed changes
from tqdm import tqdm
import gymnasium as gym 
from models import (PolicyGradient,
                    Buffer,
                    PPOClip)
from torch.utils.tensorboard import SummaryWriter
<<<<<<< Updated upstream
import warnings

warnings.filterwarnings("ignore")

SEED = 123 
<<<<<<< Updated upstream:2025/empirical-disadvantages-policy-gradient/code/train.py
ENV_NAMES = ["CartPole-v1", "HalfCheetah-v5"] 
EPISODES = [2000, 50000] 
EPISODES = [1, 1] 
=======
ENV_NAME = "Cartpole-v1" # CartPole-v1, HalfCheetah-v5, highway-v0 might need an import 
EPISODES = 20000 # since we have gpus maybe increase this to like 500,000 to 1 million depending on how long it takes
>>>>>>> Stashed changes:2025/empirical-disadvantages-pg/code/train.py
GAMMA = 0.99
ALPHA_PI_VALUES = [3e-05, 3e-04, 3e-03]
ALPHA_V_VALUES = [1e-04, 1e-03, 1e-02] 
MAX_TRAJECTORIES = 16
EPSILON = 0.2
POLICY = 'mlp'

results_dir = f"{os.getcwd()}/results/cum-rew"
os.makedirs(results_dir, exist_ok=True)

for env_idx, env_name in enumerate(ENV_NAMES):
    print(f"\n=== Training on environment: {env_name} ===")
    
    env = gym.make(env_name)
    STATE_DIM = env.observation_space.shape
    
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if is_discrete:
        ACTIONS_DIM = env.action_space.n
        print(f"Discrete action space with {ACTIONS_DIM} possible actions")
    else:
        ACTIONS_DIM = env.action_space.shape[0]
        print(f"Continuous action space with dimension {ACTIONS_DIM}")

    writer_path = os.path.join(os.path.dirname(__file__), f'runs/{env_name}')
    writer = SummaryWriter(writer_path)
    
    for model_type in ["PolicyGradient", "PPOClip"]:
        print(f"\nTraining {model_type} on {env_name}")
        
        for pi_idx, alpha_pi in enumerate(ALPHA_PI_VALUES):
            for v_idx, alpha_v in enumerate(ALPHA_V_VALUES):
                alpha_config = f"pi_{alpha_pi}_v_{alpha_v}"
                print(f"\nTesting with alpha_pi={alpha_pi}, alpha_v={alpha_v}")
                
                buffer = Buffer(max_trajectories=MAX_TRAJECTORIES)
                
                if model_type == "PolicyGradient":
                    model = PolicyGradient(
                        state_dim=STATE_DIM,
                        num_actions=ACTIONS_DIM,
                        gamma=GAMMA,
                        alpha_pi=alpha_pi,
                        alpha_v=alpha_v,
                        policy=POLICY
                    )
                else:  
                    model = PPOClip(
                        state_dim=STATE_DIM,
                        num_actions=ACTIONS_DIM,
                        gamma=GAMMA,
                        alpha_pi=alpha_pi,
                        alpha_v=alpha_v,
                        epsilon=EPSILON,
                        policy=POLICY
                    )
                
                cum_rewards = []
                
                episodes_for_env = EPISODES[env_idx] if env_idx < len(EPISODES) else EPISODES[0]
                prev_cum_rew = float('-inf')
                
                for episode in tqdm(range(episodes_for_env)):
                    buffer.reset()
                    state, _ = env.reset(seed=SEED)
                    done = False
                    cum_rew = 0
                    buffer.sample_state(state)
                    
                    while not done:
                        action = model.policy(state)
                        
                        if is_discrete:
                            if hasattr(action, "numpy"): 
                                action = int(action.numpy().argmax())
                            elif isinstance(action, np.ndarray):  
                                action = int(action.argmax())
                            else:  
                                action = int(action)
                        else:
                            if hasattr(action, "numpy"):  
                                action_np = action.numpy()
                            else:
                                action_np = np.array(action)
                            
                            if len(action_np.shape) == 0:  
                                action = np.zeros(ACTIONS_DIM)
                                action[0] = action_np
                            elif len(action_np.shape) == 1 and action_np.shape[0] != ACTIONS_DIM:
                                if action_np.shape[0] == 1:
                                    action = np.zeros(ACTIONS_DIM)
                                    action[0] = action_np[0]
                                else:
                                    action = np.zeros(ACTIONS_DIM)
                                    for i in range(min(ACTIONS_DIM, len(action_np))):
                                        action[i] = action_np[i]
                            else:
                                action = action_np
                                
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
                    
                    cum_rewards.append(float(cum_rew))
                    
                    if cum_rew >= prev_cum_rew:
                        prev_cum_rew = cum_rew
                        model_save_path = f"{model_type}_{env_name}_pi{alpha_pi}_v{alpha_v}"
                        model.save_model(model_save_path)
                        print(f"Saved model at episode: {episode} | cum_rew: {prev_cum_rew:.2f}")
                    
                    writer.add_scalar(f'{model_type}/{alpha_config}/reward', cum_rew, episode)
                
                filename = f"{env_name}_{model_type}_pi{alpha_pi}_v{alpha_v}_rewards.txt"
                file_path = os.path.join(results_dir, filename)
                
                with open(file_path, 'w') as f:
                    for reward in cum_rewards:
                        f.write(f"{reward}\n")
                
                print(f"Saved rewards to {file_path}")

    env.close()

print("\nTraining complete! All results saved to the 'results' directory")
=======

SEED = 123 
ENV_NAME = "Cartpole-v1" # CartPole-v1, HalfCheetah-v5, highway-v0 might need an import 
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
>>>>>>> Stashed changes
