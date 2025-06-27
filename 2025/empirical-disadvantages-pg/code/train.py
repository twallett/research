#%%
import os
from tqdm import tqdm
import gymnasium as gym 
from models import (PolicyGradient,
                    Buffer,
                    PPOClip)
from torch.utils.tensorboard import SummaryWriter
import warnings

warnings.filterwarnings("ignore")

SEED = 123 
ENV_NAMES = ["CartPole-v1"] 
EPISODES = [2000] 
GAMMA = 0.99
ALPHA_PI_VALUES = [1e-04, 5e-04, 1e-03, 5e-03, 1e-02] 
ALPHA_V_VALUES = [1e-04, 5e-04, 1e-03, 5e-03, 1e-02]  
MAX_TRAJECTORIES = 16
EPSILON = 0.2
POLICY = 'mlp'

results_dir = f"{os.getcwd()}/results/cum-rew"
os.makedirs(results_dir, exist_ok=True)

for env_idx, env_name in enumerate(ENV_NAMES):
    print(f"\n=== Training on environment: {env_name} ===")
    
    env = gym.make(env_name)
    
    STATE_DIM = env.observation_space.shape
    ACTIONS_DIM = env.action_space.n

    writer_path = os.path.join(os.path.dirname(__file__), f'runs/{env_name}')
    writer = SummaryWriter(writer_path)
    
    for model_type in ["PolicyGradient", "PPOClip"]:
        print(f"\nTraining {model_type} on {env_name}")
        
        for pi_idx, alpha_pi in enumerate(ALPHA_PI_VALUES):
            for v_idx, alpha_v in enumerate(ALPHA_V_VALUES):
                
                alpha_config = f"pi_{alpha_pi}_v_{alpha_v}"

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
                    
                    if model_type == "PolicyGradient":
                        
                        sampled_states = [state]
                        sampled_actions = []
                        sampled_rewards = []
                        
                        while not done:
                            
                            action = model.policy(state)            
                            state, reward, terminated, truncated, info = env.step(action)
                            done = terminated or truncated
                            
                            if not done:
                                sampled_states.append(state)
                                
                            sampled_actions.append(action)
                            sampled_rewards.append(reward)
                            cum_rew += reward
                    
                        model.update(sampled_states, sampled_actions, sampled_rewards)
                    else:
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

# %%
