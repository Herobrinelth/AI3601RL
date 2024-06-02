import gym
import dmc
import numpy as np
from collections import deque
import torch
import wandb
import argparse
import glob
from utils import save, collect_random
import random
from agent import CQLSAC
from torch.utils.data import DataLoader, TensorDataset
import os
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="walker_multi", help="Run name, default: CQL")
    parser.add_argument("--env", type=str, default="dmc", help="Gym environment name, default: Pendulum-v0")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes, default: 100")
    parser.add_argument("--seed", type=int, default=42, help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=25, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size, default: 256")
    parser.add_argument("--hidden_size", type=int, default=512, help="")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="")
    parser.add_argument("--temperature", type=float, default=1.0, help="")
    parser.add_argument("--cql_weight", type=float, default=1.0, help="")
    parser.add_argument("--target_action_gap", type=float, default=10, help="")
    parser.add_argument("--with_lagrange", type=int, default=0, help="")
    parser.add_argument("--tau", type=float, default=5e-3, help="")
    parser.add_argument("--bc_weight", type=float, default=1.0, help="")
    parser.add_argument("--eval_every", type=int, default=1, help="")
    
    args = parser.parse_args()
    return args

def animate_gif(frames, filename='animation.gif'):
    fig = plt.figure()
    plt.axis('off')
    im = plt.imshow(frames[0])

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = FuncAnimation(fig, update, frames=frames, interval=30, blit=True)
    ani.save(filename, writer='pillow')
    plt.close(fig)

def prep_dataloader(episodes, batch_size=256):
    states = None
    actions = None
    rewards = None
    next_states = None
    dones = None
    for i in range(len(episodes)):
        episode = episodes[i]
        next_state = torch.from_numpy(episode['observation']).float()[1:]
        state = torch.from_numpy(episode['observation']).float()[0:-1]
        action = torch.from_numpy(episode['action']).float()[1:]
        reward = torch.from_numpy(episode['reward']).float()[1:]
        done = torch.zeros(1001, 1).long()[1:]
        done[-1][0] = 1
        if i == 0:
            states = state
            actions = action
            rewards = reward
            next_states = next_state
            dones = done
        else:
            states = torch.cat((states, state), dim=0)
            actions = torch.cat((actions, action), dim=0)
            rewards = torch.cat((rewards, reward), dim=0)
            next_states = torch.cat((next_states, next_state), dim=0)
            dones = torch.cat((dones, done), dim=0)
    tensordata = TensorDataset(states, actions, rewards, next_states, dones)
    dataloader = DataLoader(tensordata, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader

def load_data(episodes, data_path, task_type = 'walk'):
    """
    An example function to load the episodes in the 'data_path'.
    """
    epss = sorted(glob.glob(f'{data_path}/*.npz'))
    for eps in epss:
        with open(eps, 'rb') as f:
            episode = np.load(f)
            episode = {k: episode[k] for k in episode.keys()}
            observation = episode['observation']
            task = np.ones((observation.shape[0], 1)) if task_type == 'walk' else np.zeros((observation.shape[0], 1))
            observation = np.concatenate((observation, task), axis = 1)
            episode['observation'] = observation
            episodes.append(episode)
    print("Length of {}: {}".format(data_path, len(episodes)))
    return episodes

def eval(eval_env, agent, eval_episodes, task_type = 'walk'):
    """
    An example function to conduct online evaluation for some agentin eval_env.
    """
    returns = []
    frames = []
    for episode in range(eval_episodes):
        time_step = eval_env.reset()
        cumulative_reward = 0
        while not time_step.last():
            task = np.ones((1)) if task_type == 'walk' else np.zeros((1))
            action = agent.get_action(np.concatenate((time_step.observation, task), axis=0), eval=True)
            time_step = eval_env.step(action)
            cumulative_reward += time_step.reward
            # frames.append(eval_env.physics.render(camera_id=0, height=240, width=320))
        returns.append(cumulative_reward)
    return sum(returns) / eval_episodes, frames

def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = './trained_models/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    folder_name = os.path.join(save_dir, current_time)
    os.makedirs(folder_name, exist_ok=True)
    task_name = config.run_name
    with open(os.path.join(folder_name, 'configs.txt'), 'w') as f:
        for arg, value in vars(config).items():
            f.write(f"{arg}: {value}\n")
    env_run = dmc.make('walker_run', seed=config.seed)
    env_walk = dmc.make('walker_walk', seed=config.seed)
    episodes = []
    episodes = load_data(episodes, "collected_data/walker_run-td3-medium/data", task_type = 'run')
    episodes = load_data(episodes, "collected_data/walker_run-td3-medium-replay/data", task_type = 'run')
    episodes = load_data(episodes, "collected_data/walker_walk-td3-medium/data", task_type = 'walk')
    episodes = load_data(episodes, "collected_data/walker_walk-td3-medium-replay/data", task_type = 'walk')
    print("Total length of dataset: {}".format(len(episodes)))
    dataloader = prep_dataloader(episodes=episodes, batch_size=config.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on {}".format(device))
    batches = 0
    average10_run = deque(maxlen=10)
    average10_walk = deque(maxlen=10)
    
    with wandb.init(project="CQL-offline", name=config.run_name, config=config):
        agent = CQLSAC(state_size=env_run.observation_spec().shape[0] + 1,
                        action_size=env_run.action_spec().shape[0],
                        tau=config.tau,
                        bc_weight=config.bc_weight,
                        hidden_size=config.hidden_size,
                        learning_rate=config.learning_rate,
                        temp=config.temperature,
                        with_lagrange=config.with_lagrange,
                        cql_weight=config.cql_weight,
                        target_action_gap=config.target_action_gap,
                        device=device)

        wandb.watch(agent, log="gradients", log_freq=10)
        reward_run = 0
        reward_walk = 0
        max_reward = 0
        wandb.log({"Reward Run": reward_run, "Reward Walk": reward_walk, "Episode": 0, "Batches": batches}, step=batches)
        for i in range(1, config.episodes+1):

            for batch_idx, experience in tqdm(enumerate(dataloader)):
                states, actions, rewards, next_states, dones = experience
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)
                policy_loss, alpha_loss, bellmann_error1, bellmann_error2, cql1_loss, cql2_loss, current_alpha, lagrange_alpha_loss, lagrange_alpha = agent.learn((states, actions, rewards, next_states, dones))
                batches += 1

            if i % config.eval_every == 0:
                print("Evaluating on RUN task...")
                reward_run, _ = eval(eval_env=env_run, agent=agent, eval_episodes=10, task_type='run')
                # animate_gif(frames, os.path.join(folder_name, 'run.gif'))
                print("Evaluating on WALK task...")
                reward_walk, _ = eval(eval_env=env_walk, agent=agent, eval_episodes=10, task_type='walk')
                # animate_gif(frames, os.path.join(folder_name, 'walk.gif'))
                wandb.log({"Reward Run": reward_run, "Reward Walk": reward_walk, "Episode": i, "Batches": batches}, step=batches)
                average10_walk.append(reward_walk)
                average10_run.append(reward_run)
                print("Episode: {} | Reward Run: {} | Reward Walk: {} | Polciy Loss: {} | Batches: {}".format(i, 
                                                                            reward_run, reward_walk, policy_loss, batches,))
            
            wandb.log({
                       "Average10 Run": np.mean(average10_run),
                       "Average10 Walk": np.mean(average10_walk),
                       "Policy Loss": policy_loss,
                       "Alpha Loss": alpha_loss,
                       "Lagrange Alpha Loss": lagrange_alpha_loss,
                       "CQL1 Loss": cql1_loss,
                       "CQL2 Loss": cql2_loss,
                       "Bellman error 1": bellmann_error1,
                       "Bellman error 2": bellmann_error2,
                       "Alpha": current_alpha,
                       "Lagrange Alpha": lagrange_alpha,
                       "Batches": batches,
                       "Episode": i})

            if np.mean(average10_walk) > max_reward:
                print("Saving checkpoints...")
                save(config, save_name=task_name, save_dir=folder_name, model=agent.actor_local, wandb=wandb, ep=0)
                max_reward = np.mean(average10_walk)
            # if i == 1:
            #     print("Generating gif on RUN task...")
            #     _, frames = eval(eval_env=env_run, agent=agent, eval_episodes=10, task_type='run')
            #     animate_gif(frames, os.path.join(folder_name, 'run.gif'))
            #     print("Generating gif on WALK task...")
            #     _, frames = eval(eval_env=env_walk, agent=agent, eval_episodes=10, task_type='walk')
            #     animate_gif(frames, os.path.join(folder_name, 'walk.gif'))


if __name__ == "__main__":
    config = get_config()
    train(config)
