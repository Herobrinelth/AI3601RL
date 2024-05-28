import dmc
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# class Agent:
#     # An example of the agent to be implemented.
#     # Your agent must extend from this class (you may add any other functions if needed).
#     def __init__(self, state_dim, action_dim):
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#     def act(self, state):
#         action = np.random.uniform(-5, 5, size=(self.action_dim))
#         return action
#     def load(self, load_path):
#         pass
class Agent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Add any additional variables needed for training
        
    def act(self, state):
        # Modify the action selection mechanism if needed
        action = np.random.uniform(-5, 5, size=(self.action_dim))
        return action
    
    def train(self, episodes, eval_env):
        # Training loop
        for episode in range(episodes):
            time_step = eval_env.reset()
            cumulative_reward = 0
            while not time_step.last():
                action = self.act(time_step.observation)
                next_time_step = eval_env.step(action)
                reward = next_time_step.reward
                cumulative_reward += reward
                # Use the observed transition to update the agent's parameters
                
                time_step = next_time_step
                
            # Print or log the cumulative reward for this episode

def load_data(data_path):
    """
    An example function to load the episodes in the 'data_path'.
    """
    epss = sorted(glob.glob(f'{data_path}/*.npz'))
    episodes = []
    for eps in epss:
        with open(eps, 'rb') as f:
            episode = np.load(f)
            episode = {k: episode[k] for k in episode.keys()}
            episodes.append(episode)
    print(len(episodes))
    return episodes

load_data("collected_data/walker_run-td3-medium/data")
load_data("collected_data/walker_run-td3-medium-replay/data")
load_data("collected_data/walker_walk-td3-medium/data")
load_data("collected_data/walker_walk-td3-medium-replay/data")
def render_frame(environment):
    return environment.physics.render(camera_id=0, height=240, width=320)

def plot_frame(frame):
    plt.imshow(frame)
    plt.axis('off')
    plt.show()
def eval(eval_env, agent, eval_episodes, visualize=False):
    returns = []
    frames = []
    for episode in range(eval_episodes):
        time_step = eval_env.reset()
        cumulative_reward = 0
        while not time_step.last():
            action = agent.act(time_step.observation)
            time_step = eval_env.step(action)
            cumulative_reward += time_step.reward
            
            if visualize:
                frame = render_frame(eval_env)
                frames.append(frame)
                
        returns.append(cumulative_reward)
    return sum(returns) / eval_episodes, frames

def animate(frames):
    fig = plt.figure()
    plt.axis('off')
    im = plt.imshow(frames[0])

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    plt.show()

task_name = "walker_walk"
seed = 42
eval_env = dmc.make(task_name, seed=seed)
avg_return, frames = eval(eval_env=eval_env, agent=Agent(24, 6), eval_episodes=1, visualize=True)
print(avg_return)
animate(frames)

task_name = "walker_run"
seed = 42
eval_env = dmc.make(task_name, seed=seed)
avg_return, frames = eval(eval_env=eval_env, agent=Agent(24, 6), eval_episodes=1, visualize=True)
print(avg_return)
animate(frames)