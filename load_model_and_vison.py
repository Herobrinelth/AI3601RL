import dmc
import numpy as np
import torch
import argparse
from agent import CQLSAC
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--pt_dir", type=str, help="Checkpoint dir for generating gif")
    parser.add_argument("--run_name", type=str, default="walker_multi", help="Run name, default: CQL")
    parser.add_argument("--seed", type=int, default=42, help="Seed, default: 1")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size, default: 256")
    parser.add_argument("--hidden_size", type=int, default=1024, help="")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="")
    parser.add_argument("--temperature", type=float, default=1.0, help="")
    parser.add_argument("--cql_weight", type=float, default=1.0, help="")
    parser.add_argument("--target_action_gap", type=float, default=10, help="")
    parser.add_argument("--with_lagrange", type=int, default=0, help="")
    parser.add_argument("--tau", type=float, default=0.005, help="")
    parser.add_argument("--bc_weight", type=float, default=5.0, help="")
    parser.add_argument("--save_dir", type=str, default="./trained_models", help="")
    args = parser.parse_args()
    return args

def animate(frames, filename='animation.gif'):
    fig = plt.figure()
    plt.axis('off')
    im = plt.imshow(frames[0])

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    ani.save(filename, writer='pillow')
    plt.show()

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
            frames.append(eval_env.physics.render(camera_id=0, height=240, width=320))
        returns.append(cumulative_reward)
    return sum(returns) / eval_episodes, frames

def train(config):
    env_run = dmc.make('walker_run', seed=config.seed)
    env_walk = dmc.make('walker_walk', seed=config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    model_path = os.path.join(config.save_dir, config.pt_dir, config.run_name + '.pth')
    agent.actor_local.load_state_dict(torch.load(model_path))
    print("Evaluating on RUN task...")
    reward_run, frames_run = eval(eval_env=env_run, agent=agent, eval_episodes=10, task_type='run')
    print("Evaluating on WALK task...")
    reward_walk, frames_walk = eval(eval_env=env_walk, agent=agent, eval_episodes=10, task_type='walk')
    print("Reward on Run Task: {}, on Walk Task: {}.".format(reward_run, reward_walk))

    animate(frames_run, "run.gif")
    animate(frames_walk, "walk.gif")


if __name__ == "__main__":
    config = get_config()
    train(config)
