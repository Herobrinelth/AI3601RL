import torch
import utils
from CDSagent import CQLCDSAgent
import dmc
from load_data import EpisodeDataset
from PIL import Image
import os
#change discount and batchsize here
walk_data = EpisodeDataset("collected_data/walker_run-td3-medium-replay/data", 0.99,1024)
run_data = EpisodeDataset("collected_data/walker_walk-td3-medium-replay/data", 0.99,512)

def eval(global_step, agent, env, num_eval_episodes, save_gif=False):
    step, episode, total_reward = 0, 0, 0
    eval_until_episode = utils.Until(num_eval_episodes)
    frames = []

    while eval_until_episode(episode):
        time_step = env.reset()
        episode_frames = []
        while not time_step.last():
            with torch.no_grad(), utils.eval_mode(agent):
                action = agent.act(time_step.observation, step=global_step, eval_mode=True)
            time_step = env.step(action)
            total_reward += time_step.reward
            step += 1
            if save_gif:
                frame = render_frame(env)
                episode_frames.append(frame)

        episode += 1
        if save_gif:
            frames.extend(episode_frames)

    print('episode_reward', total_reward / episode)
    print('episode_length', step / episode)
    print('step', global_step)

    if save_gif:
        gif_filename = f'eval_{global_step}.gif'
        animate_gif(frames, filename=gif_filename)

def render_frame(environment):
    return environment.physics.render(camera_id=0, height=240, width=320)

def animate_gif(frames, filename='animation.gif'):
    frames = [Image.fromarray(frame) for frame in frames]
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=50, loop=0)
def main():
    #----------create device----------
    #set device
    device = 'cuda'
    device = torch.device(device)
    folder_name="model_save"#save model in folder , create it in your work_dir first!
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    #----------create env-----------
    #set config
    task_name="walker_run"#env for walk/run
    seed=42
    env =dmc.make(task_name, seed=seed)	
    num_eval_episodes=10 #eval episodes number
    num_grad_steps = 1000000
    eval_every_steps = 10000
    #Done
    #-----------create agent-----------
    #set agent training configs
    agent=CQLCDSAgent(name=None,obs_shape=env.observation_spec().shape
                    ,action_shape=env.action_spec().shape, num_expl_steps=0,
                    device=device,actor_lr=1e-4,critic_lr=3e-4,hidden_dim=256,
                    critic_target_tau=0.01,nstep=1,batch_size=1024,use_tb=True,alpha=50,n_samples=3,target_cql_penalty=5.0,use_critic_lagrange=False)
    #-----------initial for training-----------
    global_step = 0
    train_until_step = utils.Until(num_grad_steps)
    eval_every_step = utils.Every(eval_every_steps)
    #-----------train process-----------
    replay_iter_main=iter(run_data)
    replay_iter_share=iter(walk_data)
    while train_until_step(global_step):
        # try to evaluate
        if eval_every_step(global_step):
            eval(global_step, agent, env, num_eval_episodes, save_gif=(global_step % 10000 == 0))#save gif every 10000 step
            torch.save(agent.actor.state_dict(), os.path.join(folder_name, "%d_model.pth"%global_step))
        # train the agent
        try:
            batch_main = next(replay_iter_main)
        except StopIteration:
            replay_iter_main = iter(walk_data)
            batch_main = next(replay_iter_main)

        try:
            batch_share = next(replay_iter_share)
        except StopIteration:
            replay_iter_share = iter(run_data)
            batch_share = next(replay_iter_share)

        metrics = agent.update(batch_main, batch_share, global_step, num_grad_steps)
        global_step += 1
        if global_step % 1000 == 0:
            print(metrics['batch_reward'])#output batch_reward every 1000 steps
if __name__ == '__main__':
    main()
