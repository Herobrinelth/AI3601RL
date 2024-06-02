import glob
import numpy as np
import torch
from multiprocessing.pool import ThreadPool
#load dataset
class EpisodeDataset:
    def __init__(self, data_dir, discount, batch_size):
        self.data_dir = data_dir
        self.discount_factor = discount
        self.batch_size = batch_size
        self.episodes = self._load_episodes()
        self.current_episode = 0
        self.current_step = 1

    def _load_episodes(self):
        epss = sorted(glob.glob(f'{self.data_dir}/*.npz'))
        pool = ThreadPool(processes=4)
        episodes = pool.map(self._load_episode, epss)
        return episodes

    def _load_episode(self, eps):
        with open(eps, 'rb') as f:
            episode = np.load(f)
            episode = {k: episode[k] for k in episode.keys()}
        return episode
    def __iter__(self):
        self.current_episode = 0
        self.current_step = 1  
        return self

    def __next__(self):
        # print(self.current_episode )
        # print(len(self.episodes))
        if self.current_episode >= len(self.episodes):
            raise StopIteration

        batch_obs, batch_action, batch_reward, batch_discount, batch_next_obs = [], [], [], [], []

        while len(batch_obs) < self.batch_size:
            if self.current_episode >= len(self.episodes):
                break

            episode = self.episodes[self.current_episode]

            if self.current_step >= len(episode['observation']):
                self.current_episode += 1
                self.current_step = 1
                continue

            obs = torch.tensor(episode['observation'][self.current_step - 1], dtype=torch.float32)
            action = torch.tensor(episode['action'][self.current_step], dtype=torch.float32)
            reward = torch.tensor(episode['reward'][self.current_step], dtype=torch.float32).unsqueeze(-1)
            discount = torch.tensor(episode['discount'][self.current_step] * self.discount_factor, dtype=torch.float32).unsqueeze(-1)
            next_obs = torch.tensor(episode['observation'][self.current_step], dtype=torch.float32)

            batch_obs.append(obs)
            batch_action.append(action)
            batch_reward.append(reward)
            batch_discount.append(discount)
            batch_next_obs.append(next_obs)

            self.current_step += 1

        if len(batch_obs) == 0:
            raise StopIteration

        return {
            'observations': torch.stack(batch_obs),
            'actions': torch.stack(batch_action),
            'rewards': torch.stack(batch_reward),
            'discounts': torch.stack(batch_discount),
            'next_observations': torch.stack(batch_next_obs)
        }
# class EpisodeDataset:
#     def __init__(self, data_dir, discount):
#         self.data_dir = data_dir
#         self.discount_factor = discount
#         self.episodes = self._load_episodes()
#         self.current_episode = 0
#         self.current_step = 1

#     def _load_episodes(self):
#         episodes = []
#         epss = sorted(glob.glob(f'{self.data_dir}/*.npz'))
#         for eps in epss:
#             with open(eps, 'rb') as f:
#                 episode = np.load(f)
#                 episode = {k: episode[k] for k in episode.keys()}
#                 episodes.append(episode)
#         return episodes

#     def __iter__(self):
#         self.current_episode = 0
#         self.current_step = 1  # 从 1 开始，因为 0 是初始 observation
#         return self

#     def __next__(self):
#         if self.current_episode >= len(self.episodes):
#             raise StopIteration

#         episode = self.episodes[self.current_episode]

#         if self.current_step >= len(episode['observation']):
#             self.current_episode += 1
#             self.current_step = 1
#             return self.__next__()

#         obs = torch.tensor(episode['observation'][self.current_step - 1], dtype=torch.float32)
#         action = torch.tensor(episode['action'][self.current_step], dtype=torch.float32)
#         reward = torch.tensor(episode['reward'][self.current_step], dtype=torch.float32).unsqueeze(-1)
#         discount = torch.tensor(episode['discount'][self.current_step] * self.discount_factor, dtype=torch.float32).unsqueeze(-1)
#         next_obs = torch.tensor(episode['observation'][self.current_step], dtype=torch.float32)

#         self.current_step += 1

#         return {
#             'observations': obs.unsqueeze(0),
#             'actions': action.unsqueeze(0),
#             'rewards': reward.unsqueeze(0),
#             'discounts': discount.unsqueeze(0),
#             'next_observations': next_obs.unsqueeze(0)
#         }