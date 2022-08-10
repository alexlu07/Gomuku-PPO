import torch
import random

from env import Env
from model import Model

class SelfplayWrapper(Env):
    def __init__(self, board_size=11, max_age_diff=10):
        super().__init__(board_size)
        
        self.max_age_diff = max_age_diff
        self.opponent_model = Model(self.obs_dim, self.act_dim)

        self.agent_player = None

    def reset(self, curr_age):
        obs = super().reset()
        
        opponent_age = random.randint(max(0, curr_age-self.max_age_diff), curr_age)
        self.load_model(opponent_age)

        self.agent_player = random.choice([-1, 1])
        if self.agent_player == -1:
            # Opponent moves
            act, _, _ = self.opponent_model.step(torch.as_tensor(obs, dtype=torch.float32), 
                                legal_actions=torch.tensor(self.legal_actions))
            obs, rew, done = super().step(act)
        
        return obs


    def step(self, action):
        # Agent moves
        obs, rew, done = super().step(action)
        if done:
            return obs, rew, done

        # Opponent moves
        act, _, _ = self.opponent_model.step(torch.as_tensor(obs, dtype=torch.float32), 
                            legal_actions=torch.tensor(self.legal_actions))
        obs, rew, done = super().step(act)
        if done:
            return obs, -rew, done
        
        return obs, rew, done

    def load_model(self, epoch):
        self.opponent_model.load_state_dict(torch.load(f"./results/weights/{epoch}.pt")["model"])