import random
from collections import deque

import numpy as np
import torch
from torch import nn

# DQN Network
class DQN(nn.Module):
    def __init__(
            self,
            input_dim: int, 
            output_dim: int,
            hidden_dim: int = 128
        ):
        super(DQN, self).__init__()
        self.lstm_layer = nn.LSTM(
            input_size=input_dim-2,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.flatten_params_embedding = nn.Linear(2, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.Q = nn.Linear(hidden_dim, output_dim)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                       nn.init.xavier_uniform_(param)
                    if 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, x, device):
        seq = torch.stack([it[0] for it in x]).to(device)
        flat = torch.stack([it[1] for it in x]).to(device)
        out, _ = self.lstm_layer(seq)
        out = torch.cat((out[:, -1], self.flatten_params_embedding(flat)), dim=1)
        out = self.Q(self.fc(out))
        return out


class Queue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = []
        self.size = 0

    def enqueue(self, item):
        if self.size == self.capacity:
            # raise OverflowError("Queue is full")
            self.dequeue()

        self.queue.append(item)
        self.size += 1

    def dequeue(self):
        if self.size == 0:
            raise IndexError("Queue is empty")
        item = self.queue.pop(0)
        self.size -= 1
        return item

    def sample(self, batch_size):
        if self.size == 0:
            raise IndexError("Queue is empty")
        if batch_size > self.size:
            raise ValueError("Batch size is greater than queue size")
        
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        batch = [self.queue[i] for i in indices]
        return batch, indices
    
    def __len__(self):
        return self.size
    

# DQN Agent
class DQNAgent:
    def __init__(self, config):
        self.state_dim = config.input_size
        self.action_dim = config.action_size
        self.hidden_dim = config.hidden_size
        self.memory = Queue(capacity=10000)
        self.gamma = config.gamma
        self.epsilon = config.random_eps
        self.epsilon_decay = config.random_eps_scaler
        self.epsilon_min = 0.00
        self.batch_size = config.batch_size
        self.learning_rate = config.lr

        self.policy_net = DQN(self.state_dim, self.action_dim, self.hidden_dim)
        self.target_net = DQN(self.state_dim, self.action_dim, self.hidden_dim)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()

    def change_lr(self, lr):
        for param in self.optimizer.param_groups:
            param['lr'] = lr

    def act(self, state, device, evaluate: bool = False):
        if (random.random() < self.epsilon) & (not evaluate):
            return random.randint(0, self.action_dim - 2), None

        with torch.no_grad():
            q_values = self.policy_net([state], device)
        return torch.argmax(q_values).item(), torch.max(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.enqueue((state, action, reward, next_state, done))

    def replay(self, device):
        if len(self.memory) < self.batch_size:
            return

        batch, indices = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        current_q = self.policy_net(states, device)[torch.arange(len(states)), actions]
        with torch.no_grad():
            next_policy_action = self.policy_net(next_states, device).argmax(dim=1) # get the best policy action
            next_target_qvalues = self.target_net(next_states, device)[torch.arange(len(next_states)), next_policy_action]
            target_q = rewards + self.gamma * next_target_qvalues * (1 - dones)

        # Loss computation
        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        for param in self.target_net.parameters():
            param.requires_grad = False

