import random
from collections import deque

import numpy as np
import torch
from torch import nn


# Define the noisy layer instead of standard linear layer
class NoisyLayer(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters for the weights
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))

        # Learnable parameters for the biases
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        self.sigma_init = sigma_init # Initial value for the sigma
        self.reset_parameters() # Initialize weight and bias
        self.reset_noise() # Initialize noise for exploration

    def reset_parameters(self):
        """ Initialize weight and bias parameters using uniform distribution  """
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        """  Reset the noise for both weights and biases using a factorized noise approach """
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        # Apply noise to the weights and biases during training
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)

    def _scale_noise(self, size):
        # Generate noise using a Gaussian distribution and transform it
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    

class DQN(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int = 128,
            num_atoms: int = 51,
            v_min: float = -10.0,
            v_max: float = 10.0
        ):
        super(DQN, self).__init__()

        self.output_dim = output_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.atoms = torch.linspace(v_min, v_max, num_atoms)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        self.lstm_layer = nn.LSTM(
            input_size=input_dim-2,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.flatten_params_embedding = nn.Linear(2, hidden_dim)
        self.fc = nn.Sequential(
            NoisyLayer(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            NoisyLayer(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLayer(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim * num_atoms)
        self.softmax = nn.Softmax(dim=1)

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
        seq = [it[0] for it in x]
        flat = [it[1] for it in x]
        seq = torch.stack(seq).to(device)
        flat = torch.stack(flat).to(device)

        out, _ = self.lstm_layer(seq)
        out = torch.cat((out[:, -1], self.flatten_params_embedding(flat)), dim=1)
        out = self.fc(out)
        out = self.output_layer(out)
        out = out.view(-1, self.output_dim, self.num_atoms)
        return self.softmax(out)

    def get_action(self, state, device):
        dist = self.forward(state, device)
        q_values = torch.mean(dist * self.atoms.to(device), dim=-1)
        return torch.argmax(q_values, dim=-1)


class ProbabilityQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.probabilities = []
        self.queue = []
        self.size = 0
        self.min_prob = 1e-4

    def enqueue(self, item, probability=None):
        if probability is None:
            probability = self.min_prob

        if self.size == self.capacity:
            # raise OverflowError("Queue is full")
            self.dequeue()

        self.probabilities.append(probability)
        if probability != probability:
            print('here')
        self.queue.append(item)
        self.size += 1

    def dequeue(self):
        if self.size == 0:
            raise IndexError("Queue is empty")
        item = self.queue.pop(0)
        prob = self.probabilities.pop(0)
        self.size -= 1
        return item, prob

    def sample(self, batch_size):
        if self.size == 0:
            raise IndexError("Queue is empty")
        if batch_size > self.size:
            raise ValueError("Batch size is greater than queue size")
        
        probs = np.array(self.probabilities)
        probs /= probs.sum()

        indices = np.random.choice(self.size, size=batch_size, replace=False, p=probs)
        batch = [self.queue[i] for i in indices]
        return batch, indices
    
    def get_probabilities(self, indices):
        prob_sum = sum(self.probabilities)
        return [self.probabilities[idx]/prob_sum for idx in indices]
    
    def update_probabilities(self, indices, new_probabilities):
        if len(indices) != len(new_probabilities):
            raise ValueError("Indices and new_probabilities must have the same length")
        for i, index in enumerate(indices):
            if not (0 <= index < self.size):
                raise ValueError(f"Index {index} is out of range")
            self.probabilities[index] = self.min_prob if new_probabilities[i] < self.min_prob else new_probabilities[i]
     
    def __len__(self):
        return self.size


class DQNAgent:
    def __init__(self, config):
        self.state_dim = config.input_size
        self.action_dim = config.action_size
        self.hidden_dim = config.hidden_size
        self.num_atoms = config.num_atoms
        self.memory = ProbabilityQueue(capacity=10000)
        self.gamma = config.gamma
        self.beta = config.beta
        self.epsilon = config.random_eps
        self.epsilon_decay = config.random_eps_scaler
        self.epsilon_min = 0.00
        self.batch_size = config.batch_size
        self.learning_rate = config.lr

        self.policy_net = DQN(self.state_dim, self.action_dim, self.hidden_dim, self.num_atoms).to(config.device)
        self.target_net = DQN(self.state_dim, self.action_dim, self.hidden_dim, self.num_atoms).to(config.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def change_lr(self, lr):
        for param in self.optimizer.param_groups:
            param['lr'] = lr

    def act(self, state, device, evaluate: bool = False):
        if (random.random() < self.epsilon) & (not evaluate):
            return random.randint(0, self.action_dim - 1), None

        action = self.policy_net.get_action([state], device)
        return action, None

    def remember(self, state, action, reward, next_state, done):
        self.memory.enqueue((state, action, reward, next_state, done))

    def replay(self, device):
        if len(self.memory) < self.batch_size:
            return

        batch, indices = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Get the current distribution
        current_dist = self.policy_net(states, device)
        actions = actions.unsqueeze(1).expand((actions.shape[0], self.num_atoms)).unsqueeze(1)
        current_dist = current_dist.gather(1, actions).squeeze()

        with torch.no_grad():
            next_policy_action = self.policy_net.get_action(next_states, device)
            next_policy_action = next_policy_action.unsqueeze(1).expand((next_policy_action.shape[0], self.num_atoms)).unsqueeze(1)
            next_target_dist = self.target_net(next_states, device)
            next_target_dist = next_target_dist.gather(1, next_policy_action).squeeze()

            # Project the target distribution
            target_atoms = rewards + self.gamma * self.policy_net.atoms.to(device) * (1 - dones)
            target_atoms = target_atoms.clamp(self.policy_net.v_min, self.policy_net.v_max)
            b = (target_atoms - self.policy_net.v_min) / self.policy_net.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = torch.linspace(0, (self.batch_size - 1) * self.policy_net.num_atoms, self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.policy_net.num_atoms).to(device)

            target_dist = torch.zeros(next_target_dist.size()).to(device)
            target_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_target_dist * (u.float() - b)).view(-1))
            target_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_target_dist * (b - l.float())).view(-1))

        # Compute KL divergence loss
        loss = -(target_dist * torch.log(current_dist + 1e-8)).mean(dim=-1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update sample probabilities
        td_errors = torch.abs(target_dist - current_dist).mean(dim=-1)
        new_probabilities = td_errors.detach().cpu().ravel().tolist()
        self.memory.update_probabilities(indices, new_probabilities)

        # Compute importance-sampling weights
        N = len(self.memory)  # Total number of transitions in memory
        sampling_probs = torch.FloatTensor(self.memory.get_probabilities(indices)).to(device)
        loss_weights = (1 / (N * sampling_probs)).pow(self.beta).unsqueeze(1)

        if loss_weights.max() == 0:
            loss_weights = 1
        else:
            loss_weights /= loss_weights.max()  # Normalize weights to keep stability

        assert loss_weights.sum() == loss_weights.sum(), f'get nan weights, max weight is {loss_weights.max()}'

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        for param in self.target_net.parameters():
            param.requires_grad = False
