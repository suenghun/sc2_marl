import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
from torch.distributions import Categorical
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(device))

class VDN(nn.Module):

    def __init__(self):
        super(VDN, self).__init__()

    def forward(self, q_local):
        return torch.sum(q_local, dim = 2)

class Network(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size):
        super(Network, self).__init__()
        self.fcn_1 = nn.Linear(obs_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first = True)
        self.fcn_2 = nn.Linear(hidden_size, action_size)

    def forward(self, obs, hidden_state):
        x = F.relu(self.fcn_1(obs))
        out, h = self.rnn(x, hidden_state)
        q = self.fcn_2(out)
        return q, h

class Replay_Buffer_for_RNN:
    def __init__(self, buffer_size, batch_size, num_agent, action_size):
        self.buffer = deque(maxlen=buffer_size)
        self.num_agent = num_agent
        self.agent_id = np.eye(self.num_agent).tolist()
        self.one_hot_actions = np.eye(action_size).tolist()
        self.batch_size = batch_size
        self.episode_indices = []
        self.episode_idx = 0

    def initialize_episode(self):
        obs_buffer = list()
        state_buffer = list()
        action_buffer = list()
        reward_buffer = list()
        avail_action_buffer = list()
        done_buffer = list()
        padding_buffer = list()

        obs_next_buffer = list()
        state_next_buffer = list()
        avail_action_next_buffer = list()


        self.buffer.append([obs_buffer,
                            state_buffer,
                            action_buffer,
                            reward_buffer,
                            avail_action_buffer,
                            done_buffer,
                            padding_buffer,
                            obs_next_buffer,
                            state_next_buffer,
                            avail_action_next_buffer])

        if self.episode_idx < 5000:
            self.episode_indices.append(self.episode_idx)
        self.episode_idx += 1

        # OBS의 목표 SHAPE
        # batch X num_agent X sequence_length X feature

    def concat_generator(self, obs, last_action):

        for n in range(self.num_agent):

            obs_n = obs[n].tolist()
            obs_n.extend(last_action[n])
            obs_n.extend(self.agent_id[n])
            yield obs_n

    def pop(self):
        self.buffer.pop()

    def memory(self, obs, state, action, reward, avail_actions, done, padding, obs_next, state_next, avail_actions_next, last_action):

        obs_gen = self.concat_generator(obs, last_action)
        obs = list(obs_gen)

        one_hot_action = [self.one_hot_actions[a] for a in action]

        obs_next_gen = self.concat_generator(obs_next, one_hot_action)
        obs_next = list(obs_next_gen)

        self.buffer[-1][0].append(obs)
        self.buffer[-1][1].append(state)
        self.buffer[-1][2].append(action)
        self.buffer[-1][3].append(reward)
        self.buffer[-1][4].append(avail_actions)
        self.buffer[-1][5].append(done)
        self.buffer[-1][6].append(padding)

        self.buffer[-1][7].append(obs_next)
        self.buffer[-1][8].append(state_next)
        self.buffer[-1][9].append(avail_actions_next)

    def generating_mini_batch(self, datas, batch_idx, cat):

        for s in batch_idx:
            if cat == 'obs':
                yield datas[s][0]
            if cat == 'state':
                yield datas[s][1]
            if cat == 'action':
                yield datas[s][2]
            if cat == 'reward':
                yield datas[s][3]
            if cat == 'avail_actions':
                yield datas[s][4]
            if cat == 'done':
                yield datas[s][5]
            if cat == 'padded':
                yield datas[s][6]
            if cat == 'obs_next':
                yield datas[s][7]
            if cat == 'state_next':
                yield datas[s][8]
            if cat == 'avail_action_next':
                yield datas[s][9]

    def sample(self, episode):
        if episode<self.batch_size:
            sampled_batch_idx = random.sample(self.episode_indices, episode + 1)
        else:
            sampled_batch_idx = random.sample(self.episode_indices, self.batch_size)


        obs = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='obs')
        obses = list(obs)
        state = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='state')
        states = list(state)
        action = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='action')
        actions = list(action)


        reward = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='reward')
        rewards = list(reward)



        avail_action = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='avail_actions')
        avail_actions = list(avail_action)
        done = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='done')
        dones = list(done)
        padded = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='padded')
        paddedes = list(padded)

        obs_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='obs_next')
        obses_next = list(obs_next)

        state_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='state_next')
        states_next = list(state_next)

        avail_action_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='avail_action_next')
        avail_actions_next = list(avail_action_next)

        return obses, states, actions, rewards, avail_actions, dones, paddedes, obses_next, states_next, avail_actions_next


class Agent:
    def __init__(self, num_agent, obs_size, state_size, hidden_size, action_size, buffer_size, batch_size, max_episode_len,gamma):
        self.hidden_size = hidden_size
        self.num_agent = num_agent
        self.Q = Network(obs_size, action_size, hidden_size).to(device)
        self.Q_tar = Network(obs_size, action_size, hidden_size).to(device)
        self.gamma = gamma
        self.agent_id = np.eye(self.num_agent).tolist()

        self.max_norm = 10
        self.VDN = VDN().to(device)
        self.VDN_target = VDN().to(device)

        self.Q_tar.load_state_dict(self.Q.state_dict())
        self.VDN_target.load_state_dict(self.VDN.state_dict())

        self.eval_params = list(self.VDN.parameters()) + list(self.Q.parameters())

        self.buffer = Replay_Buffer_for_RNN(buffer_size, batch_size, num_agent, action_size)
        self.batch_size = batch_size
        self.max_episode_len = max_episode_len

        self.optimizer = optim.RMSprop(self.eval_params, lr = 5e-4)
        self.action_space = [i for i in range(action_size)]

    @torch.no_grad()
    def sample_action(self, obs, hidden_state, avail_action, epsilon, last_action, agent_id):
        #print(epsilon)
        "obs의 shape는 num_agent X feature"
        "last_action의 shape는 action_size"
        "agent_id의 shapes는 num_agent"
        hidden_state = hidden_state.to(device)

        obs = torch.cat([torch.tensor(obs, device=torch.device('cuda:0')),
                         torch.tensor(last_action, device=torch.device('cuda:0')),
                         torch.tensor(self.agent_id[agent_id], device=torch.device('cuda:0'))])
        obs = obs.unsqueeze(0)

        "obs :  1 x feature_size, hidden : 1 x hidden_size"

        Q, hidden_state = self.Q(obs, hidden_state)
        mask = torch.tensor(avail_action, device=torch.device('cuda:0')).unsqueeze(0).bool()
        Q = Q.masked_fill(mask==0, float('-inf'))

        greedy_u = torch.argmax(Q)

        mask = np.array(avail_action, dtype=np.float64)

        if np.random.uniform(0, 1) >= epsilon:
            u = greedy_u
        else:
            u = np.random.choice(self.action_space, p=mask / np.sum(mask))


        return u, hidden_state

    def cal_Q(self, obses, actions, avail_actions, target, agent_id, episode):
        "mask의 shape는 batch_size X time_step X 1"
        if episode < self.batch_size:
            batch = episode + 1
        else:
            batch = self.batch_size


        hidden = torch.zeros((1, batch, self.hidden_size),device=torch.device('cuda:0'))
        if target == True:
            avail_action = avail_actions[:, :, agent_id]
            "avail_actions의 shape는 batch_size X time_step X action_size"

            q_tar, _ = self.Q_tar(obses[:, :, agent_id], hidden)
            "q_tar의 shape는 batch_size X time_step X action_size"
            q_tar = q_tar.masked_fill(avail_action==False, float('-inf'))
            q_tar = torch.max(q_tar, dim = 2)[0]
            temp_q = q_tar[:, 1:].clone()
            q_tar[:, 0:-1] = temp_q
            return q_tar
        else:
            q, _ = self.Q(obses[:, :, agent_id], hidden)
            action = actions[:, :, agent_id].unsqueeze(2)
            q = torch.gather(q, 2, action).squeeze(2)
            return q

    def learn(self, episode, variance = False, regularizer = 1):

        import time
        s = time.time()

        obses, states, actions, rewards, avail_actions, dones, paddings, obses_next, states_next, avail_actions_next = self.buffer.sample(episode)


        obses = torch.tensor(obses, device=torch.device('cuda:0'))

        dones = torch.tensor(dones, device=torch.device('cuda:0')).float()
        obses_next = torch.tensor(obses_next, device=torch.device('cuda:0'))
        states_next = torch.tensor(states_next, device=torch.device('cuda:0'))

        states = torch.tensor(states, device=torch.device('cuda:0'))
        actions = torch.tensor(actions, device=torch.device('cuda:0')).long()
        rewards = torch.tensor(rewards, device=torch.device('cuda:0'))
        avail_actions = torch.tensor(avail_actions, device=torch.device('cuda:0')).bool()
        avail_actions_next = torch.tensor(avail_actions_next, device=torch.device('cuda:0')).bool()
        paddings = torch.tensor(paddings, device=torch.device('cuda:0'))
        mask = paddings

        "batch_size X sequence_length X num_agent"
        q = [self.cal_Q(obses, actions, avail_actions, agent_id=n,target = False, episode = episode) for n in range(self.num_agent)]
        q_target = [self.cal_Q(obses, actions, avail_actions, agent_id=n,target = True, episode = episode) for n in range(self.num_agent)]


        q_tot = torch.stack(q, dim = 2)
        if variance == True:
            loss2 = torch.var(q_tot, dim=2).sum()/mask.sum()
            q_tot_target = torch.stack(q_target, dim = 2)
            q_tot = self.VDN(q_tot)
            q_tot_target = self.VDN_target(q_tot_target)
            target = rewards + self.gamma * q_tot_target*(1 - dones)
            td_target = (target.detach()-q_tot)*mask
            loss = (td_target**2).sum()/mask.sum() + regularizer * loss2
        else:
            q_tot_target = torch.stack(q_target, dim = 2)
            q_tot = self.VDN(q_tot)
            q_tot_target = self.VDN_target(q_tot_target)
            target = rewards + self.gamma * q_tot_target*(1 - dones)
            td_target = (target.detach()-q_tot)*mask
            loss = (td_target**2).sum()/mask.sum()



        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.eval_params, 10)
        self.optimizer.step()


        tau = 1e-3

        if episode % 200 == 0 and episode > 0:
            self.Q_tar.load_state_dict(self.Q.state_dict())
            self.VDN_target.load_state_dict(self.VDN.state_dict())
        # for target_param, local_param in zip(self.Q_tar.parameters(), self.Q.parameters()):
        #     target_param.data.copy_(tau*local_param.data +(1-tau)*target_param.data)
        # for target_param, local_param in zip(self.VDN_target.parameters(), self.VDN.parameters()):
        #     target_param.data.copy_(tau*local_param.data+(1-tau)*target_param.data)
        return loss

        #print("계산시간", time.time() - s)

    def memory(self, obs, state, action, reward, avail_actions, done, padding, obs_next, state_next, avail_actions_next, last_action):
        self.buffer.memory(obs, state, action, reward, avail_actions, done, padding, obs_next, state_next, avail_actions_next, last_action)


    def initialize_episode(self):
        self.buffer.initialize_episode()