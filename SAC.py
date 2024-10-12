import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from RL_SAC_utils import soft_update, hard_update
from RL_SAC_model import QNetwork, DeterministicPolicy, GaussianPolicy

# # Environment
class Environment(object):
    def __init__(self, state0, num_epis, num_prod, num_inj, my_rom):
        super(Environment, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state = state0
        self.nsteps = num_epis
        self.num_prod = num_prod
        self.num_inj = num_inj
        self.istep = 0
        self.dt = torch.tensor(np.ones((1,1)), dtype=torch.float32).to(device=self.device)  # dt=20days, normalized to 1
        self.rom = my_rom
        
        self.noise = torch.Tensor(state0.shape[-1]).to(self.device)
        self.scale_bhp = (2500-2200)/(4069.2-2200)
        self.bias_bhp = (2200-2200)/(4069.2-2200)
        self.scale_rate = (1.0e6-1.0e5)/(1.2e6-0)
        self.bias_rate = (1.0e5-0)/(1.2e6-0)
        
        self.Qdiff_w = 3151.0 - 0 
        self.Qdiff_g = 1.2e6 - 0

    def step(self, action):
        self.istep +=1
        self.state, yobs = self.rom.predict_latent(self.state, self.dt, action)

        yobs[:, :self.num_prod] = yobs[:, :self.num_prod]*self.Qdiff_w
        yobs[:, self.num_prod:self.num_prod*2] = yobs[:, self.num_prod:self.num_prod*2]*self.Qdiff_g
        # self.state += action

        reward = reward_fun(yobs, action, self.num_prod, self.num_inj)
        done =  self.istep == self.nsteps
        return self.state, reward, done
    
    def reset(self, z0):
        self.istep =0 
        # noise = self.noise.normal_(0., std=0.10)
        # z00 = z0 + noise
        z00 = z0
        self.state = z00

        return z00
    
    def sample_action(self):
        # action_bhp = torch.FloatTensor(2000+(2500-2000)*torch.rand(self.num_prod)).to(self.device).unsqueeze(0)             ## bhp_min +(bhp_max-bhp_min)*sigma
        # action_bhp_norm = (action_bhp-2000)/(3322.3*1.25-2000)
        # action_rate = torch.FloatTensor(500+(1000 - 500)*torch.rand(self.num_inj)).to(self.device).unsqueeze(0)             ## q_min +(q_max-q_min)*sigma
        # action_rate_norm = (action_rate-0)/(1000*1.2-0)
        action_bhp = torch.FloatTensor(torch.rand(self.num_prod)).to(self.device).unsqueeze(0)*  self.scale_bhp + self.bias_bhp           ## bhp_min +(bhp_max-bhp_min)*sigma
        action_rate = torch.FloatTensor(torch.rand(self.num_inj)).to(self.device).unsqueeze(0)*  self.scale_rate + self.bias_rate             ## q_min +(q_max-q_min)*sigma
        # action = torch.cat((action_bhp_norm,action_rate_norm),dim=1)
        action = torch.cat((action_bhp,action_rate),dim=1)
        return action
    
def reward_fun(yobs, action, num_prod, num_inj):
    lf3toton =0.1167*4.536e-4 # convert lf^3 to ton 
    PV = ((50-10)*lf3toton*torch.sum(action[:,num_prod:], dim=1) - 5.0*torch.sum(yobs[:, :num_prod],dim=1) - 50.0*lf3toton*torch.sum(yobs[:, num_prod:num_prod*2], dim=1))/1000
    return PV

class SAC(object):
    def __init__(self, num_inputs, u_dim):
        super(SAC, self).__init__()
        self.gamma = 0.986
        self.tau = 0.005
        # self.alpha = 0.20
        self.alpha = 0.00

        # self.policy_type = Deterministic
        self.target_update_interval = 1
        self.automatic_entropy_tuning = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic = QNetwork(num_inputs, u_dim, 200).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=0.0001)

        self.critic_target = QNetwork(num_inputs, u_dim, 200).to(self.device)
        hard_update(self.critic_target, self.critic)


        self.alpha = 0
        self.automatic_entropy_tuning = False
        self.policy = DeterministicPolicy(num_inputs, u_dim, 200).to(device=self.device)
        # self.policy = GaussianPolicy(num_inputs, u_dim, 200).to(device=self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=0.0001)

    def select_action(self, state, evaluate=False):
        # state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        # state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch, action_batch, reward_batch, next_state_batch = memory.sample(batch_size=batch_size)
        
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            # next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            next_q_value = reward_batch + self.gamma * (min_qf_next_target)
        # print(state_batch)
        # print(action_batch)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        # qf_loss.backward()
        qf_loss.backward(retain_graph=True)
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        # print(policy_loss)
        self.policy_optim.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optim.step()

#         if self.automatic_entropy_tuning:
#             alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

#             self.alpha_optim.zero_grad()
#             alpha_loss.backward()
#             self.alpha_optim.step()

#             self.alpha = self.log_alpha.exp()
#             alpha_tlogs = self.alpha.clone() # For TensorboardX logs
#         else:
        alpha_loss = torch.tensor(0.).to(self.device)
        alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        # print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()