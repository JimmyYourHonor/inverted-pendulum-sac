import os
import torch
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork, LinearVAE, ActorNetwork_2

class Agent():
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=8, env=None, gamma=0.99,
    n_actions=2, max_size=1000000, tau=0.005, layer1_size=256, layer2_size=256, batch_size=256, 
    reward_scale=2 ):
        self.gamma = 0.99
        self.tau = tau
        self.memeory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, input_dims, env.action_space.high, n_actions=n_actions)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions, name='critic_2')
        self.value = ValueNetwork(beta, input_dims, name='value')
        self.target_value = ValueNetwork(beta, input_dims, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)
    
    def choose_action(self, observation):
        state = torch.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memeory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_dict = dict(target_value_params)
        value_dict = dict(value_params)

        for name in target_value_dict:
            target_value_dict[name] = tau*value_dict[name].clone() + \
                (1-tau)*target_value_dict[name].clone()

        self.target_value.load_state_dict(target_value_dict)
    
    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
    
    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()

    def learn(self):
        if self.memeory.mem_cntr < self.batch_size:
            return
        
        states, new_states, actions, rewards, dones = self.memeory.sample_buffer(self.batch_size)
        states = torch.tensor(states, dtype=torch.float).to(self.actor.device)
        new_states = torch.tensor(new_states, dtype=torch.float).to(self.actor.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.actor.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.actor.device)
        dones = torch.tensor(dones).to(self.actor.device)
        

        states_value = self.value(states).view(-1)
        new_states_value = self.target_value(new_states).view(-1)
        new_states_value[dones] = 0.0
        
        action, log_probs = self.actor.sample_normal(states, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1(states, action)
        q2_new_policy = self.critic_2(states, action)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(states_value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        action, log_probs = self.actor.sample_normal(states, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1(states, action)
        q2_new_policy = self.critic_2(states, action)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1) 

        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q_hat = self.scale*rewards + self.gamma*new_states_value
        q1_old_policy = self.critic_1(states, actions).view(-1)
        q2_old_policy = self.critic_2(states, actions).view(-1)
        critic1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic1_loss + critic2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        
        self.update_network_parameters()
        
        return 0, value_loss, actor_loss, critic_loss



class Agent_2():
    def __init__(self, alpha=0.00005, beta=0.00005, input_dims=5, env=None, gamma=0.99,
    n_actions=2, max_size=1000000, tau=0.005, layer1_size=256, layer2_size=256, batch_size=256, 
    reward_scale=2 ):
        self.gamma = 0.99
        self.tau = tau
        self.memeory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        latent_dims = 10

        self.actor = ActorNetwork_2(alpha, latent_dims, env.action_space.high, n_actions=n_actions)
        self.critic_1 = CriticNetwork(beta, latent_dims, n_actions, name='critic_det_1')
        self.critic_2 = CriticNetwork(beta, latent_dims, n_actions, name='critic__det_2')
        self.value = ValueNetwork(beta, latent_dims, name='value_det')
        self.target_value = ValueNetwork(beta, latent_dims, name='target_value_det')
        self.VAE = LinearVAE()

        self.scale = reward_scale
        self.update_network_parameters(tau=1)
    
    def choose_action(self, observation):
        state = torch.Tensor([observation]).to(self.actor.device)
        state_latent = self.VAE.sample_normal(state)
        actions = self.actor(state_latent)
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memeory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_dict = dict(target_value_params)
        value_dict = dict(value_params)

        for name in target_value_dict:
            target_value_dict[name] = tau*value_dict[name].clone() + \
                (1-tau)*target_value_dict[name].clone()

        self.target_value.load_state_dict(target_value_dict)
    
    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
    
    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()

    def learn(self):
        if self.memeory.mem_cntr < self.batch_size:
            return
        
        states, new_states, actions, rewards, dones = self.memeory.sample_buffer(self.batch_size)
        states = torch.tensor(states, dtype=torch.float).to(self.actor.device)
        new_states = torch.tensor(new_states, dtype=torch.float).to(self.actor.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.actor.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.actor.device)
        dones = torch.tensor(dones).to(self.actor.device)
        
        # Train VAE with KL divergence + reconstruction_loss + log_probs
        reconstruction, mu, logvar, log_probs = self.VAE(states)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        reconstruction_loss = F.mse_loss(reconstruction, states)
        final_loss = KLD + reconstruction_loss
        self.VAE.optimizer.zero_grad()
        final_loss.backward(retain_graph=True)
        self.VAE.optimizer.step()

        latent_states = self.VAE.sample_normal(states)
        states_value = self.value(latent_states).view(-1)
        new_latent_states = self.VAE.sample_normal(new_states)
        new_states_value = self.target_value(new_latent_states).view(-1)
        new_states_value[dones] = 0.0
        
        action = self.actor(latent_states)
        q1_new_policy = self.critic_1(latent_states, action)
        q2_new_policy = self.critic_2(latent_states, action)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value
        value_loss = 0.5 * F.mse_loss(states_value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actor_loss = - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q_hat = self.scale*rewards + self.gamma*new_states_value
        q1_old_policy = self.critic_1(latent_states, actions).view(-1)
        q2_old_policy = self.critic_2(latent_states, actions).view(-1)
        critic1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic1_loss + critic2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        self.update_network_parameters()
        return final_loss, value_loss, actor_loss, critic_loss
    
class Agent_sm():
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=8, env=None, gamma=0.99,
    n_actions=2, max_size=1000000, tau=0.005, layer1_size=256, layer2_size=256, batch_size=256, 
    reward_scale=2 ):
        self.gamma = 0.99
        self.tau = tau
        self.memeory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, input_dims, env.action_space.high, n_actions=n_actions)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions, name='critic_2')
        self.value = ValueNetwork(beta, input_dims, name='value')
        self.target_value = ValueNetwork(beta, input_dims, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)
    
    def choose_action(self, observation):
        state = torch.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memeory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_dict = dict(target_value_params)
        value_dict = dict(value_params)

        for name in target_value_dict:
            target_value_dict[name] = tau*value_dict[name].clone() + \
                (1-tau)*target_value_dict[name].clone()

        self.target_value.load_state_dict(target_value_dict)
    
    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
    
    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()

    def learn(self):
        if self.memeory.mem_cntr < self.batch_size:
            return
        
        states, new_states, actions, rewards, dones = self.memeory.sample_buffer(self.batch_size)
        states = torch.tensor(states, dtype=torch.float).to(self.actor.device)
        new_states = torch.tensor(new_states, dtype=torch.float).to(self.actor.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.actor.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.actor.device)
        dones = torch.tensor(dones).to(self.actor.device)
        

        states_value = self.value(states).view(-1)
        new_states_value = self.target_value(new_states).view(-1)
        new_states_value[dones] = 0.0
        
        action, log_probs = self.actor.sample_normal(states, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1(states, action)
        q2_new_policy = self.critic_2(states, action)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(states_value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        action, log_probs = self.actor.sample_normal(states, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1(states, action)
        q2_new_policy = self.critic_2(states, action)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1) 

        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q_hat = self.scale*rewards + self.gamma*new_states_value
        q1_old_policy = self.critic_1(states, actions).view(-1)
        q2_old_policy = self.critic_2(states, actions).view(-1)
        critic1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic1_loss + critic2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        
        self.update_network_parameters()
        
        return 0, value_loss, actor_loss, critic_loss
    
    def learn_sm(self):
        if self.memeory.mem_cntr < self.batch_size:
            return

        states, new_states, actions, rewards, dones = self.memeory.sample_buffer(self.batch_size)
        states = torch.tensor(states, dtype=torch.float).to(self.actor.device)
        new_states = torch.tensor(new_states, dtype=torch.float).to(self.actor.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.actor.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.actor.device)
        dones = torch.tensor(dones).to(self.actor.device)


        states_value = self.value(states).view(-1)
        new_states_value = self.target_value(new_states).view(-1)
        new_states_value[dones] = 0.0

#         action, log_probs = self.actor.sample_normal(states, reparameterize=False)
#         log_probs = log_probs.view(-1)
#         q1_new_policy = self.critic_1(states, action)
#         q2_new_policy = self.critic_2(states, action)
#         critic_value = torch.min(q1_new_policy, q2_new_policy)
#         critic_value = critic_value.view(-1)

#         self.value.optimizer.zero_grad()
#         value_target = critic_value - log_probs
#         value_loss = 0.5 * F.mse_loss(states_value, value_target)
#         value_loss.backward(retain_graph=True)
#         self.value.optimizer.step()

#         action, log_probs = self.actor.sample_normal(states, reparameterize=True)
        action, _ = self.actor.sample_normal(states, reparameterize=True)
#         log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1(states, action)
        q2_new_policy = self.critic_2(states, action)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1) 

        # sample actions for next batch states
        action_next, _ = self.actor.sample_normal(new_states, reparameterize=True)
        q1_new_policy = self.critic_1(new_states, action_next)
        q2_new_policy = self.critic_2(new_states, action_next)
        critic_value_next = torch.min(q1_new_policy, q2_new_policy)
        critic_value_next = critic_value.view(-1) 

#         actor_loss = log_probs - critic_value
        actor_loss = - (critic_value + critic_value_next) + F.mse_loss(action,action_next) 
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

#         self.critic_1.optimizer.zero_grad()
#         self.critic_2.optimizer.zero_grad()

#         q_hat = self.scale*rewards + self.gamma*new_states_value
#         q1_old_policy = self.critic_1(states, actions).view(-1)
#         q2_old_policy = self.critic_2(states, actions).view(-1)
#         critic1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
#         critic2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
#         critic_loss = critic1_loss + critic2_loss
#         critic_loss.backward()
#         self.critic_1.optimizer.step()
#         self.critic_2.optimizer.step()

#         self.update_network_parameters()

        return 0, 0, actor_loss, 0




