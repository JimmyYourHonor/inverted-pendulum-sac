import pybullet_envs
import gym
import numpy as np
from sac_torch import Agent, Agent_sm
import logging
import sys
import os
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join('tmp', 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if __name__ == '__main__':
    env = gym.make('InvertedPendulumBulletEnv-v0')
    #print(env.action_space.shape[0])
    agent = Agent_sm(input_dims=env.observation_space.shape[0], env=env, 
                n_actions=env.action_space.shape[0])
    n_games = 200
    best_score = env.reward_range[0]
    score_history = []
    load_checkpoints = False

    if load_checkpoints:
        agent.load_models()
        #env.render(mode='human')
    
    #save losses: final_loss, value_loss, actor_loss, critic_loss
    final_loss = []
    value_loss = []
    actor_loss = []
    critic_loss = []
    action_record = []
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        step = 0
        while not done:
            action = agent.choose_action(observation)
            action_record.append(action)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            #if not load_checkpoints:
            l = agent.learn()
            if l is not None:
                final_loss.append(l[0])
                value_loss.append(l[1])
                actor_loss.append(l[2])
                critic_loss.append(l[3])
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

#         if avg_score > best_score:
#             best_score = avg_score
#             #if not load_checkpoints:
#             agent.save_models()
        logging.info('episode %d score %.1f avg score %.1f', i, score, avg_score)
    agent.save_models()
    PATH = 'losses.pt'
    torch.save({
            'value_loss': value_loss,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'actions': action_record
            }, PATH)

    #save losses: final_loss, value_loss, actor_loss, critic_loss
    n_games_sm = 50
    final_loss_sm = []
    value_loss_sm = []
    actor_loss_sm = []
    critic_loss_sm = []
    action_record_sm = []
    for i in range(n_games_sm):
        observation = env.reset()
        done = False
        score = 0
        step = 0
        while not done:
            action = agent.choose_action(observation)
            action_record_sm.append(action)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            #if not load_checkpoints:
            l = agent.learn_sm()
            if l is not None:
                final_loss_sm.append(l[0])
                value_loss_sm.append(l[1])
                actor_loss_sm.append(l[2])
                critic_loss_sm.append(l[3])
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

    #         if avg_score > best_score:
    #             best_score = avg_score
    #             #if not load_checkpoints:
    #             agent.save_models()
        logging.info('episode %d score %.1f avg score %.1f', i, score, avg_score)
        
    agent.save_models()
    PATH = 'losses_sm.pt'
    torch.save({
            'value_loss': value_loss_sm,
            'actor_loss': actor_loss_sm,
            'critic_loss': critic_loss_sm,
            'actions': action_record_sm
            }, PATH)