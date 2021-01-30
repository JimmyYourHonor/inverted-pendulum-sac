import pybullet_envs
import gym
import numpy as np
from sac_torch import Agent

if __name__ == '__main__':
    env = gym.make('InvertedPendulumBulletEnv-v0')
    #print(env.action_space.shape[0])
    agent = Agent(input_dims=env.observation_space.shape[0], env=env, 
                n_actions=env.action_space.shape[0])
    n_games = 10
    best_score = env.reward_range[0]
    score_history = []
    load_checkpoints = True

    if load_checkpoints:
        agent.load_models()
        env.render(mode='human')
    
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            #if not load_checkpoints:
            #agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            #if not load_checkpoints:
            agent.save_models()
        
        print('episode ', i, ' score %.1f ' % score, 'avg score %.1f ' % avg_score)