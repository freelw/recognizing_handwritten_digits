import gym
import warnings
import numpy as np

warnings.filterwarnings('ignore')


# 导入环境并查看观测空间和动作空间

env = gym.make('MountainCar-v0')
print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('观测范围 = {} ~ {}'.format(env.observation_space.low, env.observation_space.high))
print('动作数 = {}'.format(env.action_space.n))


# 根据指定确定性策略决定动作的智能体

class BespokeAgent(object):
    def __init__(self, env):
        pass
    
    def decide(self, observation):
        position, velocity = observation
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03, 0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.06
        
        if lb < velocity < ub:
            action = 2
        else:
            action = 0
        return action

    def learn(self, *args):
        pass

# 智能体和环境交互一个回合的代码

def play_montecarlo(env, agent, render=False, train=False):
    episode_reward = 0.0
    observation = env.reset()
    
    while True:
        if render:
            env.render()
        
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        
        if train:
            agent.learn(observation, action, reward, done)
        if done:
            break
        
        observation = next_observation
    
    return episode_reward

agent = BespokeAgent(env)
env.seed(0)

episode_reward = play_montecarlo(env, agent, render=True)
print('回合奖励 = {}'.format(episode_reward))
env.close()

# 运行100回合求平均以测试性能

episode_rewards = [play_montecarlo(env, agent) for _ in range(100)]
print('平均回合奖励 = {}'.format(np.mean(episode_rewards)))