from __future__ import print_function #使得python2,3兼容print
import numpy as np
import time
from env import Env

EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
MAX_STEP = 50

np.random.seed(1)

def epsilon_greedy(Q,state): #epsilon_greedy()：按照一定的概率来决定智能体下一步的action是随机产生，还是选取Q最大的那一个action
    if (np.random.uniform() > 1 - EPSILON) or ((Q[state,:] == 0).all()):
        action = np.random.randint(0,4) #随机选取一个action
    else:
        action = Q[state,:].argmax()
    return action

e=Env() #迷宫环境
Q=np.zeros((e.state_num,4)) #初始化Q

for i in range(200):
    e = Env() #迷宫环境
    acion = epsilon_greedy(Q, e.present_state) #根据当前状态，利用epsilon_greedy()给出智能体下一步的action
    while (e.is_end is False) and (e.step < MAX_STEP):
        state = e.present_state #未action时的 state
        reward = e.interact(action) #得到action所带来的奖励
        new_state = e.present_state #action以后的state
        new_action = epsilon_greedy(Q,e.present_state) #得出在new_state下智能体的new_action
        Q[state,action] = (1-ALPHA) * Q[state,action] + ALPHA * (reward + GAMMA * Q[new_state,new_action]) #sarsa算法与Q Learning算法的区别所在，即Q[state,action]的更新函数不同
        action = new_action #用于下一步 new_state状态下，智能体的action
        e.print_map()
        time.sleep(0.1)
    print('Episode:',i,'Total Step:',e.step,'Total Reward:',e.total_reward)
    time.sleep(2) #没结束一次游戏，休息2s
    
