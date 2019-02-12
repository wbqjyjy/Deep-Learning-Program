from __future__ import print_function
import numpy as np
import time
from env import Env

EPSILON = 0.1 #epsilon-greedy策略中的参数，用于表示：随机行走的 概率
ALPHA = 0.1 #更新Q函数的一个超参数，用于“平缓Q函数的更新”
GAMMA = 0.9 #更新Q函数的一个参数：用于平衡 reward 和 Q(new_state,:).max()
MAX_STEP = 30 #每次游戏所走的最大步数

np.random.seed(0)

def epsilon_greedy(Q, state): #根据当前状态，及Q函数，判断action
    if (np.random.uniform() > 1-EPSILON) or ((Q[state,:] == 0).all()): #如果符合该条件，智能体随机行走
        action = np.random.randint(0,4)
    else:
        action = Q[state,:].argmax()
    return action

e = Env()
Q =np.zeros((e.state_num,4)) #初始化的Q函数值

for i in range(200): #进行200轮的游戏
    e = Env()
    while (e.is_end is False) and (e.step < MAX_STEP): #如果没有找到宝藏，且，智能体所走步数小于最大步数
        action = epsilon_greedy(Q,e.present_state) #用epsilon()选择action
        state = e.present_state #执行动作之前的状态
        reward = e.interact(action) #通过智能体的行为结果，判断其得到什么样的奖励，并将这些奖励值赋给reward
        new_state = e.present_state #将智能体执行动作之后的“状态”赋给new_state
        Q[state,action] = (1 - ALPHA) * Q[state,action] + \ #之所以有这一部分，是想平滑Q的更新过程；
                          ALPHA * (reward + GAMMA * Q[new_state,:].max()) #利用这个公式更新“智能体行为前的Q[state,action]”
        #随着训练的不断进行，Q[state,action]会越来越符合 游戏目标，驱使 智能体向更接近“宝藏”的地方前进
        e.print_map() #打印智能体在map中的状态
        time.sleep(0.1)
    print('Episode:',i,'Total Step:',e.step,'Total Reward:',e.total_reward)
    time.sleep(2) #睡眠2s，在进行下一轮游戏
        
