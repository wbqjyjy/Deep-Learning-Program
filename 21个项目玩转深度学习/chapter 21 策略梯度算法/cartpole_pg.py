#coding:utf-8
from __future__ import division #进行精确除法
from __future__ import print_function #兼容Python2,3的print函数

import numpy as np
import tensorflow as tf
import gym

#gym环境
env = gym.make('CartPole-v0') #构建游戏环境

#超参数
D = 4 #输入层神经元个数 ：对应平台坐标，以及杆坐标
H = 10 #隐层神经元个数
batch_size = 5 #一个batch中有5个episode，即5次游戏
gamma = 0.99 #奖励折扣率

#定义policy网络
#输入观察值，输出右移的概率
observations = tf.placeholder(tf.float32,[None,D],name='input_x')
W1 = tf.get_variable('W1',shape=[D,H],initializer=tf.contrib.layers.xavier_initializer()) #第一个隐含层的权重
layer1 = tf.nn.relu(tf.matmul(observations,W1)) #第一个隐含层激活value
W2 = tf.get_variable('W2',shape=[H,1],initializer=tf.contrib.layers.xavier_initializer()) #第二个隐含层的权重
score = tf.matmul(layer1,W2) #计算Output
probability = tf.nn.sigmoid(score) #计算右移的概率值

#定义和训练、Loss有关的变量
tvars = tf.trainable_variables() #获得要训练的变量
input_y = tf.placeholder(tf.float32,[None,1],name='input_y') #loss = (1 - input_y)(input_y + probability) + (input_y)(input_y - probability) ；前者为action=1(右移)的概率，后者为action=0(左移)的概率；loss function为最大似然函数
advantages = tf.placeholder(tf.float32,name='reward_signal') #每次action的奖励
#input_y , advantages  : tf.placeholder()

#定义loss函数
loglik = tf.log(input_y * (input_y - probability) + （1 - input_y) * (input_y + probability))
loss = -tf.reduce_mean(loglik * advantages) #最终的损失函数：给出一个训练样本{input:(平台坐标，杆坐标），output：A}，其input为（平台坐标，杆坐标）,output为：A * sum(loglik)；当赢得一次游戏时(score>threshold)，令A=1, or else, A = -1；要想使得Loss很小，则当train_data是一个win时，A=1,要最小化Loss，则要最大化loglik，否则当train_data是一个lose时，A=-1,要想最小化loss，则要最小化loglik；其背后含义时，如果为lose时，最小化loglik意味着要最小化智能体此次action的概率值，即错误决策的概率值
newGrads = tf.gradients(loss,tvars) #计算梯度

#优化器、梯度
adam = tf.train.AdamOptimizer(learning_rate = learning_rate)
W1Grad = tf.placeholder(tf.float32,name='batch_grad1')
W2Grad = tf.placeholder(tf.float32,name='batch_grad2')
batchGrad = [W1Grad,W2Grad] #占位符：newGrads
updateGrads = adam.apply_gradients(zip(batchGrad,tvars)) #更新参数

def discount_rewards(r):
    """输入：1维的float类型数组，表示每个时刻的奖励；
       输出：计算折扣率gamma后的期望奖励"""
    discounted_r = np.zeros_like(r) #给出与r同shape的0数组
    running_add = 0 #??
    for t in reversed(range(0,r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[]
running_reward = None #???
reward_sum = 0 #???
episode_number = 1
total_episodes = 10000
init = tf.global_variables_initializer() #初始化全局变量

#开始训练
with tf.Session() as sess:
    rendering = False
    sess.run(init) #初始化全局变量
    #observation是环境的初始观察量（输入神经网络的值）
    observation = env.reset() #初始state

    #gradBuffer会存储梯度，此处做一初始化
    gradBuffer = sess.run(tvars)
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    #开始玩游戏
    while episode_number < total_episodes:
        #当一个batch内的reward_average>180时，显示游戏窗口
        if reward_sum / batch_size > 180 or rendering = True: #这里有歧义，到batch_size最后一轮时，在未计算最后一轮加入后的reward_sum情况下，就计算了reward_sum/batch_size求reward_average，而当计算了reward_sum后，最后一轮结束时，会清空reward_sum，在到下一轮循环时，reward_sum=0，无法计算reward_average！
            env.render()
            rendering = True 
            
        #先根据现有的“策略梯度”玩一局游戏
        #输入state，获得action
        x = np.reshape(observation,[1,D]) #input_state
        #如果获得action:策略梯度网络输出的是“向右移动的概率值”,如果random.uniform<prob，则智能体右移action=1,否则action=0
        tfprob = sess.run(probability,feed_dict = {observations:x})
        action = 1 if np.random.uniform() < tfprob else 0 #计算action
        #知道action以后，即可求得智能体在该action下的next_state和reward
        observation,reward,done,info = env.step(action) #state，reward，游戏是否结束，info
        #求reward_sum
        reward_sum += reward
        #记录每一步的observation,action,reward
        xs.append(x) #记录observation
        y = 1 if action == 0 else 0 #loss中的input_y
        ys.append(y) #记录input_y
        drs.append(reward) #记录每一步的reward

        #如果一局游戏结束，则利用该局游戏作为训练样本，进行训练
        if done:
            episode_number += 1 #游戏次数加1
            #将observation，input_y，reward存储list转为array
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)

            #对epr计算期望reward
            discounted_epr = discount_rewards(epr)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr //= np.std(discounted_epr)

            #计算梯度
            tGrad = sess.run(newGrads,feed_dict= {observation:epx,input_y:epy,advantages:discounted_epr})
            #将计算的梯度存到buffer里
            for ix,grad in enumerate(tGrad):
                gradBuffer[ix] += grad
            #如果游戏次数达到batch_size,更新参数
            if episode_number % batch_size == 0:
                #利用梯度更新参数
                sess.run(updateGrads,feed_dict={W1Grad:gradBuffer[0],W2Grad:gradBuffer[1]})
                #更新完参数以后，将gradBuffer中保存的梯度清零
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                #打印一些信息
                print('Episode: %d ~ %d Average reward: %f.  ' % (episode_number - batch_size + 1, episode_number, reward_sum // batch_size))

                #当我们在游戏中拿到200分以后，就停止训练
                if reward_sum // bath_size >= 200:
                    print("Task solved in", episode_number, 'episodes!')
                    break
                reward_sum = 0 #batch训练完后，将reward_sum清0，重新计算下一个batch

            #当完成一次游戏后，要重新初始化智能体state
            observation = env.reset()

    #当训练结束后，打印总共训练的episode次数
    print(episode_number, 'Episodes completed.')
        

        
        
        
    


















































    

