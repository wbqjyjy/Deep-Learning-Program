from __future__ import print_function
import copy

MAP = \
    '''
.........
.       .
.     o .
.       .
.........
'''
# MAP = \
#     '''
# .........
# .  x    .
# .   x o .
# .       .
# .........
# '''

MAP = MPA.strip().split('\n')
MAP = [[c for c in line] for line in MAP]

DX = [-1,1,0,0]
DY = [0,0,-1,1]

class Env(object):
    def __init__(self):
        self.map = copy.deepcopy(MAP)
        self.x = 1 #self.x, self.y 存储了当前智能体的位置
        self.y = 1
        self.step = 0
        self.total_reward = 0
        self.is_end = False #智能体是否得到宝藏

    def interact(self,action):
        assert self.is_end is False
        new_x = self.x + DX[action] #智能体在x轴方向新的动作
        new_y = self.y + DY[action] #智能体在y轴方向新的动作
        new_pos_char = self.map[new_x][new_y] 
        self.step += 1
        if new_pos_char == '.':
            reward = 0 #如果智能体触到边界，奖励为0
        elif new_pos_char == ' ':
            self.x = new_x
            self.y = new_y #如果智能体走在迷宫里，则更新其位置信息，此时，奖励为0
            reward = 0
        elif new_pos_char == 'o':
            self.x = new_x
            self.y = new_y
            self.map[new_x][new_y] = ' '
            self.is_end = True #如果智能体找到宝藏，则游戏结束，给予奖励
            reward = 100
        elif new_pos_char == 'x': #将地图改为带有陷阱的地图，x代表陷进，智能体掉入陷进，reward=-5
            self.x = new_x
            self.y = new_y
            self.map[new_x][new_y] = ' '
            reward = -5 #如果智能体走到 x，给惩罚
        self.total_reward += reward #判断智能体位置，给出奖励后，更新奖励
        return reward

    def state_num(self):
        rows = len(self.map)
        cols = len(self.map[0])
        return rows * cols #给出智能体可能的状态 个数

    def present_state(self):
        cols = len(self.map[0])
        return self.x * cols + self.y #给出智能体当前所在的状态 ；计算方法？

    def print_map(self):
        printed_map = copy.deepcopy(self.map)
        printed_map[self.x][self.y] = 'A'
        print('\n'.join([''.join([c for c in line]) for line in printed_map])) #打印出A在map中的位置

    def print_map_with_reprint(self,output_list):
        printed_map = copy.deepcopy(self.map)
        printed_map[self.x][self.y] = 'A'
        printed_list = [''.join([c for c in line]) for line in printed_map]
        for i,line in enumerate(printed_list):
            output_list[i] = line

            
        
        
