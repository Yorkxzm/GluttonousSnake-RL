import random
import pygame
import sys
from pygame.locals import *
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
#服务器需要开启
import copy
#贪吃蛇环境
class GluttonousSnake:#该环境状态由计算得到，可以快速训练一个表现不错的AI
    def __init__(self,speedchange=False,state_dim=6):
        self.state_dim=state_dim
        self.snake_speed = 10 # 贪吃蛇的速度
        self.snake_speedstart=self.snake_speed#初始速度
        self.windows_width = 1000
        self.windows_height = 700  # 游戏窗口的大小
        self.cell_size = 50  # 贪吃蛇身体方块大小,注意身体大小必须能被窗口长宽整除
        self.map_width = int(self.windows_width / self.cell_size)
        self.map_height = int(self.windows_height / self.cell_size)
        self.black = (0, 0, 0)
        self.white=(255,255,255)
        self.gray = (230, 230, 230)
        self.dark_gray = (40, 40, 40)
        self.DARKGreen = (0, 155, 0)
        self.Green = (0, 255, 0)
        self.Red = (255, 0, 0)
        self.blue = (0, 0, 255)
        self.dark_blue = (0, 0, 139)
        self.headcolor=(0, 200, 0)#蛇头颜色
        self.BG_COLOR = (2,2,2)  # 游戏背景颜色
        # 定义方向
        self.UP = 0
        self.DOWN = 1
        self.LEFT = 2
        self.RIGHT = 3
        self.speedchange=speedchange
        self.HEAD = 0  # 贪吃蛇头部下标
        self.speedadds=0#增加速度的判断，到达一定程度应该加一些速
        pygame.init()  # 模块初始化
        self.snake_speed_clock = pygame.time.Clock()  # 创建Pygame时钟对象

        [self.snake_coords,self.direction,self.food,self.state] = [None,None,None,None]

    def reset(self):
        startx = random.randint(3, self.map_width - 8)  # 开始位置
        starty = random.randint(3, self.map_height - 8)
        self.snake_coords = [{'x': startx, 'y': starty},  # 初始贪吃蛇
                        {'x': startx - 1, 'y': starty},
                        {'x': startx - 2, 'y': starty}]
        self.direction = self.RIGHT  # 开始时向右移动
        self.food = self.get_random_location()  # 实物随机位置
        self.snake_speed=self.snake_speedstart
        return self.getState()

    def step(self,action):
        if action == self.LEFT and self.direction != self.RIGHT:
            self.direction = self.LEFT
        elif action == self.RIGHT and self.direction != self.LEFT:
            self.direction = self.RIGHT
        elif action == self.UP and self.direction != self.DOWN:
            self.direction = self.UP
        elif action == self.DOWN and self.direction != self.UP:
            self.direction = self.DOWN
        self.move_snake(self.direction,self.snake_coords)
        ret = self.snake_is_alive(self.snake_coords)
        d = True if not ret else False
        flag = self.snake_is_eat_food(self.snake_coords, self.food)
        reward = self.getReward(flag,d)
        self.speedadds+=1
        if self.speedchange==True:
            if self.speedadds==100:#一定时间过去了，蛇应该加速
                self.snake_speed+=2
                self.speedadds=0
        return [self.getState(),reward,d,None]

    def getReward(self,flag, d):
        reward = 0
        if flag:
            reward += 1#吃到东西给1
            if self.speedchange==True:
                self.snake_speed-=0.1#变速模式下吃到豆子可以让速度有所缓解，但不多
        if d: reward -= 15
        return reward
    def render(self):
        self.screen = pygame.display.set_mode((self.windows_width, self.windows_height))
        self.screen.fill(self.BG_COLOR)
        self.draw_snake(self.screen,self.snake_coords)
        self.draw_food(self.screen,self.food)
        self.draw_score(self.screen,len(self.snake_coords)-3)
        pygame.display.update()
        self.snake_speed_clock.tick(self.snake_speed) #控制fps

    def getState(self):
        #蛇头的x，y坐标与蛇头与食物的相对x,y坐标
        [xhead, yhead] = [self.snake_coords[self.HEAD]['x'], self.snake_coords[self.HEAD]['y']]
        [xfood, yfood] = [self.food['x'], self.food['y']]
        deltax = (xfood - xhead) / self.map_width
        deltay = (yfood - yhead) / self.map_height
        checkPoint = [[xhead,yhead-1],[xhead-1,yhead],[xhead,yhead+1],[xhead+1,yhead]]
        tem = [0,0,0,0]
        for coord in self.snake_coords[1:]:
            if [coord['x'],coord['y']] in checkPoint:
                index = checkPoint.index([coord['x'],coord['y']])
                tem[index] = 1
        for i,point in enumerate(checkPoint):
            if point[0]>=self.map_width or point[0]<0 or point[1]>=self.map_height or point[1]<0:
                tem[i] = 1
        state = [deltax,deltay]
        state.extend(tem)
        if self.state_dim==10:
            length = len(self.snake_coords)
            snake_mid = [self.snake_coords[int(length/2)]['x']-xhead,self.snake_coords[int(length/2)]['y']-yhead]
            snake_tail = [self.snake_coords[-1]['x']-xhead,self.snake_coords[-1]['y']-yhead]
            state.extend(snake_mid+snake_tail)
        return state#共6维,增广是10维

    def draw_food(self,screen, food):
        x = food['x'] * self.cell_size
        y = food['y'] * self.cell_size
        appleRect = pygame.Rect(x, y, self.cell_size, self.cell_size)
        pygame.draw.rect(screen, self.Red, appleRect)

    # 将贪吃蛇画出来
    def draw_snake(self,screen, snake_coords):
        for i,coord in enumerate(snake_coords):
            x = coord['x'] * self.cell_size
            y = coord['y'] * self.cell_size
            wormSegmentRect = pygame.Rect(x, y, self.cell_size, self.cell_size)
            pygame.draw.rect(screen, self.Green, wormSegmentRect)#蛇身为绿色
            if i==0:#蛇头
                wormInnerSegmentRect = pygame.Rect( 
                    x + 4, y + 4, self.cell_size - 8, self.cell_size - 8)
                pygame.draw.rect(screen, self.headcolor, wormInnerSegmentRect)

    # 移动贪吃蛇
    def move_snake(self,direction, snake_coords):
        if direction == self.UP:
            newHead = {'x': snake_coords[self.HEAD]['x'], 'y': snake_coords[self.HEAD]['y'] - 1}
        elif direction == self.DOWN:
            newHead = {'x': snake_coords[self.HEAD]['x'], 'y': snake_coords[self.HEAD]['y'] + 1}
        elif direction == self.LEFT:
            newHead = {'x': snake_coords[self.HEAD]['x'] - 1, 'y': snake_coords[self.HEAD]['y']}
        elif direction == self.RIGHT:
            newHead = {'x': snake_coords[self.HEAD]['x'] + 1, 'y': snake_coords[self.HEAD]['y']}
        else:
            newHead = None
            raise Exception('error for direction!')

        snake_coords.insert(0, newHead)

    # 判断是否存活
    def snake_is_alive(self,snake_coords):
        tag = True
        if snake_coords[self.HEAD]['x'] == -1 or snake_coords[self.HEAD]['x'] == self.map_width or snake_coords[self.HEAD]['y'] == -1 or \
                snake_coords[self.HEAD]['y'] == self.map_height:
            tag = False 
        for snake_body in snake_coords[1:]:
            if snake_body['x'] == snake_coords[self.HEAD]['x'] and snake_body['y'] == snake_coords[self.HEAD]['y']:
                tag = False  
        return tag
    # 判断是否吃到食物
    def snake_is_eat_food(self,snake_coords, food):  
        flag = False
        if snake_coords[self.HEAD]['x'] == food['x'] and snake_coords[self.HEAD]['y'] == food['y']:
            while True:
                food['x'] = random.randint(2, self.map_width - 2)
                food['y'] = random.randint(2, self.map_height - 2)  # 实物位置重新设置
                tag = 0
                for coord in snake_coords:
                    if [coord['x'],coord['y']] == [food['x'],food['y']]:
                        tag = 1
                        break
                if tag == 1: continue
                break
            flag = True
        else:
            del snake_coords[-1]  
        return flag
    # 食物随机生成
    def get_random_location(self):
        return {'x': random.randint(2, self.map_width - 2), 'y': random.randint(2, self.map_height - 2)}
    # 画成绩
    def draw_score(self,screen, score):
        font = pygame.font.Font('scorefont.ttf', 20)
        scoreSurf = font.render('Score: %s' % score, True, self.white)
        scoreRect = scoreSurf.get_rect()
        scoreRect.topleft = (self.windows_width - 120, 10)
        screen.blit(scoreSurf, scoreRect)
    # 程序终止
    def terminate(self):
        pygame.quit()