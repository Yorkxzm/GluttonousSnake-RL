# GluttonousSnake-RL
这是一个使用强化学习玩贪吃蛇的项目。
贪吃蛇环境基于gym和pygame，实现参考了：
https://www.cnblogs.com/dengfaheng/p/9241267.html<br>
该环境允许游戏设置为变速模式：<br>
当设置 speedchange=True时
随着时间的推移蛇的速度会逐渐增加。

本文共实现了两种强化学习算法。PPO（在线学习）与SAC(离线学习），并对比了他们的训练效果。

环境的状态参考了https://github.com/ZYunfeii/DRL4SnakeGame。<br>
这里仅采用6维状态就可以得到一个较好的模型：
1.蛇头与食物的相对x与y坐标（2维）
2.蛇头上下左右是否有边界（4维）
