import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#GAE:广义优势估计,用于估计优势函数
def compute_advantage(gamma,lamda,td_delta):
    td_delta=td_delta.detach().numpy()
    advantagelist=[]
    advantage=0.0
    for delta in td_delta[::-1]:
        advantage=gamma*lamda*advantage+delta
        advantagelist.append(advantage)
    advantagelist.reverse()
    return torch.tensor(advantagelist,dtype=float)
class PolicyNet(nn.Module):#策略网络
    def __init__(self,state_dim,hidden_dim,action_dim) :
        super(PolicyNet,self).__init__()
        self.fc1=nn.Linear(state_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)          
        self.fc3=nn.Linear(hidden_dim,action_dim)
    def forward(self,x):
        x=F.leaky_relu(self.fc1(x))
        x=F.leaky_relu(self.fc2(x))
        x=F.softmax(self.fc3(x),dim=1)#该softmax不能省去，输出归一化的概率
        return x
class ValueNet(nn.Module):#价值网络，目的是拟合价值函数V
    def __init__(self,state_dim,hidden_dim) :
        super(ValueNet,self).__init__()
        self.fc1=nn.Linear(state_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.fc3=nn.Linear(hidden_dim,1)
    def forward(self,x):
        x=F.leaky_relu(self.fc1(x))
        x=F.leaky_relu(self.fc2(x))
        x=self.fc3(x)
        return x
class QValueNet(nn.Module):#价值网络，目的是拟合价值函数V
    def __init__(self,state_dim,hidden_dim,action_dim) :
        super(QValueNet,self).__init__()
        self.fc1=nn.Linear(state_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.fc3=nn.Linear(hidden_dim,action_dim)
    def forward(self,x):
        x=F.leaky_relu(self.fc1(x))
        x=F.leaky_relu(self.fc2(x))
        x=self.fc3(x)
        return x
class PPO:
    def __init__(self,state_dim,hidden_dim,action_dim,learning_rate1,learning_rate2,gamma,lmbda,epochs,eps,device,savepath='model/PPOmodel/'):    
        self.device=device
        self.actor=PolicyNet(state_dim,hidden_dim,action_dim).to(self.device)
        self.critic=ValueNet(state_dim,hidden_dim).to(self.device)
        self.actoroptimizer=torch.optim.Adam(self.actor.parameters(),lr=learning_rate1)
        self.criticoptimizer=torch.optim.Adam(self.critic.parameters(),lr=learning_rate2)
        self.gamma=gamma#折扣因子
        self.epochs=epochs#KL距离最大参数
        self.eps=eps#PPO截断参数
        self.lmbda=lmbda
        self.savepath=savepath
        
    @property
    def savemodel(self):
        torch.save(self.actor.state_dict(),self.savepath+'actor.pth')
        torch.save(self.critic.state_dict(),self.savepath+'critic.pth')
    @property
    def loadmodel(self):
        self.actor.load_state_dict(torch.load(self.savepath+'actor.pth'))
        self.critic.load_state_dict(torch.load(self.savepath+'critic.pth'))       

    def take_action(self,state):
        state=torch.tensor([state],dtype=torch.float).to(self.device)
        probs=self.actor(state)
        action_dist=torch.distributions.Categorical(probs)
        action=action_dist.sample()
        return action.item()
    
    def update(self,transition_dict):#transition_dict 是外部传入的字典，用键访问元素。
        states=torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions=torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        rewards=torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1,1).to(self.device)
        next_states=torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones=torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1,1).to(self.device)
        td_target=rewards+self.gamma*self.critic(next_states)*(1-dones)
        td_delta=td_target-self.critic(states)
        advantage=compute_advantage(self.gamma,self.lmbda,td_delta.cpu()).to(self.device)
        old_log_probs=torch.log(self.actor(states).gather(1,actions)).detach()
        for i in range(self.epochs):
            log_probs=torch.log(self.actor(states).gather(1,actions))
            ratio=torch.exp(log_probs-old_log_probs)
            surr1=ratio*advantage
            surr2=torch.clamp(ratio,1-self.eps,1+self.eps)
            actor_loss=torch.mean(-torch.min(surr1,surr2))
            critic_loss=torch.mean(F.mse_loss(self.critic(states),td_target.detach()))
            self.actoroptimizer.zero_grad()
            actor_loss.backward()
            self.actoroptimizer.step()
            self.criticoptimizer.zero_grad()
            critic_loss.backward()
            self.criticoptimizer.step()

class SAC:
    ''' 处理离散动作的SAC算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 alpha_lr, target_entropy, tau, gamma, device,savepath='model/SACmodel/'):
        # 策略网络
        self.state_dim=state_dim
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        # 第一个Q网络
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 第二个Q网络
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = QValueNet(state_dim, hidden_dim,
                                         action_dim).to(device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNet(state_dim, hidden_dim,
                                         action_dim).to(device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.savepath=savepath
    @property
    def savemodel(self):
        torch.save(self.actor.state_dict(),self.savepath+'actor.pth')
        torch.save(self.critic_1.state_dict(),self.savepath+'critic1.pth')
        torch.save(self.critic_2.state_dict(),self.savepath+'critic2.pth')
        torch.save(self.target_critic_1.state_dict(),self.savepath+'target_critic_1.pth')
        torch.save(self.target_critic_2.state_dict(),self.savepath+'target_critic_2.pth')
    @property
    def loadmodel(self):
        self.actor.load_state_dict(torch.load(self.savepath+'actor.pth'))
        self.critic_1.load_state_dict(torch.load(self.savepath+'critic1.pth'))
        self.critic_2.load_state_dict(torch.load(self.savepath+'critic2.pth'))
        self.target_critic_1.load_state_dict(torch.load(self.savepath+'target_critic_1.pth'))
        self.target_critic_2.load_state_dict(torch.load(self.savepath+'target_critic_2.pth'))
    
    
    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']),dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']),dtype=torch.int64).view(-1,1).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']),dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']),dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']),dtype=torch.float).view(-1,1).to(self.device)

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_q_values = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(
            F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(
            F.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        # 直接根据概率计算熵
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  #
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)  # 直接根据概率计算期望
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)