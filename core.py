# -*- coding: UTF-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Distribution, Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
ACTION_BOUND_EPSILON = 1E-6
# 参数取值来自MBPO论文，在MuJoCo的Hopper环境中训练
mbpo_target_entropy_dict = {'Hopper-v2':-1, 'HalfCheetah-v2':-3, 'Walker2d-v2':-3, 'Ant-v2':-4, 'Humanoid-v2':-2}
mbpo_epoches = {'Hopper-v2':125, 'Walker2d-v2':300, 'Ant-v2':300, 'HalfCheetah-v2':400, 'Humanoid-v2':300}

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class ReplayBuffer:
    """
    FIFO experience replay buffer
    """
    def __init__(self, obs_dim, act_dim, size):
        """
        :param obs_dim: observation维数
        :param act_dim: action维数
        :param size: replay buffer大小
        """
        # 初始化buffer，以np矩阵的形式存储数据
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        """
        将数据存储到self.ptr所指位置
        """
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        # 更新self.ptr位置，向栈顶移动
        self.ptr = (self.ptr+1) % self.max_size
        ## keep track of the current buffer size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32, idxs=None):
        """
        :param batch_size: 此次采样的记录数量
        :param idxs: 若要取特定的记录，用idx索引
        """
        if idxs is None:
            idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    idxs=idxs)

# q net的网络结构
class Mlp(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_sizes,
            hidden_activation=F.relu
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation

        self.hidden_layers = nn.ModuleList()
        in_size = input_size

        # 初始化隐藏层
        for i, next_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(in_size, next_size)
            in_size = next_size
            self.hidden_layers.append(fc_layer)

        # 初始化最后一个全连接层
        self.last_fc_layer = nn.Linear(in_size, output_size)
        self.apply(weights_init_)

    def forward(self, input):
        h = input
        for i, fc_layer in enumerate(self.hidden_layers):
            h = fc_layer(h)
            h = self.hidden_activation(h)
        output = self.last_fc_layer(h)
        return output

class TanhNormal(Distribution):
    """
    用于表示X的分布：
        X ~ tanh(Z)
        Z ~ N(mean, std)
    原注释：
    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: 正态分布的均值
        :param normal_std: 正态分布的标准差
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def log_prob(self, value, pre_tanh_value=None):
        """
        返回value在分布中的对数概率
        """
        # use arctanh formula to compute arctanh(value)
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1+value) / (1-value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - \
               torch.log(1 - value * value + self.epsilon)

    def sample(self, return_pretanh_value=False):
        """
        采样
        原注释：
        Gradients will and should *not* pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        在重新参数化（reparameterization）情况下采样
        """
        z = (
            self.normal_mean +
            self.normal_std *
            Normal(
                torch.zeros(self.normal_mean.size()),
                torch.ones(self.normal_std.size())
            ).sample()
        )
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

# policy net的网络结构，有限制action的特性
class TanhGaussianPolicy(Mlp):
    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_sizes,
            hidden_activation=F.relu,
            action_limit=1.0
    ):
        super().__init__(
            input_size=obs_dim,
            output_size=action_dim,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
        )
        last_hidden_size = obs_dim
        if len(hidden_sizes) > 0:
            last_hidden_size = hidden_sizes[-1]
        # 输出log_std的层，用小的权值与偏差对其进行初始化
        self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
        # 限制action，原注释：
        # action limit: for example, humanoid has an action limit of -0.4 to 0.4
        self.action_limit = action_limit
        self.apply(weights_init_)

    def forward(
            self,
            obs,
            deterministic=False,
            return_log_prob=True,
    ):
        """
        :param reparameterize: 表示是否进行重新参数化
        :param deterministic: 表示是否是test阶段（test使用决策的action，区别于explore）
        :param return_log_prob: 表示是否返回样本的对数概率
        """
        h = obs
        for fc_layer in self.hidden_layers:
            h = self.hidden_activation(fc_layer(h))
        mean = self.last_fc_layer(h)

        log_std = self.last_fc_log_std(h)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        normal = Normal(mean, std)

        if deterministic:
            pre_tanh_value = mean
            action = torch.tanh(mean)
        else:
            pre_tanh_value = normal.rsample()
            action = torch.tanh(pre_tanh_value)

        if return_log_prob:
            log_prob = normal.log_prob(pre_tanh_value)
            log_prob -= torch.log(1 - action.pow(2) + ACTION_BOUND_EPSILON)
            log_prob = log_prob.sum(1, keepdim=True)
        else:
            log_prob = None

        return (
            action * self.action_limit, mean, log_std, log_prob, std, pre_tanh_value,
        )

def soft_update_model1_with_model2(model1, model2, rou):
    """
    对一个网络进行polyak更新
    """
    for model1_param, model2_param in zip(model1.parameters(), model2.parameters()):
        model1_param.data.copy_(rou*model1_param.data + (1-rou)*model2_param.data)

def test_agent(agent, test_env, max_ep_len, logger, n_eval=1):
    """
    测试训练的agent的表现，运行n_eval个episode
    在测试情况下，agent只能采取根据policy决策的action
    返回测试中所有epsiode的返回值
    """
    ep_return_list = np.zeros(n_eval)
    for j in range(n_eval):
        o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
        while not (d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time
            a = agent.get_test_action(o)
            o, r, d, _ = test_env.step(a)
            ep_ret += r
            ep_len += 1
        ep_return_list[j] = ep_ret
        if logger is not None:
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
    return ep_return_list
