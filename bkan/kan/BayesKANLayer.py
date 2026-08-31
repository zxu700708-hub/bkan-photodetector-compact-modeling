import torch
import torch.nn as nn
import numpy as np
from .spline import *
from .utils import sparse_mask
import torch.nn.functional as F

class BayesKANLayer(nn.Module):
    """
    贝叶斯 KAN 层 - 在参数中引入不确定性
    
    使用变分推断（Bayes by Backprop）方法，为 spline 系数和尺度参数
    添加概率分布。支持通过蒙特卡洛采样进行不确定性量化。

    Attributes:
    -----------
        in_dim: int
            输入维度
        out_dim: int
            输出维度
        num: int
            网格区间数
        k: int
            B-spline 多项式阶数
        
        coef_mu: Parameter
            B-spline 系数的均值，shape (in_dim, out_dim, num+k)
        coef_log_sigma: Parameter
            B-spline 系数对数方差，shape (in_dim, out_dim, num+k)
            
        scale_base_mu: Parameter
            基函数尺度的均值，shape (in_dim, out_dim)
        scale_base_log_sigma: Parameter
            基函数尺度的对数方差，shape (in_dim, out_dim)
            
        scale_sp_mu: Parameter
            spline 尺度的均值，shape (in_dim, out_dim)
        scale_sp_log_sigma: Parameter
            spline 尺度的对数方差，shape (in_dim, out_dim)
            
        prior_mu: float
            先验分布均值（通常为 0）
        prior_log_sigma: float
            先验分布对数标准差（固定值，如 log(1) = 0）
            
        device: str
            计算设备
    """

    def __init__(self, 
                 in_dim=3, 
                 out_dim=2, 
                 num=5, 
                 k=3, 
                 noise_scale=0.5, 
                 scale_base_mu=0.0, 
                 scale_base_sigma=1.0, 
                 scale_sp=1.0, 
                 base_fun=torch.nn.SiLU(), 
                 grid_eps=0.02, 
                 grid_range=[-1, 1], 
                 sp_trainable=True, 
                 sb_trainable=True,
                 save_plot_data=True, 
                 device='cpu', 
                 sparse_init=False,
                 prior_mu=0.0,
                 prior_log_sigma=0.0,
                 posterior_init_sigma=0.1):
        """
        初始化贝叶斯 KAN 层
        
        Args:
        -----
            in_dim : int
                输入维度。默认: 3
            out_dim : int
                输出维度。默认: 2
            num : int
                网格区间数。默认: 5
            k : int
                B-spline 阶数。默认: 3
            noise_scale : float
                初始化噪声尺度。默认: 0.5
            scale_base_mu : float
                基函数尺度均值初始化。默认: 0.0
            scale_base_sigma : float
                基函数尺度标准差初始化。默认: 1.0
            scale_sp : float
                spline 尺度。默认: 1.0
            base_fun : function
                基函数。默认: SiLU()
            grid_eps : float
                网格自适应参数。默认: 0.02
            grid_range : list
                网格范围。默认: [-1, 1]
            sp_trainable : bool
                spline 尺度是否可训练
            sb_trainable : bool
                基函数尺度是否可训练
            device : str
                计算设备。默认: 'cpu'
            sparse_init : bool
                是否使用稀疏初始化
            prior_mu : float
                先验均值。默认: 0.0
            prior_log_sigma : float
                先验对数标准差。默认: 0.0（对应标准差为 1）
            posterior_init_sigma : float
                后验初始标准差。默认: 0.01
        """
        super(BayesKANLayer, self).__init__()
        
        # 基本参数
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k
        self.device = device
        
        # 先验参数
        self.prior_mu = prior_mu
        self.prior_log_sigma = prior_log_sigma
        
        # 初始化网格
        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None, :].expand(self.in_dim, num + 1)
        grid = extend_grid(grid, k_extend=k)
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        
        # ==================== 系数参数化 ====================
        # 从标准 KAN 的初始化中获取均值
        noises = (torch.rand(self.num + 1, self.in_dim, self.out_dim) - 1 / 2) * noise_scale / num
        coef_init = curve2coef(self.grid[:, k:-k].permute(1, 0), noises, self.grid, k)
        
        # 系数均值和对数方差
        self.coef_mu = torch.nn.Parameter(coef_init).requires_grad_(True)
        self.coef_log_sigma = torch.nn.Parameter(
            torch.ones_like(coef_init) * np.log(posterior_init_sigma)
        ).requires_grad_(True)
        
        # ==================== 基函数尺度参数化 ====================
        scale_base_init = scale_base_mu * 1 / np.sqrt(in_dim) + \
                          scale_base_sigma * (torch.rand(in_dim, out_dim) * 2 - 1) * 1 / np.sqrt(in_dim)
        
        self.scale_base_mu = torch.nn.Parameter(scale_base_init).requires_grad_(sb_trainable)
        self.scale_base_log_sigma = torch.nn.Parameter(
            torch.ones(in_dim, out_dim) * np.log(posterior_init_sigma)
        ).requires_grad_(sb_trainable)
        
        # ==================== Spline 尺度参数化 ====================
        if sparse_init:
            mask = sparse_mask(in_dim, out_dim)
        else:
            mask = torch.ones(in_dim, out_dim)
            
        scale_sp_init = torch.ones(in_dim, out_dim) * scale_sp * 1 / np.sqrt(in_dim) * mask
        
        self.scale_sp_mu = torch.nn.Parameter(scale_sp_init).requires_grad_(sp_trainable)
        self.scale_sp_log_sigma = torch.nn.Parameter(
            torch.ones(in_dim, out_dim) * np.log(posterior_init_sigma)
        ).requires_grad_(sp_trainable)
        
        # ==================== 掩码 ====================
        self.mask = torch.nn.Parameter(mask).requires_grad_(False)
        
        # ==================== 其他参数 ====================
        self.base_fun = base_fun
        self.grid_eps = grid_eps
        self.save_plot_data = save_plot_data
        
        self.to(device)
    
    def to(self, device):
        super(BayesKANLayer, self).to(device)
        self.device = device
        return self

    

    def sample_parameters(self):
    
        
        coef_sigma = torch.clamp(F.softplus(self.coef_log_sigma), max=5.0)
        scale_base_sigma = torch.clamp(F.softplus(self.scale_base_log_sigma), max=5.0)
        scale_sp_sigma = torch.clamp(F.softplus(self.scale_sp_log_sigma), max=5.0)
        # ===== 采样 =====
        eps_coef = torch.randn_like(self.coef_mu)
        coef_sample = self.coef_mu + eps_coef * coef_sigma

        eps_base = torch.randn_like(self.scale_base_mu)
        scale_base_sample = self.scale_base_mu + eps_base * scale_base_sigma

        eps_sp = torch.randn_like(self.scale_sp_mu)
        scale_sp_sample = self.scale_sp_mu + eps_sp * scale_sp_sigma

        return coef_sample, scale_base_sample, scale_sp_sample
    
    def forward(self, x, sample=True):
        """
        贝叶斯 KAN 层前向传播
        
        Args:
        -----
            x : Tensor
                输入，shape (batch_size, in_dim) 或 (batch_size,)
            sample : bool
                是否从后验采样参数。默认: True
                如果 False，使用均值（类似于标准 KAN）
        
        Returns:
        --------
            y : Tensor
                输出，shape (batch_size, out_dim)
            preacts : Tensor
                预激活，shape (batch_size, out_dim, in_dim)
            postacts : Tensor
                后激活，shape (batch_size, out_dim, in_dim)
            postspline : Tensor
                spline 输出，shape (batch_size, out_dim, in_dim)
        """
        # ✅ 修复：确保输入总是 2D
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # (batch_size,) -> (batch_size, 1)
        elif x.dim() != 2:
            raise ValueError(f"Expected input to be 1D or 2D, got {x.dim()}D")
        
        batch = x.shape[0]
        
        # 验证输入维度
        if x.shape[1] != self.in_dim:
            raise ValueError(f"Expected input dimension {self.in_dim}, got {x.shape[1]}")
        
        # 预激活（展开输入）
        preacts = x[:, None, :].clone().expand(batch, self.out_dim, self.in_dim)
        
        # 基函数
        base = self.base_fun(x)  # (batch, in_dim)
        
        # 采样或使用均值
        if sample:
            coef, scale_base, scale_sp = self.sample_parameters()
        else:
            coef = self.coef_mu
            scale_base = self.scale_base_mu
            scale_sp = self.scale_sp_mu
        
        # Spline 项
        y = coef2curve(x_eval=x, grid=self.grid, coef=coef, k=self.k)
        postspline = y.clone().permute(0, 2, 1)
        
        # 组合基函数和 spline
        y = scale_base[None, :, :] * base[:, :, None] + scale_sp[None, :, :] * y
        y = self.mask[None, :, :] * y
        
        postacts = y.clone().permute(0, 2, 1)
        
        # 求和得到输出
        y = torch.sum(y, dim=1)
        
        return y, preacts, postacts, postspline
    
    

    def kl_divergence(self):
        # ===== 使用 softplus 对应的 sigma =====
        # Use the same effective posterior standard deviations as stochastic
        # forward sampling.  Keeping the cap in both places avoids optimizing
        # a KL term for a different q(theta) than the one used by the
        # likelihood estimator.
        coef_sigma = torch.clamp(F.softplus(self.coef_log_sigma), max=5.0)
        coef_log_sigma_eff = torch.log(coef_sigma + 1e-8)

        scale_base_sigma = torch.clamp(
            F.softplus(self.scale_base_log_sigma), max=5.0
        )
        scale_base_log_sigma_eff = torch.log(scale_base_sigma + 1e-8)

        scale_sp_sigma = torch.clamp(
            F.softplus(self.scale_sp_log_sigma), max=5.0
        )
        scale_sp_log_sigma_eff = torch.log(scale_sp_sigma + 1e-8)

        # ===== KL 计算 =====
        kl = 0.0

        kl += self._kl_gaussian(
            self.coef_mu,
            coef_log_sigma_eff,
            self.prior_mu,
            self.prior_log_sigma
        ).sum()

        kl += self._kl_gaussian(
            self.scale_base_mu,
            scale_base_log_sigma_eff,
            self.prior_mu,
            self.prior_log_sigma
        ).sum()

        kl += self._kl_gaussian(
            self.scale_sp_mu,
            scale_sp_log_sigma_eff,
            self.prior_mu,
            self.prior_log_sigma
        ).sum()

        return kl
    
    @staticmethod
    def _kl_gaussian(mu_q, log_sigma_q, mu_p=0.0, log_sigma_p=0.0):
        """
        两个高斯分布之间的 KL 散度
        
        Args:
        -----
            mu_q : Tensor
                后验均值
            log_sigma_q : Tensor
                后验对数标准差
            mu_p : float or Tensor
                先验均值
            log_sigma_p : float or Tensor
                先验对数标准差
                
        Returns:
        --------
            kl : Tensor
                逐元素 KL 散度
        """
        # ===== 类型对齐（关键修复）=====
        if not torch.is_tensor(log_sigma_p):
            log_sigma_p = torch.tensor(log_sigma_p, device=mu_q.device, dtype=mu_q.dtype)

        if not torch.is_tensor(mu_p):
            mu_p = torch.tensor(mu_p, device=mu_q.device, dtype=mu_q.dtype)
        sigma_q = torch.exp(log_sigma_q)
        sigma_p = torch.exp(log_sigma_p)
        
        kl = log_sigma_p - log_sigma_q + \
             (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2 * sigma_p ** 2) - 0.5
        
        return kl
    
    def update_grid_from_samples(self, x, mode='sample'):
        """
        从样本更新网格 (修复版：完美匹配标准 KAN 的维度对齐)
        """
        batch = x.shape[0]
        x_pos = torch.sort(x, dim=0)[0]
        
        # 获取纯净的核心区间数量
        num_interval = self.grid.shape[1] - 1 - 2 * self.k
        
        # 1. 生成基于数据密度的自适应核心网格
        ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
        grid_adaptive = x_pos[ids, :].permute(1, 0)
        
        # 2. 提取当前的核心均匀网格 (掐头去尾，去掉 k_extend)
        grid_uniform = self.grid.data[:, self.k : -self.k]
        
        # 3. 完美混合 (维度绝对一致)
        grid_new = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        
        # 4. 重新扩展边缘并覆盖
        self.grid.data = extend_grid(grid_new, k_extend=self.k)
