import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .BayesKANLayer import BayesKANLayer
from .MultKAN import MultKAN


class BayesMultKAN(nn.Module):
    """
    贝叶斯 MultKAN 模型 - 支持不确定性量化的 KAN
    
    将多个贝叶斯 KAN 层堆叠，支持变分推断训练。
    支持 Epistemic (参数不确定性) 和 Aleatoric (模型方差) 两种不确定性。
    
    Attributes:
    -----------
        width : list
            网络宽度配置，例如 [2, 64, 64, 1]
        grid : int
            网格区间数
        k : int
            B-spline 阶数
        layers : ModuleList
            贝叶斯 KAN 层列表
        depth : int
            网络深度
        device : str
            计算设备
        kl_weight : float
            KL 散度的权重系数（用于 ELBO 损失）
        num_mc_samples : int
            蒙特卡洛采样的样本数（用于推断）
    """
    
    def __init__(self, 
                 width=[2, 64, 64, 1],
                 grid=5, 
                 k=3, 
                 noise_scale=0.3, 
                 scale_base_mu=0.0,
                 scale_base_sigma=1.0, 
                 base_fun='silu',
                 grid_eps=0.02, 
                 grid_range=[-1, 1],
                 sp_trainable=True, 
                 sb_trainable=True,
                 prior_mu=0.0,
                 prior_log_sigma=0.0,
                 posterior_init_sigma=0.01,
                 seed=1, 
                 device='cpu',
                 kl_weight=0.01,
                 num_mc_samples=100,
                 inference_method='vi',
                 dropout_rate=0.0):
        """
        初始化贝叶斯 MultKAN
        
        Args:
        -----
            width : list
                网络宽度配置。例如 [2, 64, 64, 1] 表示
                输入 2 维，两个隐层各 64 个神经元，输出 1 维
                
            grid : int
                网格区间数。默认: 5
                
            k : int
                B-spline 阶数。默认: 3
                
            noise_scale : float
                初始化噪声尺度。默认: 0.3
                
            scale_base_mu : float
                基函数尺度的初始均值。默认: 0.0
                
            scale_base_sigma : float
                基函数尺度初始化的标准差。默认: 1.0
                
            base_fun : str or callable
                基函数。默认: 'silu'
                支持: 'silu', 'relu', 'tanh', 或自定义函数
                
            grid_eps : float
                网格自适应参数。默认: 0.02
                
            grid_range : list
                网格范围。默认: [-1, 1]
                
            sp_trainable : bool
                spline 尺度是否可训练
                
            sb_trainable : bool
                基函数尺度是否可训练
                
            prior_mu : float
                先验均值。默认: 0.0
                
            prior_log_sigma : float
                先验对数标准差。默认: 0.0
                
            posterior_init_sigma : float
                后验初始标准差。默认: 0.01
                
            seed : int
                随机种子
                
            device : str
                计算设备。默认: 'cpu'
                
            kl_weight : float
                KL 散度在 ELBO 中的权重。默认: 0.01
                用于调整正则化强度
                
            num_mc_samples : int
                推断时的蒙特卡洛采样次数。默认: 100
        """
        super(BayesMultKAN, self).__init__()
        
        self.width = width
        self.grid = grid
        self.k = k
        self.depth = len(width) - 1
        self.device = device
        self.kl_weight = kl_weight
        self.num_mc_samples = num_mc_samples
        self.inference_method = inference_method
        self.dropout_rate = float(dropout_rate or 0.0)
        self.base_fun_name = base_fun if isinstance(base_fun, str) else None
        
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 设置基函数
        if isinstance(base_fun, str):
            if base_fun == 'silu':
                base_fn = torch.nn.SiLU()
            elif base_fun == 'relu':
                base_fn = torch.nn.ReLU()
            elif base_fun == 'tanh':
                base_fn = torch.nn.Tanh()
            else:
                raise ValueError(f"Unknown base function: {base_fun}")
        else:
            base_fn = base_fun
        
        self.base_fun_module = base_fn
        
        # 构建贝叶斯层堆栈
        self.layers = nn.ModuleList()
        for i in range(self.depth):
            layer = BayesKANLayer(
                in_dim=width[i],
                out_dim=width[i + 1],
                num=grid,
                k=k,
                noise_scale=noise_scale,
                scale_base_mu=scale_base_mu,
                scale_base_sigma=scale_base_sigma,
                scale_sp=1.0,
                base_fun=base_fn,
                grid_eps=grid_eps,
                grid_range=grid_range,
                sp_trainable=sp_trainable,
                sb_trainable=sb_trainable,
                prior_mu=prior_mu,
                prior_log_sigma=prior_log_sigma,
                posterior_init_sigma=posterior_init_sigma,
                device=device
            )
            self.layers.append(layer)
        
        self.to(device)
    
    def to(self, device):
        """移动模型到指定设备"""
        super(BayesMultKAN, self).to(device)
        self.device = device
        return self
    
    def forward(self, x, sample=True):
        """
        前向传播
        
        Args:
        -----
            x : Tensor
                输入，shape (batch_size, input_dim)
                
            sample : bool
                是否从后验采样参数。默认: True
                
        Returns:
        --------
            y : Tensor
                输出，shape (batch_size, output_dim)
                
            meta : dict
                元数据，包含中间激活等信息
        """
        batch = x.shape[0]
        meta = {}
        
        preacts_list = []
        postacts_list = []
        postspline_list = []
        
        y = x
        use_parameter_sampling = sample and self.inference_method == 'vi'
        for i, layer in enumerate(self.layers):
            y, preacts, postacts, postspline = layer(y, sample=use_parameter_sampling)
            if (
                self.inference_method == 'dropout'
                and self.dropout_rate > 0.0
                and i < self.depth - 1
            ):
                y = F.dropout(y, p=self.dropout_rate, training=True)
            
            preacts_list.append(preacts)
            postacts_list.append(postacts)
            postspline_list.append(postspline)
        
        meta['preacts'] = preacts_list
        meta['postacts'] = postacts_list
        meta['postspline'] = postspline_list
        
        
        return y, meta
    
    def forward_sample(self, x, num_samples=None):
        """
        蒙特卡洛采样前向传播
        用于推断时的不确定性量化
        
        Args:
        -----
            x : Tensor
                输入，shape (batch_size, input_dim)
                
            num_samples : int
                采样次数。默认: 使用 self.num_mc_samples
                
        Returns:
        --------
            samples : Tensor
                输出样本，shape (num_samples, batch_size, output_dim)
                
            mean : Tensor
                平均预测，shape (batch_size, output_dim)
                
            variance : Tensor
                预测方差，shape (batch_size, output_dim)
        """
        if num_samples is None:
            num_samples = self.num_mc_samples
        
        samples_list = []
        
        sample_parameters = self.inference_method == 'vi'
        for _ in range(num_samples):
            y, _ = self.forward(x, sample=sample_parameters)
            samples_list.append(y)
        
        samples = torch.stack(samples_list, dim=0)  # (num_samples, batch_size, output_dim)
        
        mean = samples.mean(dim=0)
        variance = samples.var(dim=0, unbiased=False)
        
        return samples, mean, variance
    
    def epistemic_uncertainty(self, x, num_samples=None):
        """
        计算认知不确定性 (Epistemic Uncertainty)
        这来自参数的不确定性，可以通过贝叶斯学习来减少
        
        Args:
        -----
            x : Tensor
                输入
                
            num_samples : int
                蒙特卡洛采样次数
                
        Returns:
        --------
            epi_uncertainty : Tensor
                认知不确定性方差
        """
        _, mean, variance = self.forward_sample(x, num_samples)
        
        # Epistemic = Var_w[E_data[y|x,w]] = Var_w[f_w(x)]
        epi_var = variance
        
        return epi_var
    
    def total_uncertainty(self, x, num_samples=None):
        """
        计算总不确定性
        
        Args:
        -----
            x : Tensor
                输入
                
            num_samples : int
                蒙特卡洛采样次数
                
        Returns:
        --------
            total_uncertainty : Tensor
                总不确定性标准差
        """
        _, mean, variance = self.forward_sample(x, num_samples)
        return torch.sqrt(variance)
    
    def kl_divergence(self):
        """
        计算所有层的总 KL 散度
        
        Returns:
        --------
            kl_total : float
                总 KL 散度
        """
        if self.inference_method != 'vi':
            device = next(self.parameters()).device
            return torch.zeros((), device=device)

        kl_total = 0.0
        
        for layer in self.layers:
            kl_total += layer.kl_divergence()
        
        return kl_total
    
    def elbo_loss(self, x, y_true, criterion=nn.MSELoss(reduction='mean'), num_samples=1):
        """
        计算 ELBO 损失用于训练
        
        ELBO = -E_q[log p(y|x)] + KL(q||p) / batch_size
        
        Args:
        -----
            x : Tensor
                输入，shape (batch_size, input_dim)
                
            y_true : Tensor
                目标，shape (batch_size, output_dim)
                
            criterion : Loss
                似然函数。默认: MSE Loss
                
            num_samples : int
                训练时的采样次数。默认: 1
                
        Returns:
        --------
            elbo : Tensor
                ELBO 损失（标量）
                
            nll : Tensor
                负对数似然
                
            kl : Tensor
                KL 散度
        """
        batch_size = x.shape[0]
        
        # 计算似然项（可以采样多次以获得更好的估计）
        nll_list = []
        for _ in range(num_samples):
            y_pred, _ = self.forward(x, sample=True)
            nll = criterion(y_pred, y_true)
            nll_list.append(nll)
        
        nll = torch.stack(nll_list).mean()
        
        # 计算 KL 项
        kl = self.kl_divergence()
        
        # ELBO = NLL + KL / batch_size
        elbo = nll + self.kl_weight * kl / batch_size
        
        return elbo, nll, kl
    
    def update_grid(self, train_loader):
        """
        使用训练数据更新所有层的网格
        ✅ 终极修复：逐层前向传播，使用真实到达各层的特征更新网格
        """
        # 1. 收集所有训练数据并拼成一个大 Batch，放到正确的设备上
        all_x = []
        for x_batch, _ in train_loader:
            all_x.append(x_batch.to(self.device))
        x = torch.cat(all_x, dim=0)
        
        # 2. 逐层推进，更新网格
        with torch.no_grad():
            for layer in self.layers:
                # (a) 用真实到达当前层的特征 x，更新该层的自适应网格
                layer.update_grid_from_samples(x)
                
                # (b) 跑一次确定性的前向传播（sample=False），计算出下一层将要接收的特征
                # BayesKANLayer 的 forward 返回: y, preacts, postacts, postspline
                x, _, _, _ = layer(x, sample=False)

    def to_deterministic_kan(self, device=None, symbolic_enabled=True, auto_save=False):
        """
        Export posterior means as a deterministic MultKAN.
        ✅ 异方差修复版：如果输出层是双通道(均值+方差)，则强制切片，
        只提取均值通道(通道 0)传递给确定性 KAN 用于后续的符号回归。
        """
        target_device = device or self.device
        if self.base_fun_name in ('silu', 'identity', 'zero'):
            base_fun = self.base_fun_name
        else:
            base_fun = self.base_fun_module

        # ==========================================
        # 1. 强制设定确定性 KAN 的输出维度为 1
        # ==========================================
        det_width = list(self.width)
        if det_width[-1] == 2:
            det_width[-1] = 1

        model = MultKAN(
            width=det_width,
            grid=self.grid,
            k=self.k,
            base_fun=base_fun,
            symbolic_enabled=symbolic_enabled,
            affine_trainable=False,
            grid_eps=self.layers[0].grid_eps if self.layers else 0.02,
            sp_trainable=True,
            sb_trainable=True,
            seed=1,
            save_act=True,
            auto_save=auto_save,
            first_init=False,
            device=target_device
        )

        # ==========================================
        # 2. 参数迁移与方差通道切片
        # ==========================================
        for i, (bayes_layer, det_layer) in enumerate(zip(self.layers, model.act_fun)):
            det_layer.grid.data.copy_(bayes_layer.grid.data.to(target_device))
            
            # 关键：如果是最后一层，且原来是双通道(异方差)输出
            if i == len(self.layers) - 1 and self.width[-1] == 2:
                # 只拷贝第 0 个通道 (均值通道 [:, 0:1])，彻底抛弃方差通道
                det_layer.coef.data.copy_(bayes_layer.coef_mu.data[:, 0:1, :].to(target_device))
                det_layer.scale_base.data.copy_(bayes_layer.scale_base_mu.data[:, 0:1].to(target_device))
                det_layer.scale_sp.data.copy_(bayes_layer.scale_sp_mu.data[:, 0:1].to(target_device))
                det_layer.mask.data.copy_(bayes_layer.mask.data[:, 0:1].to(target_device))
            else:
                # 隐藏层正常拷贝
                det_layer.coef.data.copy_(bayes_layer.coef_mu.data.to(target_device))
                det_layer.scale_base.data.copy_(bayes_layer.scale_base_mu.data.to(target_device))
                det_layer.scale_sp.data.copy_(bayes_layer.scale_sp_mu.data.to(target_device))
                det_layer.mask.data.copy_(bayes_layer.mask.data.to(target_device))

        return model.to(target_device).eval()
    
    def symbolic_enabled(self, enable=True):
        """
        占位符：兼容标准 KAN 接口
        （贝叶斯版本目前不支持符号化）
        """
        raise NotImplementedError(
            "BayesMultKAN does not yet implement an in-model symbolic branch. "
            "Use to_deterministic_kan() followed by MultKAN symbolic extraction."
        )
    
    def prune_node(self, node_id, threshold=0.01):
        """
        占位符：节点剪枝
        
        Args:
        -----
            node_id : tuple
                (层索引, 节点索引)
                
            threshold : float
                剪枝阈值
        """
        raise NotImplementedError(
            "BayesMultKAN node pruning is not implemented yet. "
            "Export with to_deterministic_kan() and prune the resulting MultKAN."
        )
    
    def prune_edge(self, edge_id, threshold=0.01):
        """
        占位符：边剪枝
        
        Args:
        -----
            edge_id : tuple
                (层索引, 输入索引, 输出索引)
                
            threshold : float
                剪枝阈值
        """
        raise NotImplementedError(
            "BayesMultKAN edge pruning is not implemented yet. "
            "Export with to_deterministic_kan() and prune the resulting MultKAN."
        )
    
    def plot_uncertainty(self, x_test, y_test=None, figsize=(10, 6), save_path=None):
        """
        绘制预测不确定性
        
        Args:
        -----
            x_test : Tensor
                测试输入
                
            y_test : Tensor
                测试目标（可选）
                
            figsize : tuple
                图像尺寸
                
            save_path : str
                保存路径（可选）
        """
        import matplotlib.pyplot as plt
        
        with torch.no_grad():
            samples, mean, variance = self.forward_sample(x_test)
        
        std = torch.sqrt(variance)
        
        # 只为 1D 输出绘制
        if mean.shape[1] == 1:
            x_np = x_test.cpu().numpy()
            mean_np = mean.cpu().numpy().squeeze()
            std_np = std.cpu().numpy().squeeze()
            
            plt.figure(figsize=figsize)
            
            # 排序以绘制连续曲线
            sort_idx = np.argsort(x_np[:, 0])
            x_sorted = x_np[sort_idx, 0]
            mean_sorted = mean_np[sort_idx]
            std_sorted = std_np[sort_idx]
            
            plt.plot(x_sorted, mean_sorted, 'b-', label='Mean prediction', linewidth=2)
            plt.fill_between(x_sorted, 
                            mean_sorted - 2*std_sorted, 
                            mean_sorted + 2*std_sorted,
                            alpha=0.3, label='±2σ uncertainty')
            
            if y_test is not None:
                y_np = y_test.cpu().numpy().squeeze()[sort_idx]
                plt.plot(x_sorted, y_np, 'ro', label='Ground truth', markersize=4, alpha=0.5)
            
            plt.xlabel('Input')
            plt.ylabel('Output')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
            plt.show()
