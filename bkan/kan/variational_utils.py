import torch
import torch.nn as nn
import math
import numpy as np


PI = math.pi


# ============================================================================
# KL 散度计算
# ============================================================================

def kl_divergence_gaussian(mu_q, log_sigma_q, mu_p=0.0, log_sigma_p=0.0):
    """
    两个高斯分布之间的 KL 散度
    
    KL(q||p) = 0.5 * Σ[log(σ_p²/σ_q²) + (σ_q² + (μ_q - μ_p)²) / σ_p² - 1]
    
    Args:
    -----
        mu_q : Tensor
            后验（q）的均值
            
        log_sigma_q : Tensor
            后验（q）的对数标准差
            
        mu_p : float or Tensor
            先验（p）的均值。默认: 0.0
            
        log_sigma_p : float or Tensor
            先验（p）的对数标准差。默认: 0.0
    
    Returns:
    --------
        kl : Tensor
            逐元素 KL 散度
            
    Example:
    --------
    >>> mu_q = torch.randn(10, 20)
    >>> log_sigma_q = torch.randn(10, 20)
    >>> kl = kl_divergence_gaussian(mu_q, log_sigma_q)
    >>> print(kl.shape, kl.sum())
    """
    sigma_q = torch.exp(log_sigma_q)
    sigma_p = torch.exp(log_sigma_p)
    
    kl = log_sigma_p - log_sigma_q + \
         (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2 * sigma_p ** 2) - 0.5
    
    return kl


def kl_divergence_uniform(log_sigma_q, bound=1.0):
    """
    高斯分布与均匀分布的 KL 散度
    （用于 bound 内的均匀先验）
    
    Args:
    -----
        log_sigma_q : Tensor
            后验的对数标准差
            
        bound : float
            均匀分布的界限 [-bound, bound]
    
    Returns:
    --------
        kl : Tensor
            KL 散度
    """
    sigma_q = torch.exp(log_sigma_q)
    
    # log(p_uniform) = -log(2*bound)
    # KL 的计算相对复杂，这里用近似
    log_p_uniform = -np.log(2 * bound)
    
    # 简单近似：如果 σ << bound，KL ≈ const
    kl = -log_p_uniform + 0.5 * torch.log(sigma_q ** 2 / (bound ** 2 / 3))
    
    return kl


def kl_divergence_categorical(p_q, p_p):
    """
    两个分类分布间的 KL 散度
    
    KL(q||p) = Σ q_i * log(q_i / p_i)
    
    Args:
    -----
        p_q : Tensor
            后验的概率向量
            
        p_p : Tensor
            先验的概率向量
    
    Returns:
    --------
        kl : Tensor
            KL 散度
    """
    eps = 1e-10
    p_q = torch.clamp(p_q, min=eps)
    p_p = torch.clamp(p_p, min=eps)
    
    kl = (p_q * (torch.log(p_q) - torch.log(p_p))).sum()
    
    return kl


# ============================================================================
# 重新参数化技巧
# ============================================================================

def reparameterize_gaussian(mu, log_sigma, num_samples=1):
    """
    使用重新参数化技巧从高斯分布采样
    
    z = μ + σ * ε，其中 ε ~ N(0, I)
    
    Args:
    -----
        mu : Tensor
            均值，shape (...)
            
        log_sigma : Tensor
            对数标准差，shape (...)
            
        num_samples : int
            采样数。默认: 1
    
    Returns:
    --------
        samples : Tensor
            采样结果，shape (num_samples, ...)
    """
    sigma = torch.exp(log_sigma)
    eps = torch.randn((num_samples,) + mu.shape, device=mu.device)
    samples = mu[None, ...] + sigma[None, ...] * eps
    
    return samples


def reparameterize_lognormal(mu, log_sigma, num_samples=1):
    """
    从对数正态分布采样
    相当于先从高斯采样，再取指数
    
    Args:
    -----
        mu : Tensor
            对数空间的均值
            
        log_sigma : Tensor
            对数空间的对数标准差
            
        num_samples : int
            采样数
    
    Returns:
    --------
        samples : Tensor
            采样结果（正值）
    """
    # 先从高斯采样
    gaussian_samples = reparameterize_gaussian(mu, log_sigma, num_samples)
    
    # 取指数得到对数正态
    samples = torch.exp(gaussian_samples)
    
    return samples


def reparameterize_gamma(alpha, beta, num_samples=1):
    """
    从 Gamma 分布采样（使用 Gumbel-Max 技巧的近似）
    
    Args:
    -----
        alpha : Tensor
            形状参数，shape (...)
            
        beta : Tensor
            速率参数，shape (...)
            
        num_samples : int
            采样数
    
    Returns:
    --------
        samples : Tensor
            采样结果，shape (num_samples, ...)
    """
    # 使用 Marsaglia and Tsang 方法的简化版本
    # 这是一个近似方法
    eps = torch.randn((num_samples,) + alpha.shape, device=alpha.device)
    
    # 简单近似：使用正态分布近似
    mean = alpha / beta
    var = alpha / (beta ** 2)
    
    samples = mean[None, ...] + torch.sqrt(var[None, ...]) * eps
    samples = torch.clamp(samples, min=1e-6)  # 确保正值
    
    return samples


# ============================================================================
# 不确定性度量
# ============================================================================

def predictive_entropy(samples):
    """
    计算预测熵 H[y|x]
    
    这反映了总不确定性（epistemic + aleatoric）
    对于高斯分布：H = 0.5 * log(2πeσ²)
    
    Args:
    -----
        samples : Tensor
            蒙特卡洛样本，shape (num_samples, batch_size, output_dim)
    
    Returns:
    --------
        entropy : Tensor
            熵，shape (batch_size, output_dim)
    """
    # 计算样本方差（总方差）
    mean = samples.mean(dim=0)
    var = samples.var(dim=0, unbiased=False)
    
    # 高斯分布的熵
    entropy = 0.5 * torch.log(2 * PI * torch.e * var + 1e-8)
    
    return entropy


def mutual_information(samples):
    """
    计算互信息 I[y;w|x] - 认知不确定性的度量
    
    I[y;w|x] = H[y|x] - E_w[H[y|x,w]]
    = Var_w[E[y|x,w]] (对于高斯分布)
    
    Args:
    -----
        samples : Tensor
            蒙特卡洛样本，shape (num_samples, batch_size, output_dim)
    
    Returns:
    --------
        mi : Tensor
            互信息，shape (batch_size, output_dim)
    """
    # 认知不确定性 = 参数不同时预测的方差
    mi = samples.var(dim=0, unbiased=False)
    
    return mi


def epistemic_uncertainty(samples):
    """
    计算认知不确定性（参数不确定性）
    
    Args:
    -----
        samples : Tensor
            蒙特卡洛样本
    
    Returns:
    --------
        epi_std : Tensor
            认知不确定性标准差
    """
    epi_var = samples.var(dim=0, unbiased=False)
    epi_std = torch.sqrt(epi_var + 1e-8)
    
    return epi_std


def aleatoric_uncertainty(samples, model_variance=None):
    """
    计算偶然不确定性（数据/模型不确定性）
    
    通常从模型的输出或平均样本方差估计
    
    Args:
    -----
        samples : Tensor
            蒙特卡洛样本
            
        model_variance : Tensor
            如果模型直接输出方差，在此提供
    
    Returns:
    --------
        ale_std : Tensor
            偶然不确定性标准差
    """
    if model_variance is not None:
        # 直接使用模型输出的方差
        ale_std = torch.sqrt(model_variance + 1e-8)
    else:
        # 使用样本的平均方差（假设每个样本都是独立的高斯）
        # 这种情况下偶然不确定性会被高估
        ale_std = torch.zeros_like(samples[0])
    
    return ale_std


# ============================================================================
# 校准和置信度
# ============================================================================

def confidence_score(mean, std, metric='sigmoid', scale=1.0):
    """
    基于不确定性的置信度评分
    
    Args:
    -----
        mean : Tensor
            预测均值
            
        std : Tensor
            预测标准差
            
        metric : str
            度量方式：
            - 'sigmoid': 1 / (1 + exp(scale * std))
            - 'inverse_std': exp(-scale * std)
            - 'coefficient': 1 / (1 + std / (|mean| + eps))
            
        scale : float
            缩放因子
    
    Returns:
    --------
        confidence : Tensor
            置信度，范围 [0, 1]
    """
    if metric == 'sigmoid':
        confidence = torch.sigmoid(-scale * std)
        
    elif metric == 'inverse_std':
        confidence = torch.exp(-scale * std)
        
    elif metric == 'coefficient':
        confidence = 1.0 / (1.0 + scale * std / (torch.abs(mean) + 1e-8))
        
    else:
        raise ValueError(f"Unknown confidence metric: {metric}")
    
    return torch.clamp(confidence, 0.0, 1.0)


def uncertainty_calibration_error(predictions, uncertainty, targets, bins=10):
    """
    计算校准误差（Expected Calibration Error）
    用于评估不确定性估计的质量
    
    Args:
    -----
        predictions : Tensor
            预测值，shape (N,)
            
        uncertainty : Tensor
            预测不确定性（标准差），shape (N,)
            
        targets : Tensor
            真实值，shape (N,)
            
        bins : int
            分箱数
    
    Returns:
    --------
        ece : float
            期望校准误差
    """
    # 计算误差
    error = torch.abs(predictions - targets)
    
    # 分箱
    uncertainty_sorted, sorted_indices = torch.sort(uncertainty)
    error_sorted = error[sorted_indices]
    
    bin_size = len(uncertainty) // bins
    ece = 0.0
    
    for i in range(bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < bins - 1 else len(uncertainty)
        
        if end_idx > start_idx:
            bin_uncertainty = uncertainty_sorted[start_idx:end_idx].mean()
            bin_error = error_sorted[start_idx:end_idx].mean()
            
            # 对于高斯分布，期望误差应该约为 σ
            ece += torch.abs(bin_error - bin_uncertainty)
    
    ece /= bins
    
    return ece.item()


# ============================================================================
# OOD (Out-of-Distribution) 检测
# ============================================================================

def compute_ood_score(samples, metric='entropy'):
    """
    计算 OOD 分数
    
    Args:
    -----
        samples : Tensor
            蒙特卡洛样本，shape (num_samples, batch_size, output_dim)
            
        metric : str
            度量方式：
            - 'entropy': 预测熵
            - 'mutual_information': 互信息
            - 'variation_ratio': 1 - 最大概率（用于分类）
    
    Returns:
    --------
        ood_score : Tensor
            OOD 分数，值越大越可能是 OOD
    """
    if metric == 'entropy':
        ood_score = predictive_entropy(samples).mean(dim=-1)
        
    elif metric == 'mutual_information':
        ood_score = mutual_information(samples).mean(dim=-1)
        
    elif metric == 'variation_ratio':
        # 适用于分类任务
        # 计算多数类出现的频率
        predictions = torch.argmax(samples, dim=-1)
        mode, _ = torch.mode(predictions, dim=0)
        variation_ratio = 1.0 - (predictions == mode[None, :]).float().mean(dim=0)
        ood_score = variation_ratio
        
    else:
        raise ValueError(f"Unknown OOD metric: {metric}")
    
    return ood_score


# ============================================================================
# 主动学习采样
# ============================================================================

def bald_score(samples):
    """
    计算 BALD (Bayesian Active Learning by Disagreement) 得分
    用于主动学习中选择最有信息量的样本
    
    BALD = I[y;w|x] = H[y|x] - E_w[H[y|x,w]]
    
    Args:
    -----
        samples : Tensor
            蒙特卡洛样本
    
    Returns:
    --------
        bald : Tensor
            BALD 得分
    """
    total_entropy = predictive_entropy(samples)
    mi = mutual_information(samples)
    
    bald = total_entropy - mi
    
    return bald


def variation_ratio(predictions_samples):
    """
    变差比 (Variation Ratio)
    选择模型预测最不确定的样本
    
    Args:
    -----
        predictions_samples : Tensor
            预测样本，shape (num_samples, batch_size, num_classes)
    
    Returns:
    --------
        vr : Tensor
            变差比
    """
    # 选择每个样本的最可能类别
    max_preds = torch.argmax(predictions_samples, dim=-1)
    
    # 计算最频繁类别出现的频率
    batch_size = predictions_samples.shape[1]
    vr = []
    
    for i in range(batch_size):
        unique, counts = torch.unique(max_preds[:, i], return_counts=True)
        max_count = counts.max().float()
        vr.append(1.0 - max_count / predictions_samples.shape[0])
    
    return torch.tensor(vr)


def entropy_sampling(samples):
    """
    熵采样 (Entropy Sampling)
    选择预测熵最大的样本
    
    Args:
    -----
        samples : Tensor
            蒙特卡洛样本
    
    Returns:
    --------
        entropy : Tensor
            预测熵
    """
    entropy = predictive_entropy(samples)
    return entropy.mean(dim=-1)
