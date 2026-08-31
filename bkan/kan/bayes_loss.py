import torch
import torch.nn as nn
import torch.nn.functional as F
import math


PI_ = math.pi


def variance_from_logits(variance_logits, eps=1e-6):
    """Convert an unconstrained variance head to a positive variance.

    Training and inference must use this same smooth transformation.  Unlike a
    hard log-variance clamp, softplus keeps useful gradients in the tails.
    """
    return F.softplus(variance_logits) + eps


def gaussian_likelihood(preds, variance_logits, y_true):
    """
    高斯似然函数
    
    log p(y|x,σ²) = -0.5 * log(2πσ²) - 0.5 * (y - ŷ)² / σ²
                 = -0.5 * log_devs2 - 0.5 * (y - ŷ)² * exp(-log_devs2)
    
    Args:
    -----
        preds : Tensor
            预测值，shape (batch_size, ...)
            
        log_devs2 : Tensor
            对数方差，log(σ²)，shape (batch_size, ...) 或 scalar
            
        y_true : Tensor
            真实值，shape (batch_size, ...)
    
    Returns:
    --------
        nll : Tensor
            负对数似然（标量）
            
        mse : Tensor
            均方误差（用于诊断）
    """
    squared_error = (y_true - preds) ** 2
    variance = variance_from_logits(variance_logits)
    nll = 0.5 * (torch.log(variance) + squared_error / variance)
    
    return nll.mean(), squared_error.mean()


def student_t_likelihood(preds, log_devs2, nu, y_true, eps=1e-8):
    """
    学生 t 分布似然函数
    相比高斯分布，对异常值更鲁棒
    
    t 分布的 pdf:
    p(y|μ,σ²,ν) = Γ((ν+1)/2) / (Γ(ν/2) * sqrt(νπσ²)) * [1 + (y-μ)²/(νσ²)]^(-(ν+1)/2)
    
    Args:
    -----
        preds : Tensor
            预测值（均值）
            
        log_devs2 : Tensor
            对数方差，log(σ²)
            
        nu : Tensor or float
            自由度参数 ν（通常参数化为 log(ν)）
            
        y_true : Tensor
            真实值
            
        eps : float
            数值稳定性参数
    
    Returns:
    --------
        nll : Tensor
            负对数似然（标量）
            
        mse : Tensor
            均方误差
    """
    nu = torch.exp(nu) + eps
    squared_error = (y_true - preds) ** 2
    variance = variance_from_logits(log_devs2, eps=eps)
    
    # log gamma functions
    numerator = torch.lgamma((nu + 1.0) / 2.0)
    denominator = torch.lgamma(nu / 2.0) + 0.5 * torch.log(nu * PI_ * variance)
    
    # log p(y|x)
    log_pdf = numerator - denominator - \
              (nu + 1.0) * 0.5 * torch.log(1.0 + squared_error / (nu * variance))
    
    # NLL = -log p(y|x)
    nll = -log_pdf
    
    return nll.mean(), squared_error.mean()


def laplace_likelihood(preds, log_devs2, y_true):
    """
    拉普拉斯（指数）分布似然函数
    
    log p(y|μ,b) = -log(2b) - |y - μ| / b
    其中 b = sqrt(0.5 * exp(log_devs2))
    
    Args:
    -----
        preds : Tensor
            预测值
            
        log_devs2 : Tensor
            对数方差（2*log(b)）
            
        y_true : Tensor
            真实值
    
    Returns:
    --------
        nll : Tensor
            负对数似然（标量）
            
        mae : Tensor
            平均绝对误差
    """
    abs_error = torch.abs(y_true - preds)
    b = torch.sqrt(0.5 * variance_from_logits(log_devs2))
    
    nll = -torch.log(2 * b) + abs_error / b
    
    return nll.mean(), abs_error.mean()


def nig_likelihood(preds, s, alpha, beta, y_true, eps=1e-8):
    """
    Normal-Inverse-Gamma (NIG) 分布似然函数
    用于联合建模均值和方差的不确定性
    
    Args:
    -----
        preds : Tensor
            均值预测
            
        s : Tensor
            方差的缩放参数
            
        alpha : Tensor
            Gamma 分布的形状参数
            
        beta : Tensor
            Gamma 分布的速率参数
            
        y_true : Tensor
            真实值
            
        eps : float
            数值稳定性参数
    
    Returns:
    --------
        nll : Tensor
            负对数似然（标量）
            
        mse : Tensor
            均方误差
    """
    squared_error = (y_true - preds) ** 2
    
    # NIG log probability
    nll = 0.5 * torch.log(torch.tensor(PI_)) - 0.5 * torch.log(s) - \
          alpha * torch.log(beta) + torch.lgamma(alpha + 0.5) - torch.lgamma(alpha) - \
          (alpha + 0.5) * torch.log(beta + 0.5 * s * squared_error + eps)
    
    return -nll.mean(), squared_error.mean()


class BayesianLoss(nn.Module):
    """
    贝叶斯损失模块 - 结合 ELBO 和多种似然函数
    """
    
    def __init__(self, likelihood='gaussian', kl_weight=0.01):
        """
        初始化贝叶斯损失
        
        Args:
        -----
            likelihood : str
                似然函数类型：'gaussian', 'student_t', 'laplace', 'nig'
                
            kl_weight : float
                KL 散度的权重系数
        """
        super(BayesianLoss, self).__init__()
        self.likelihood_type = likelihood
        self.kl_weight = kl_weight
        
        if likelihood == 'gaussian':
            self.likelihood_fn = gaussian_likelihood
        elif likelihood == 'student_t':
            self.likelihood_fn = student_t_likelihood
        elif likelihood == 'laplace':
            self.likelihood_fn = laplace_likelihood
        elif likelihood == 'nig':
            self.likelihood_fn = nig_likelihood
        else:
            raise ValueError(f"Unknown likelihood type: {likelihood}")
    
    def forward(self, preds, y_true, kl_divergence, num_train_samples, log_devs2=None, nu=None):
        """
        计算贝叶斯损失
        
        ELBO = -log p(y|x) + KL / batch_size
        
        Args:
        -----
            preds : Tensor
                预测值
                
            y_true : Tensor
                真实值
                
            kl_divergence : float or Tensor
                KL 散度（已计算）
                
            batch_size : int
                批次大小
                
            log_devs2 : Tensor
                对数方差（高斯/拉普拉斯等需要）
                
            nu : Tensor or float
                学生 t 分布的自由度参数
        
        Returns:
        --------
            elbo_loss : Tensor
                ELBO 损失
                
            nll : Tensor
                似然项
                
            kl_term : Tensor
                KL 项
        """
        # ✅ 核心修改：动态拆分双通道输出
        if preds.shape[-1] == 2 and y_true.shape[-1] == 1:
            mu_pred = preds[..., 0:1]         # 第 0 个通道：预测均值
            log_devs2 = preds[..., 1:2]       # 第 1 个通道：预测对数方差
        else:
            mu_pred = preds
            if log_devs2 is None:
                unit_variance_logit = math.log(math.expm1(1.0))
                log_devs2 = torch.full_like(mu_pred, unit_variance_logit)
        # 计算似然
        if self.likelihood_type == 'gaussian':
            nll, _ = self.likelihood_fn(mu_pred, log_devs2, y_true)
            
        elif self.likelihood_type == 'student_t':
            if nu is None:
                nu = torch.tensor(2.0)
            nll, _ = self.likelihood_fn(mu_pred, log_devs2, nu, y_true)
            
        elif self.likelihood_type == 'laplace':
            nll, _ = self.likelihood_fn(mu_pred, log_devs2, y_true)
            
        elif self.likelihood_type == 'nig':
            raise NotImplementedError("NIG loss requires additional parameters")
        
        # 转换 KL 散度为张量
        if isinstance(kl_divergence, (int, float)):
            kl_term = torch.tensor(kl_divergence, dtype=preds.dtype, device=preds.device)
        else:
            kl_term = kl_divergence
        
        # 计算 ELBO
        # Per-observation final-phase variational objective:
        # E_q[-log p(y|x,theta)] + beta * KL(q||p) / N.  This equals the
        # standard negative ELBO only for beta=1; beta!=1 defines a
        # generalized/tempered variational target.
        kl_weighted = self.kl_weight * kl_term / num_train_samples
        elbo_loss = nll + kl_weighted

        return elbo_loss, nll, kl_weighted


def predictive_entropy(samples):
    """
    计算预测熵 - 总不确定性的一个度量
    
    H = -E[log p(y|x)]
    其中期望是对预测分布取的
    
    Args:
    -----
        samples : Tensor
            蒙特卡洛样本，shape (num_samples, batch_size, output_dim)
    
    Returns:
    --------
        entropy : Tensor
            逐样本熵
    """
    # 计算平均预测
    mean_pred = samples.mean(dim=0)
    
    # 计算方差
    var_pred = samples.var(dim=0, unbiased=False)
    
    # 对于高斯分布，熵 = 0.5 * log(2πe σ²)
    entropy = 0.5 * torch.log(2 * math.pi * math.e * var_pred + 1e-8)
    
    return entropy


def mutual_information(samples):
    """
    计算互信息 - 认知不确定性的一个度量
    
    I = E_w[log p(y|x,w)] - log p(y|x,w_mean)
    = H[y] - E_w[H[y|x,w]]
    
    Args:
    -----
        samples : Tensor
            蒙特卡洛样本，shape (num_samples, batch_size, output_dim)
    
    Returns:
    --------
        mi : Tensor
            逐样本互信息
    """
    # 均值和方差
    mean_pred = samples.mean(dim=0)
    
    # 样本间方差（认知不确定性）
    epi_var = samples.var(dim=0, unbiased=False)
    
    # 互信息 ≈ 0.5 * log(1 + 12 * epi_var)
    # 或更准确地：对于高斯，MI ≈ 0.5 * log(总方差 / 偶然方差)
    mi = 0.5 * torch.log(epi_var + 1e-8)
    
    return mi


def confidence_score(mean, std, metric='entropy'):
    """
    基于不确定性的置信度评分
    
    Args:
    -----
        mean : Tensor
            预测均值
            
        std : Tensor
            预测标准差
            
        metric : str
            度量方式：'entropy', 'inverse_std', 'softmax'
    
    Returns:
    --------
        confidence : Tensor
            置信度得分 (0 到 1 之间)
    """
    if metric == 'entropy':
        # 基于熵：熵越小置信度越高
        entropy = 0.5 * torch.log(std ** 2 + 1e-8)
        confidence = 1.0 / (1.0 + entropy)
        
    elif metric == 'inverse_std':
        # 直接基于标准差的倒数
        confidence = torch.exp(-std)
        
    elif metric == 'softmax':
        # 基于不确定性的 softmax
        confidence = 1.0 / (1.0 + std / (mean.abs() + 1e-8))
        
    else:
        raise ValueError(f"Unknown confidence metric: {metric}")
    
    # 限制在 [0, 1]
    confidence = torch.clamp(confidence, 0.0, 1.0)
    
    return confidence
