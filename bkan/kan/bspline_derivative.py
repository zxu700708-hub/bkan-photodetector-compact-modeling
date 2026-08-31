"""
B-spline KAN 解析导数模块 (B-spline Analytical Derivative for KAN)

核心思路:
  KAN 的 B-spline 基函数通过 Cox-de Boor 递归计算，这是纯 PyTorch tensor 运算。
  autograd 自动给出精确的 dφ/dx，不需要手动实现 B-spline 导数递归。

  对于单输出模型, 直接对完整模型做 autograd: d(y_norm)/d(x_norm[:, col])
  对于多输出模型, 逐层计算 Jacobian 然后链式累积。

对比符号公式导数:
  - 符号公式: 用 exp/1/x 等全局函数拟合 B-spline → 导数结构完全不同
  - B-spline 解析导数: 直接对拟合的 B-spline 曲面求导 → 导数与模型一致

贝叶斯导数不确定性:
  对后验采样 N 组权重 → 每组计算解析导数 → 得到导数的均值±2σ
"""

import torch
import numpy as np


# ==============================================================================
# 模型 forward 包装
# ==============================================================================

def _model_forward(model, x, sample_mode=False):
    """对 BayesMultKAN / MultKAN 的统一 forward 调用, 只返回输出 tensor."""
    if sample_mode and hasattr(model, 'layers'):
        y, _ = model.forward(x, sample=True)
    elif hasattr(model, 'layers'):
        y, _ = model.forward(x, sample=False)
    elif hasattr(model, 'act_fun'):
        y = model.forward(x)
    else:
        y = model.forward(x)
    return y


# ==============================================================================
# 单输出模型: 直接 autograd (简单、高效、无链式累积问题)
# ==============================================================================

def kan_model_derivative_direct(model, x, diff_col_idx, sample_mode=False):
    """单输出模型: 直接对 model(x) 做 autograd 得到 d(y_norm)/d(x_norm[:, col]).

    参数:
        model: BayesMultKAN 或 MultKAN (out_dim=1)
        x: (batch, in_dim) torch tensor — 已在标准化空间
        diff_col_idx: int — 求导的输入列索引
        sample_mode: 是否用贝叶斯采样

    返回:
        deriv: (batch,) numpy array — d(normalized_output)/d(normalized_input[:, diff_col_idx])
    """
    x_tensor = x.detach().clone().requires_grad_(True)
    y = _model_forward(model, x_tensor, sample_mode)  # (batch, out_dim)

    # y 的每一行独立 → sum(y) 对 x[b] 的梯度就是 d(y[b])/d(x[b])
    grad = torch.autograd.grad(y.sum(), x_tensor, create_graph=False)[0]  # (batch, in_dim)
    return grad[:, diff_col_idx].detach().cpu().numpy()


# ==============================================================================
# 多层多输出模型: 逐层 Jacobian + 链式法则
# ==============================================================================

def _layer_forward(layer, x, sample_mode=False):
    """Wrap layer forward — 显式控制 sample 参数."""
    if sample_mode and hasattr(layer, 'sample_parameters'):
        y, _, _, _ = layer.forward(x, sample=True)
    else:
        y, _, _, _ = layer.forward(x, sample=False)
    return y


def kan_layer_jacobian(layer, x, sample_mode=False):
    """计算 KAN/BayesKAN 单层对输入的 Jacobian.

    参数:
        layer: KANLayer 或 BayesKANLayer
        x: (batch, in_dim) torch tensor — 必须在计算图中 (requires_grad 或非叶节点)
        sample_mode: 是否用贝叶斯采样

    返回:
        J: (batch, out_dim, in_dim) — per-sample Jacobian
    """
    batch, in_dim = x.shape
    out_dim = layer.out_dim

    # 关键: 如果 x 已在计算图中, 直接使用它 (不要 detach+clone)
    # 如果 x 是 detached 的, 则需要 clone+requires_grad 来建立新图
    if x.grad_fn is None and not x.requires_grad:
        x_tensor = x.detach().clone().requires_grad_(True)
    else:
        x_tensor = x

    y = _layer_forward(layer, x_tensor, sample_mode)  # (batch, out_dim)

    # 验证计算图
    if y.grad_fn is None:
        raise RuntimeError(
            "Layer forward produced output without grad_fn. "
            "The B-spline computation graph was not tracked. "
            "Check that the input requires grad and layer parameters are connected."
        )

    J = torch.zeros(batch, out_dim, in_dim, dtype=x.dtype, device=x.device)
    for j in range(out_dim):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[:, j] = 1.0
        grads = torch.autograd.grad(
            y, x_tensor, grad_outputs=grad_outputs,
            create_graph=False, retain_graph=(j < out_dim - 1),
            allow_unused=False,
        )[0]
        J[:, j, :] = grads.detach()

    return J


def kan_model_jacobian_multi_output(model, x, sample_mode=False):
    """多输出模型: 逐层 Jacobian + 链式法则.

    链式法则: J_total = J_L @ J_{L-1} @ ... @ J_1

    参数:
        model: BayesMultKAN 或 MultKAN
        x: (batch, in_dim) torch tensor — 已在标准化空间
        sample_mode: 是否用贝叶斯采样

    返回:
        J: (batch, out_dim, in_dim) — per-sample total Jacobian
    """
    layers = model.layers if hasattr(model, 'layers') else model.act_fun
    batch = x.shape[0]
    out_dim = model.width[-1]
    in_dim = model.width[0]

    # 带梯度地前向传播, 保存每层输入 (不断开计算图)
    current_x = x.detach().clone().requires_grad_(True)
    layer_inputs = [current_x]

    for layer in layers:
        y = _layer_forward(layer, current_x, sample_mode)
        # 不 detach — 保持完整计算图以便后续层求 Jacobian
        layer_inputs.append(y)
        current_x = y

    # 链式累积: J_total = J_L @ J_{L-1} @ ... @ J_1
    # 从 I 开始: d(layer0_input)/d(model_input) = I
    J_accum = torch.eye(in_dim, device=x.device).unsqueeze(0).expand(batch, in_dim, in_dim)

    for l_idx, layer in enumerate(layers):
        # layer_in 在计算图中 (除了第一层是 fresh leaf)
        layer_in = layer_inputs[l_idx]
        J_layer = kan_layer_jacobian(layer, layer_in, sample_mode)  # (batch, out, in)
        J_accum = torch.bmm(J_layer, J_accum)  # (batch, layer_out, in_0)
        # 断开当前累积的梯度 — 下一层需要新的 leaf
        J_accum = J_accum.detach()

    return J_accum


# ==============================================================================
# 统一接口
# ==============================================================================

def kan_model_derivative(model, x, diff_col_idx, sample_mode=False):
    """计算 KAN 模型输出对指定输入列的导数.

    自动选择方法:
      - 单输出 (out_dim=1): 直接 autograd (更快更稳)
      - 多输出: 逐层 Jacobian + 链式法则

    参数:
        model: BayesMultKAN 或 MultKAN
        x: (batch, in_dim) torch tensor — 已在标准化空间
        diff_col_idx: int — 求导的输入列索引
        sample_mode: 是否用贝叶斯采样

    返回:
        deriv: (batch,) numpy array — d(normalized_output)/d(normalized_input[:, diff_col_idx])
    """
    out_dim = model.width[-1]
    if out_dim == 1:
        return kan_model_derivative_direct(model, x, diff_col_idx, sample_mode)
    else:
        J = kan_model_jacobian_multi_output(model, x, sample_mode)
        return J[:, 0, diff_col_idx].detach().cpu().numpy()


# ==============================================================================
# 贝叶斯 MC 导数 (不确定性量化)
# ==============================================================================

def kan_model_derivative_bayesian(model, x, diff_col_idx, n_samples=100):
    """贝叶斯 MC 采样计算导数分布.

    对后验采样 n_samples 组权重，每组计算解析导数，统计均值±2σ。

    参数:
        model: BayesMultKAN
        x: (batch, in_dim) torch tensor — 已在标准化空间
        diff_col_idx: int
        n_samples: MC 采样次数 (默认 100)

    返回:
        dict: {
            'mean': (batch,) — 导数均值
            'std': (batch,) — 导数标准差
            'lower_2sigma': (batch,) — 均值 - 2*std
            'upper_2sigma': (batch,) — 均值 + 2*std
            'samples': (n_samples, batch) — 全部采样
        }
    """
    batch = x.shape[0]
    all_samples = np.zeros((n_samples, batch), dtype=np.float64)

    for s in range(n_samples):
        all_samples[s] = kan_model_derivative(model, x, diff_col_idx, sample_mode=True)

    mean = all_samples.mean(axis=0)
    std = all_samples.std(axis=0, ddof=1)

    return {
        'mean': mean,
        'std': std,
        'lower_2sigma': mean - 2 * std,
        'upper_2sigma': mean + 2 * std,
        'samples': all_samples,
    }
