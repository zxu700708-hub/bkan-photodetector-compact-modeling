import copy
import math
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, LBFGS
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from .bayes_loss import BayesianLoss, variance_from_logits
from .variational_utils import (
    predictive_entropy, mutual_information,
    epistemic_uncertainty, confidence_score,
    compute_ood_score
)


class BayesianExperiment:
    """
    贝叶斯 KAN 模型的训练和评估框架
    
    Attributes:
    -----------
        model : BayesMultKAN
            贝叶斯 KAN 模型
            
        train_loader : DataLoader
            训练数据加载器
            
        val_loader : DataLoader
            验证数据加载器
            
        test_loader : DataLoader
            测试数据加载器
            
        optimizer : torch.optim.Optimizer
            优化器
            
        device : str
            计算设备
    """
    
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader=None,
                 test_loader=None,
                 optimizer_name='adam',
                 lr=1e-3,
                 weight_decay=1e-5,
                 device='cpu',
                 likelihood='gaussian',
                 kl_weight=0.01):
        """
        初始化贝叶斯实验
        
        Args:
        -----
            model : BayesMultKAN
                贝叶斯 KAN 模型
                
            train_loader : DataLoader
                训练数据加载器
                
            val_loader : DataLoader
                验证数据加载器（可选）
                
            test_loader : DataLoader
                测试数据加载器（可选）
                
            optimizer_name : str
                优化器：'adam', 'sgd', 'lbfgs'。默认: 'adam'
                
            lr : float
                学习率。默认: 1e-3
                
            weight_decay : float
                权重衰减。默认: 1e-5
                
            device : str
                计算设备。默认: 'cpu'
                
            likelihood : str
                似然函数类型。默认: 'gaussian'
                
            kl_weight : float
                KL 散度权重。默认: 0.01
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # 优化器
        if optimizer_name == 'adam':
            self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            self.optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_name == 'lbfgs':
            self.optimizer = LBFGS(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        # 损失函数
        self.criterion = BayesianLoss(likelihood=likelihood, kl_weight=kl_weight)
        self.mse_loss = nn.MSELoss()
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_nll': [],
            'train_kl': [],
            'val_loss': [],
            'val_mse': [],
            'best_val_loss': float('inf')
        }
    
    def train_epoch(self, epoch, num_mc_samples=1):
        """
        训练一个 epoch
        """
        self.model.train()
        total_loss = 0.0
        total_nll = 0.0
        total_kl = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        # ✅ 【关键修复点】：在这里（for循环外部）获取当前训练集的总样本数！
        num_train_samples = len(self.train_loader.dataset)
        
        for batch_idx, (x_batch, y_batch) in enumerate(pbar):
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            sampled_predictions = [
                self.model(x_batch, sample=True)[0]
                for _ in range(max(1, int(num_mc_samples)))
            ]
            
            # 计算损失
            kl_div = self.model.kl_divergence()
            
            # KL annealing: linear warmup from 0 to 1 over anneal_epochs.
            # After anneal_epochs the full KL penalty is active (no cap).
            anneal_epochs = 100
            kl_scale = min(1.0, epoch / anneal_epochs)

            sampled_nll = []
            sampled_mse = []
            kl = None
            for y_pred in sampled_predictions:
                _, sample_nll, kl = self.criterion(
                    y_pred, y_batch, kl_div, num_train_samples
                )
                mu_pred = y_pred[..., 0:1] if y_pred.shape[-1] == 2 else y_pred
                sampled_nll.append(sample_nll)
                sampled_mse.append(self.mse_loss(mu_pred, y_batch))
            nll = torch.stack(sampled_nll).mean()
            mse_loss = torch.stack(sampled_mse).mean()

            # Smooth warmup: start with pure MSE for mean stability,
            # then cosine-blend into NLL over transition_epochs.
            warmup_epochs = 50
            transition_epochs = 30

            if epoch <= warmup_epochs:
                loss = mse_loss + kl_scale * kl
            else:
                progress = min(1.0, (epoch - warmup_epochs) / transition_epochs)
                alpha = 0.5 * (1.0 + math.cos(math.pi * progress))
                loss = alpha * mse_loss + (1.0 - alpha) * nll + kl_scale * kl
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            total_nll += nll.item()
            total_kl += kl.item()
            
            pbar.set_postfix({
                'Loss': loss.item(),
                'NLL': nll.item(),
                'KL': kl.item(),
                'KL_scale': kl_scale
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_nll = total_nll / len(self.train_loader)
        avg_kl = total_kl / len(self.train_loader)
        
        self.history['train_loss'].append(avg_loss)
        self.history['train_nll'].append(avg_nll)
        self.history['train_kl'].append(avg_kl)
        
        return avg_loss
    
    def validate(self, num_mc_samples=10, min_delta=0.0, validation_seed=1729):
        """
        验证模型：计算验证集上的 ELBO 和 MSE
        """
        if self.val_loader is None:
            return None, None
        
        self.model.eval()
        total_loss = 0.0
        total_nll = 0.0
        total_mse = 0.0
        
        # 获取验证集总样本数用于 KL 缩放
        num_val_samples = len(self.val_loader.dataset)
        
        rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(int(validation_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(validation_seed))
        with torch.no_grad():
            for x_batch, y_batch in self.val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # 1. 计算似然项 (NLL)
                # 验证时通常使用 sample=False (均值模式) 或者少量采样
                sampled_predictions = [
                    self.model(x_batch, sample=True)[0]
                    for _ in range(max(1, int(num_mc_samples)))
                ]
                
                # 获取当前批次的 KL 散度
                kl_div = self.model.kl_divergence()
                
                # 调用 criterion 计算验证集 ELBO (包含 NLL 和缩放后的 KL)
                # 注意：这里我们传入验证集的总样本数进行 KL 归一化
                sampled_nll = []
                for y_pred in sampled_predictions:
                    _, sample_nll, _ = self.criterion(
                        y_pred, y_batch, kl_div, num_val_samples
                    )
                    sampled_nll.append(sample_nll)
                val_nll = torch.stack(sampled_nll).mean()
                # Select checkpoints by held-out expected NLL.  Re-applying KL
                # with the much smaller validation-set denominator biases model
                # selection toward overly diffuse posteriors.
                val_elbo = val_nll
                
                total_loss += val_elbo.item()
                total_nll += val_nll.item()
                predictive_output = torch.stack(sampled_predictions).mean(dim=0)
                
                # 2. 计算标准 MSE 用于常规监控
                if y_pred.shape[-1] == 2: # 异方差模式提取均值
                    y_pred_mu = predictive_output[..., 0:1]
                else:
                    y_pred_mu = predictive_output
                    
                mse = self.mse_loss(y_pred_mu, y_batch)
                total_mse += mse.item()
        
        torch.random.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)

        avg_loss = total_loss / len(self.val_loader)
        avg_mse = total_mse / len(self.val_loader)
        
        # 记录到历史中，这样绘图逻辑就能读取到了
        self.history['val_loss'].append(avg_loss)
        self.history['val_mse'].append(avg_mse)
        
        # 更新最佳模型
        if avg_loss < self.history['best_val_loss'] - float(min_delta):
            self.history['best_val_loss'] = avg_loss
            self.best_model_state = copy.deepcopy(self.model.state_dict())
        
        return avg_loss, avg_mse
    
    def train(
        self,
        num_epochs,
        val_freq=10,
        train_mc_samples=1,
        validation_mc_samples=10,
        early_stopping_patience=None,
        early_stopping_min_delta=0.0,
        validation_seed=1729,
    ):
        """
        训练模型
        
        Args:
        -----
            num_epochs : int
                训练 epoch 数
                
            val_freq : int
                验证频率。默认: 每 10 个 epoch 验证一次
        """
        self.model.update_grid(self.train_loader)
        checks_without_improvement = 0
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch, num_mc_samples=train_mc_samples)

            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            
            if epoch % val_freq == 0 and self.val_loader is not None:
                previous_best = self.history["best_val_loss"]
                val_loss, val_mse = self.validate(
                    num_mc_samples=validation_mc_samples,
                    min_delta=early_stopping_min_delta,
                    validation_seed=validation_seed,
                )
                improved = self.history["best_val_loss"] < previous_best
                checks_without_improvement = 0 if improved else checks_without_improvement + 1
                print(f"Epoch {epoch}: Train Loss={train_loss:.6f}, "
                      f"Val Loss={val_loss:.6f}, Val MSE={val_mse:.6f}")
                if (
                    early_stopping_patience is not None
                    and int(early_stopping_patience) > 0
                    and checks_without_improvement >= int(early_stopping_patience)
                ):
                    print(
                        "Early stopping: validation loss did not improve for "
                        f"{checks_without_improvement} checks."
                    )
                    break
            else:
                print(f"Epoch {epoch}: Train Loss={train_loss:.6f}")
        # ==========================================
        # ✅ 新增：训练结束后，时光倒流，恢复到验证集误差最小的那个时刻
        # ==========================================
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
            print(f"\n[Best] Training done. Restored best val state (Best Val Loss: {self.history['best_val_loss']:.6f})")
    
    def evaluate(self, data_loader=None, num_mc_samples=100):
       
        if data_loader is None:
            data_loader = self.test_loader
        
        if data_loader is None:
            raise ValueError("No test loader provided")
        
        self.model.eval()
        
        all_preds = []
        all_means = []
        all_vars = []
        all_targets = []
        
        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # 蒙特卡洛采样
                samples, _, _ = self.model.forward_sample(x_batch, num_samples=num_mc_samples)
                
                # ✅ 核心修改：重新解耦总方差
                if samples.shape[-1] == 2 and y_batch.shape[-1] == 1:
                    mu_samples = samples[..., 0:1]
                    variance_logits = samples[..., 1:2]
                    
                    batch_mean = mu_samples.mean(dim=0)
                    # 认知方差(均值的方差) + 偶然方差(对数方差取指数后的期望)
                    batch_var = (
                        mu_samples.var(dim=0)
                        + variance_from_logits(variance_logits).mean(dim=0)
                    )
                else:
                    batch_mean = samples.mean(dim=0)
                    batch_var = samples.var(dim=0) + 0.0001

                all_preds.append(samples)
                all_means.append(batch_mean)
                all_vars.append(batch_var)
                all_targets.append(y_batch)
        
        # 合并批次
        all_means = torch.cat(all_means, dim=0)
        all_vars = torch.cat(all_vars, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # 计算指标
        mse = self.mse_loss(all_means, all_targets).item()
        mae = (torch.abs(all_means - all_targets)).mean().item()
        
        # RMSE
        rmse = torch.sqrt(torch.tensor(mse)).item()
        
        # ==========================================
        # ✅ 修复：总方差 = 认知方差 (all_vars) + 偶然方差
        # 在对数空间，我们假设似然的高斯方差为 1.0 (对应 log_devs2=0)
        # ==========================================
       
        total_vars = all_vars 
        
        # 不确定性指标使用总方差
        std = torch.sqrt(total_vars)
        
        # NLL 必须使用 total_vars，防止 all_vars 极小时导致除以 0 
        nll = 0.5 * (torch.log(total_vars) + (all_targets - all_means) ** 2 / total_vars).mean().item()
        
        # 校准指标也使用 std (基于 total_vars)
        calibration_error = (torch.abs(all_targets - all_means) - std).abs().mean().item()
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'nll': nll,
            'mean_std': std.mean().item(),
            'calibration_error': calibration_error
        }
        
        return metrics, all_means, all_vars, all_targets
    
    def predict_with_uncertainty(self, x_test, num_mc_samples=100):
        """
        进行预测并返回不确定性
        
        Args:
        -----
            x_test : Tensor
                测试输入
                
            num_mc_samples : int
                蒙特卡洛采样次数
        
        Returns:
        --------
            mean : Tensor
                预测均值
                
            std : Tensor
                预测标准差
                
            epistemic : Tensor
                认知不确定性
                
            confidence : Tensor
                置信度
        """
        self.model.eval()
        
        with torch.no_grad():
            x_test = x_test.to(self.device)
            samples, mean, var = self.model.forward_sample(x_test, num_samples=num_mc_samples)
            
            std = torch.sqrt(var)
            epistemic = epistemic_uncertainty(samples)
            confidence = confidence_score(mean, std, metric='sigmoid')
        
        return mean, std, epistemic, confidence
    
    def detect_ood(self, x_test, ood_samples, metric='entropy'):
        """
        检测 OOD (Out-of-Distribution) 样本
        
        Args:
        -----
            x_test : Tensor
                测试输入
                
            ood_samples : Tensor
                OOD 样本（如果为 None，则返回 OOD 分数而不比较）
                
            metric : str
                度量方式
        
        Returns:
        --------
            scores : Tensor
                OOD 分数
                
            auroc : float
                AUROC（如果提供了 OOD 样本）
        """
        self.model.eval()
        
        with torch.no_grad():
            x_test = x_test.to(self.device)
            samples_test, _, _ = self.model.forward_sample(x_test)
            scores_test = compute_ood_score(samples_test, metric=metric)
            
            if ood_samples is not None:
                ood_samples = ood_samples.to(self.device)
                samples_ood, _, _ = self.model.forward_sample(ood_samples)
                scores_ood = compute_ood_score(samples_ood, metric=metric)
                
                # 计算 AUROC
                from sklearn.metrics import roc_auc_score
                
                y_true = torch.cat([
                    torch.zeros_like(scores_test),
                    torch.ones_like(scores_ood)
                ]).cpu().numpy()
                
                scores = torch.cat([scores_test, scores_ood]).cpu().numpy()
                
                auroc = roc_auc_score(y_true, scores)
                
                return scores_test, auroc
            else:
                return scores_test, None
    
    def plot_training_history(self, save_path=None):
        """
        绘制训练历史
        
        Args:
        -----
            save_path : str
                保存路径（可选）
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 损失
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        if self.history['val_loss']:
            axes[0, 0].plot(self.history['val_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('ELBO Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # NLL
        axes[0, 1].plot(self.history['train_nll'], label='NLL')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('NLL')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # KL
        axes[1, 0].plot(self.history['train_kl'], label='KL')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('KL Divergence')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # MSE
        if self.history['val_mse']:
            axes[1, 1].plot(self.history['val_mse'], label='Val MSE')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('MSE')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_uncertainty_calibration(self, save_path=None):
        """
        绘制不确定性校准曲线
        
        Args:
        -----
            save_path : str
                保存路径（可选）
        """
        if self.test_loader is None:
            print("No test loader provided")
            return
        
        metrics, means, vars, targets = self.evaluate()
        
        stds = torch.sqrt(vars)
        errors = torch.abs(targets - means)
        
        # 按不确定性排序
        sorted_idx = torch.argsort(stds.squeeze())
        sorted_stds = stds.squeeze()[sorted_idx]
        sorted_errors = errors.squeeze()[sorted_idx]
        
        # 计算分箱统计
        num_bins = 10
        bin_size = len(sorted_stds) // num_bins
        
        bin_stds = []
        bin_errors_mean = []
        bin_errors_std = []
        
        for i in range(num_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < num_bins - 1 else len(sorted_stds)
            
            bin_std = sorted_stds[start:end].mean().item()
            bin_error_mean = sorted_errors[start:end].mean().item()
            bin_error_std = sorted_errors[start:end].std().item()
            
            bin_stds.append(bin_std)
            bin_errors_mean.append(bin_error_mean)
            bin_errors_std.append(bin_error_std)
        
        # 绘图
        plt.figure(figsize=(10, 6))
        
        # 完美校准线
        max_val = max(max(bin_stds), max(bin_errors_mean))
        plt.plot([0, max_val], [0, max_val], 'k--', label='Perfect Calibration', linewidth=2)
        
        # 实际校准
        plt.errorbar(bin_stds, bin_errors_mean, yerr=bin_errors_std,
                    fmt='o-', label='Empirical', capsize=5, markersize=8)
        
        plt.xlabel('Predicted Std Dev')
        plt.ylabel('Absolute Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title('Uncertainty Calibration')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
