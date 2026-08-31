"""
贝叶斯 KAN 物理建模核心工具箱 (bayes_modeler.py)
作为底层算法 (kan/) 与顶层业务脚本 (run_xxx.py) 之间的桥梁。
"""
import os
from contextlib import contextmanager

import torch
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, TensorDataset

# 导入底层贝叶斯算法
from kan import BayesMultKAN, BayesianExperiment, variance_from_logits
from kan.variational_utils import predictive_entropy
from device_modeling.photodetector.prediction_plots import plot_predictions_with_uncertainty


class BayesKANDeviceModeler:
    """贝叶斯 KAN 物理建模器 (通用版)"""
    
    def __init__(self, task_name="default_task", results_dir="artifacts/results", device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.task_name = task_name
        # 隔离不同仿真任务的输出目录
        result_leaf = os.path.basename(os.path.normpath(results_dir))
        if result_leaf.endswith(("-results", "_results")):
            self.save_dir = results_dir
        else:
            self.save_dir = os.path.join(results_dir, f"{self.task_name}_bayes_results")
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.model = None
        self.experiment = None
        self.input_cols = None
        self.output_col = None
        self.scaler_x = None
        self.scaler_y = None
        self.data_stats = {}
        
        # 物理量特性参数
        self.use_log_transform = True
        self.y_bounds = (-np.inf, np.inf)
        self._calib_z = 1.96
        self._calib_method = "theoretical_normal"
        self._calib_sigma_floor = 1e-8
        self._calib_sigma_floor_standardized = 1e-8
        self._calib_raw_quantile = 1.96
        self._calib_unit = "uncalibrated"
        self._calib_group_cols = []
        self._calibration_scores = np.array([], dtype=np.float64)
        self._calib_score_count = 0
        self.inference_method = "vi"
        self.hmc_parameter_samples = []
        self.hmc_chain_parameter_samples = []
        self.hmc_acceptance_rate = np.nan
        self.hmc_chain_acceptance_rates = []
        self.hmc_diagnostics = {}
        self.hmc_mass_matrix_diagnostics = {}

    @staticmethod
    def _conformal_quantile(scores, target_picp):
        scores = np.asarray(scores, dtype=np.float64)
        scores = scores[np.isfinite(scores)]
        if len(scores) == 0:
            return np.nan
        coverage = float(target_picp) / 100.0
        rank = int(np.ceil((len(scores) + 1) * coverage))
        if rank > len(scores):
            raise ValueError(
                f"No finite {coverage:.1%} conformal order statistic: "
                f"n={len(scores)}, required rank={rank}. Increase the number "
                "of independent calibration groups or lower target_picp."
            )
        # Select the one-indexed split-conformal order statistic directly.
        # Mapping rank / n through np.quantile(..., method="higher") is not
        # equivalent when rank < n because NumPy indexes quantiles on n - 1;
        # for example, n=24 and 90% coverage would incorrectly select rank 24
        # instead of the required rank 23.
        return float(np.sort(scores)[rank - 1])

    @contextmanager
    def _fixed_prediction_seed(self):
        """Make posterior Monte Carlo predictions reproducible without altering training RNG."""
        seed = self.bayes_config.get("prediction_seed") if hasattr(self, "bayes_config") else None
        if seed is None:
            yield
            return
        cpu_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        try:
            yield
        finally:
            torch.random.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def _transform_target(self, y_raw):
        """内部方法：处理目标变量的转换"""
        if self.use_log_transform:
            return np.log10(np.maximum(y_raw, 1e-30)).reshape(-1, 1)
        return y_raw.reshape(-1, 1)

    def _filter_valid_data(self, X, y_transformed):
        """内部方法：根据设定的物理边界过滤数据"""
        y_flat = y_transformed.flatten()
        valid = (y_flat > self.y_bounds[0]) & (y_flat < self.y_bounds[1])
        return X[valid], y_transformed[valid]

    def load_data(self, df, input_cols, output_col, use_log_transform=True, y_bounds=(-np.inf, np.inf)):
        self.input_cols = input_cols
        self.output_col = output_col
        self.use_log_transform = use_log_transform
        self.y_bounds = y_bounds
        
        X = df[input_cols].values.astype(np.float32)
        y_raw = df[output_col].values.astype(np.float32)

        y_transformed = self._transform_target(y_raw)
        X, y_transformed = self._filter_valid_data(X, y_transformed)
        
        from sklearn.preprocessing import StandardScaler
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y_transformed)
        
        self.X_train = torch.from_numpy(X_scaled).float().to(self.device)
        self.y_train = torch.from_numpy(y_scaled).float().to(self.device)
        
        self.data_stats = {
            'X_mean': self.scaler_x.mean_, 'X_std': self.scaler_x.scale_,
            'y_mean': self.scaler_y.mean_[0], 'y_std': self.scaler_y.scale_[0],
        }
        print(f"[{self.task_name}] Data loaded: {X_scaled.shape[0]} samples")

    def build_model(self, bayes_config):
        self.bayes_config = bayes_config.copy()
        self.bayes_config['width'][0] = len(self.input_cols)
        self.inference_method = self.bayes_config.get('inference_method', 'vi')
        if self.inference_method in ('dropout', 'hmc'):
            self.bayes_config['kl_weight'] = 0.0
        
        self.model = BayesMultKAN(
            width=self.bayes_config['width'], grid=self.bayes_config['grid'], k=self.bayes_config['k'],
            kl_weight=self.bayes_config['kl_weight'], num_mc_samples=self.bayes_config['num_mc_samples'],
            prior_mu=self.bayes_config.get('prior_mu', 0.0), prior_log_sigma=self.bayes_config.get('prior_log_sigma', 0.0),
            posterior_init_sigma=self.bayes_config.get('posterior_init_sigma', 0.01),
            grid_range=self.bayes_config.get('grid_range', [-3, 3]),
            device=self.device, seed=int(self.bayes_config.get('seed', 42)),
            base_fun=self.bayes_config.get('base_fun', 'silu'),
            inference_method=self.inference_method,
            dropout_rate=self.bayes_config.get('dropout_rate', 0.0),
        )
        print("Model built successfully.")

    def train(self, train_config, df_val=None):
        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
        
        val_loader = None
        if df_val is not None:
            X_val = df_val[self.input_cols].values.astype(np.float32)
            y_val_raw = df_val[self.output_col].values.astype(np.float32)
            y_val_transformed = self._transform_target(y_val_raw)
            X_val, y_val_transformed = self._filter_valid_data(X_val, y_val_transformed)

            X_val_scaled = self.scaler_x.transform(X_val)
            y_val_scaled = self.scaler_y.transform(y_val_transformed)
            
            val_dataset = TensorDataset(torch.from_numpy(X_val_scaled).float().to(self.device), 
                                        torch.from_numpy(y_val_scaled).float().to(self.device))
            val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False)
        
        self.experiment = BayesianExperiment(
            model=self.model, train_loader=train_loader, val_loader=val_loader,
            optimizer_name='adam', lr=train_config['lr'], weight_decay=train_config['weight_decay'],
            device=self.device, likelihood=self.bayes_config['likelihood'], kl_weight=self.bayes_config['kl_weight']
        )
        self.experiment.train(
            num_epochs=train_config['num_epochs'],
            val_freq=train_config['val_freq'],
            train_mc_samples=train_config.get('train_mc_samples', 1),
            validation_mc_samples=train_config.get('validation_mc_samples', 10),
            early_stopping_patience=train_config.get('early_stopping_patience'),
            early_stopping_min_delta=train_config.get('early_stopping_min_delta', 0.0),
            validation_seed=train_config.get('validation_seed', 1729),
        )
        self._estimate_global_aleatoric_var()
        if self.inference_method == "hmc" and not self.bayes_config.get("hmc_defer_sampling", False):
            self._fit_hmc_posterior()

    def _hmc_parameters(self):
        if self.model is None:
            return []
        named_params = [
            (name, param) for name, param in self.model.named_parameters()
            if param.requires_grad and name.endswith("_mu")
        ]
        scope = getattr(self, "bayes_config", {}).get("hmc_parameter_scope", "all")
        if scope == "all":
            return [param for _, param in named_params]
        if scope not in {"last_layer", "last_layer_coef"}:
            raise ValueError(f"Unknown HMC parameter scope: {scope}")

        layer_indices = [
            int(name.split(".")[1])
            for name, _ in named_params
            if name.startswith("layers.") and name.split(".")[1].isdigit()
        ]
        if not layer_indices:
            raise RuntimeError("No layer parameters found for last-layer HMC")
        last_layer_prefix = f"layers.{max(layer_indices)}."
        return [
            param for name, param in named_params
            if name.startswith(last_layer_prefix)
            and (scope == "last_layer" or name.endswith(".coef_mu"))
        ]

    def _hmc_log_prob_and_grad(self, theta, params, noise_var, prior_var):
        vector_to_parameters(theta, params)
        self.model.zero_grad(set_to_none=True)
        pred, _ = self.model(self.X_train, sample=False)
        mu = pred[:, 0:1] if pred.ndim == 2 else pred
        residual = mu - self.y_train
        log_like = -0.5 * residual.pow(2).sum() / noise_var
        grads = torch.autograd.grad(log_like, params, allow_unused=True)
        grad_parts = [
            torch.zeros_like(param).reshape(-1) if grad is None else grad.reshape(-1)
            for param, grad in zip(params, grads)
        ]
        grad_vec = torch.cat(grad_parts) - theta / prior_var
        log_prior = -0.5 * theta.pow(2).sum() / prior_var
        return (log_like + log_prior).detach(), grad_vec.detach()

    def _hmc_hessian_mass_matrix(self, theta, params, noise_var, prior_var):
        vector_to_parameters(theta, params)
        self.model.zero_grad(set_to_none=True)
        pred, _ = self.model(self.X_train, sample=False)
        mu = pred[:, 0:1] if pred.ndim == 2 else pred
        residual = mu - self.y_train
        potential = (
            0.5 * residual.pow(2).sum() / noise_var
            + 0.5 * sum(param.pow(2).sum() for param in params) / prior_var
        )
        first_grads = torch.autograd.grad(
            potential, params, create_graph=True, allow_unused=True
        )
        grad_vec = torch.cat(
            [
                torch.zeros_like(param).reshape(-1)
                if grad is None
                else grad.reshape(-1)
                for param, grad in zip(params, first_grads)
            ]
        )
        rows = []
        for row_idx in range(grad_vec.numel()):
            second_grads = torch.autograd.grad(
                grad_vec[row_idx],
                params,
                retain_graph=row_idx < grad_vec.numel() - 1,
                allow_unused=True,
            )
            rows.append(
                torch.cat(
                    [
                        torch.zeros_like(param).reshape(-1)
                        if grad is None
                        else grad.reshape(-1)
                        for param, grad in zip(params, second_grads)
                    ]
                ).detach()
            )
        hessian = torch.stack(rows).double()
        hessian = 0.5 * (hessian + hessian.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
        floor = max(1e-6, float(torch.max(eigenvalues).item()) * 1e-10)
        eigenvalues = torch.clamp(eigenvalues, min=floor)
        mass_sqrt = (
            eigenvectors
            @ torch.diag(torch.sqrt(eigenvalues))
            @ eigenvectors.T
        ).to(dtype=theta.dtype, device=self.device)
        inverse_mass = (
            eigenvectors
            @ torch.diag(torch.reciprocal(eigenvalues))
            @ eigenvectors.T
        ).to(dtype=theta.dtype, device=self.device)
        diagnostics = {
            "mode": "hessian",
            "eigenvalue_min": float(torch.min(eigenvalues).item()),
            "eigenvalue_max": float(torch.max(eigenvalues).item()),
            "condition_number": float(
                (torch.max(eigenvalues) / torch.min(eigenvalues)).item()
            ),
        }
        self.model.zero_grad(set_to_none=True)
        return mass_sqrt, inverse_mass, diagnostics

    def _fit_hmc_posterior(self):
        params = self._hmc_parameters()
        if not params:
            self.hmc_parameter_samples = []
            self.hmc_chain_parameter_samples = []
            self.hmc_acceptance_rate = np.nan
            self.hmc_chain_acceptance_rates = []
            self.hmc_diagnostics = {}
            self.hmc_mass_matrix_diagnostics = {}
            return

        n_samples = int(self.bayes_config.get("hmc_samples", self.bayes_config.get("num_mc_samples", 100)))
        burn_in = int(self.bayes_config.get("hmc_burn_in", 40))
        leapfrog_steps = int(self.bayes_config.get("hmc_leapfrog_steps", 8))
        step_size = float(self.bayes_config.get("hmc_step_size", 1e-3))
        thinning = max(1, int(self.bayes_config.get("hmc_thinning", 1)))
        prior_scale = float(self.bayes_config.get("hmc_prior_scale", 1.0))
        n_chains = max(1, int(self.bayes_config.get("hmc_chains", 1)))
        chain_init_jitter = max(0.0, float(self.bayes_config.get("hmc_chain_init_jitter", 0.0)))
        hmc_seed = int(self.bayes_config.get("hmc_seed", 42))
        mass_matrix_mode = self.bayes_config.get("hmc_mass_matrix", "identity")
        prior_var = max(prior_scale * prior_scale, 1e-8)
        noise_var = max(float(getattr(self, "global_aleatoric_var", 1e-4)), 1e-6)

        original_theta = parameters_to_vector(params).detach().to(self.device).clone()
        if mass_matrix_mode == "hessian":
            if self.bayes_config.get("hmc_parameter_scope") != "last_layer_coef":
                raise ValueError(
                    "Hessian HMC mass matrix currently requires "
                    "hmc_parameter_scope=last_layer_coef"
                )
            mass_sqrt, inverse_mass, mass_diagnostics = (
                self._hmc_hessian_mass_matrix(
                    original_theta, params, noise_var, prior_var
                )
            )
        elif mass_matrix_mode == "identity":
            mass_sqrt = None
            inverse_mass = None
            mass_diagnostics = {"mode": "identity"}
        else:
            raise ValueError(f"Unknown HMC mass matrix: {mass_matrix_mode}")
        self.hmc_mass_matrix_diagnostics = mass_diagnostics
        chain_samples = []
        chain_acceptance_rates = []
        total_iters = burn_in + n_samples * thinning

        print(
            f"Running HMC posterior sampling: chains={n_chains}, samples={n_samples}, "
            f"burn_in={burn_in}, leapfrog={leapfrog_steps}, step_size={step_size:g}, "
            f"noise_var={noise_var:g}, mass_matrix={mass_matrix_mode}"
        )
        if mass_matrix_mode == "hessian":
            print(
                "  Hessian mass matrix: "
                f"condition={mass_diagnostics['condition_number']:.3g}, "
                f"eigenvalues=[{mass_diagnostics['eigenvalue_min']:.3g}, "
                f"{mass_diagnostics['eigenvalue_max']:.3g}]"
            )
        for chain_idx in range(n_chains):
            torch.manual_seed(hmc_seed + chain_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(hmc_seed + chain_idx)

            theta = original_theta.clone()
            if chain_init_jitter > 0.0:
                theta = theta + chain_init_jitter * torch.randn_like(theta)
            vector_to_parameters(theta, params)
            current_logp, current_grad = self._hmc_log_prob_and_grad(
                theta, params, noise_var, prior_var
            )
            samples = []
            accepted = 0

            for iteration in range(total_iters):
                standard_momentum = torch.randn_like(theta)
                momentum = (
                    standard_momentum
                    if mass_sqrt is None
                    else mass_sqrt @ standard_momentum
                )
                proposal_theta = theta.clone()
                proposal_momentum = momentum + 0.5 * step_size * current_grad
                proposal_logp = current_logp
                proposal_grad = current_grad

                finite_proposal = True
                for leapfrog_idx in range(leapfrog_steps):
                    velocity = (
                        proposal_momentum
                        if inverse_mass is None
                        else inverse_mass @ proposal_momentum
                    )
                    proposal_theta = proposal_theta + step_size * velocity
                    proposal_logp, proposal_grad = self._hmc_log_prob_and_grad(
                        proposal_theta, params, noise_var, prior_var
                    )
                    if (
                        not torch.isfinite(proposal_logp).item()
                        or not torch.isfinite(proposal_grad).all().item()
                    ):
                        finite_proposal = False
                        break
                    if leapfrog_idx != leapfrog_steps - 1:
                        proposal_momentum = proposal_momentum + step_size * proposal_grad

                if finite_proposal:
                    proposal_momentum = proposal_momentum + 0.5 * step_size * proposal_grad
                    current_velocity = (
                        momentum
                        if inverse_mass is None
                        else inverse_mass @ momentum
                    )
                    proposal_velocity = (
                        proposal_momentum
                        if inverse_mass is None
                        else inverse_mass @ proposal_momentum
                    )
                    current_energy = (
                        -current_logp + 0.5 * torch.dot(momentum, current_velocity)
                    )
                    proposal_energy = (
                        -proposal_logp
                        + 0.5 * torch.dot(proposal_momentum, proposal_velocity)
                    )
                    log_accept = torch.clamp(current_energy - proposal_energy, max=0.0)
                    accept = torch.log(torch.rand((), device=self.device)) < log_accept
                else:
                    accept = torch.tensor(False, device=self.device)

                if bool(accept.item()):
                    theta = proposal_theta.detach()
                    current_logp = proposal_logp.detach()
                    current_grad = proposal_grad.detach()
                    accepted += 1
                else:
                    vector_to_parameters(theta, params)

                if iteration >= burn_in and (iteration - burn_in) % thinning == 0:
                    samples.append(theta.detach().cpu())

            acceptance_rate = accepted / max(total_iters, 1)
            chain_samples.append(samples)
            chain_acceptance_rates.append(acceptance_rate)
            print(
                f"  HMC chain {chain_idx + 1}/{n_chains} acceptance rate: "
                f"{acceptance_rate:.3f}"
            )

        vector_to_parameters(original_theta, params)
        self.hmc_chain_parameter_samples = chain_samples
        self.hmc_parameter_samples = [
            sample for samples in chain_samples for sample in samples
        ]
        self.hmc_chain_acceptance_rates = chain_acceptance_rates
        self.hmc_acceptance_rate = float(np.mean(chain_acceptance_rates))
        self.hmc_diagnostics = self._hmc_convergence_diagnostics(chain_samples)
        print(f"HMC mean acceptance rate: {self.hmc_acceptance_rate:.3f}")
        if self.hmc_diagnostics:
            print(
                "HMC convergence diagnostics: "
                f"max split R-hat={self.hmc_diagnostics['rhat_max']:.3f}, "
                f"min bulk ESS={self.hmc_diagnostics['ess_bulk_min']:.1f}"
            )

    @staticmethod
    def _hmc_convergence_diagnostics(chain_samples):
        if len(chain_samples) < 2 or min(map(len, chain_samples), default=0) < 4:
            return {}

        draws = np.stack(
            [
                np.stack([sample.numpy() for sample in samples], axis=0)
                for samples in chain_samples
            ],
            axis=0,
        ).astype(np.float64, copy=False)

        half = draws.shape[1] // 2
        split = np.concatenate([draws[:, :half, :], draws[:, -half:, :]], axis=0)
        n = split.shape[1]
        chain_means = np.mean(split, axis=1)
        within_vars = np.var(split, axis=1, ddof=1)
        within_var = np.mean(within_vars, axis=0)
        between_var = n * np.var(chain_means, axis=0, ddof=1)
        var_plus = ((n - 1) / n) * within_var + between_var / n
        with np.errstate(divide="ignore", invalid="ignore"):
            rhat = np.sqrt(var_plus / within_var)

        centered = split - chain_means[:, None, :]
        fft_size = 1 << (2 * n - 1).bit_length()
        spectrum = np.fft.rfft(centered, n=fft_size, axis=1)
        autocov = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size, axis=1)
        autocov = autocov[:, :n, :] / np.arange(n, 0, -1)[None, :, None]
        mean_autocov = np.mean(autocov, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = 1.0 - (within_var[None, :] - mean_autocov) / var_plus[None, :]
        rho[0, :] = 1.0

        pair_sums = rho[1:-1:2, :] + rho[2::2, :]
        positive_pairs = np.zeros_like(pair_sums)
        active = np.ones(pair_sums.shape[1], dtype=bool)
        previous = np.full(pair_sums.shape[1], np.inf)
        for pair_idx in range(pair_sums.shape[0]):
            current = pair_sums[pair_idx]
            active &= np.isfinite(current) & (current > 0.0)
            monotone = np.minimum(current, previous)
            positive_pairs[pair_idx, active] = monotone[active]
            previous[active] = monotone[active]

        tau = 1.0 + 2.0 * np.sum(positive_pairs, axis=0)
        total_draws = split.shape[0] * split.shape[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            ess = np.minimum(total_draws / tau, total_draws)

        finite_rhat = rhat[np.isfinite(rhat)]
        finite_ess = ess[np.isfinite(ess)]
        if not len(finite_rhat) or not len(finite_ess):
            return {}
        return {
            "rhat_max": float(np.max(finite_rhat)),
            "rhat_p99": float(np.quantile(finite_rhat, 0.99)),
            "rhat_median": float(np.median(finite_rhat)),
            "ess_bulk_min": float(np.min(finite_ess)),
            "ess_bulk_p01": float(np.quantile(finite_ess, 0.01)),
            "ess_bulk_median": float(np.median(finite_ess)),
            "parameter_count": int(draws.shape[2]),
        }

    def _predict_hmc_samples(self, X_tensor):
        params = self._hmc_parameters()
        if not params or not self.hmc_parameter_samples:
            return None

        original_theta = parameters_to_vector(params).detach()
        outputs = []
        for theta_cpu in self.hmc_parameter_samples:
            vector_to_parameters(theta_cpu.to(self.device), params)
            pred, _ = self.model(X_tensor, sample=False)
            outputs.append(pred[:, 0:1])
        vector_to_parameters(original_theta, params)
        return torch.stack(outputs, dim=0)

    def hmc_function_space_diagnostics(self, X):
        if len(self.hmc_chain_parameter_samples) < 2:
            return {}
        X_scaled = self.scaler_x.transform(np.asarray(X, dtype=np.float32))
        X_tensor = torch.from_numpy(X_scaled.astype(np.float32)).to(self.device)
        params = self._hmc_parameters()
        original_theta = parameters_to_vector(params).detach().clone()
        function_chains = []
        with torch.no_grad():
            for chain in self.hmc_chain_parameter_samples:
                function_samples = []
                for theta_cpu in chain:
                    vector_to_parameters(theta_cpu.to(self.device), params)
                    prediction, _ = self.model(X_tensor, sample=False)
                    function_samples.append(prediction[:, 0].detach().cpu())
                function_chains.append(function_samples)
        vector_to_parameters(original_theta, params)

        diagnostics = self._hmc_convergence_diagnostics(function_chains)
        if not diagnostics:
            return {}
        draws = np.stack(
            [
                np.stack([sample.numpy() for sample in chain], axis=0)
                for chain in function_chains
            ],
            axis=0,
        )
        chain_means = np.mean(draws, axis=1)
        pairwise_rmse = [
            float(np.sqrt(np.mean((chain_means[left] - chain_means[right]) ** 2)))
            for left in range(len(chain_means))
            for right in range(left)
        ]
        return {
            key: value
            for key, value in diagnostics.items()
            if key != "parameter_count"
        } | {
            "point_count": int(draws.shape[2]),
            "chain_mean_pairwise_rmse_max": float(max(pairwise_rmse, default=np.nan)),
            "within_chain_std_median": float(np.median(np.std(draws, axis=1))),
        }

    def cross_validate(self, df_data, bayes_config, train_config, n_folds=5, random_state=42):
        """K-fold cross-validation.

        Trains *n_folds* independent models and returns per-fold metrics so the
        caller can report mean ± std.  The final ``self`` is left with the best-
        validation model from the last fold (a convenience, not a blend).
        """
        from sklearn.model_selection import KFold

        X_all = df_data[self.input_cols].values.astype(np.float32)
        y_raw = df_data[self.output_col].values.astype(np.float32)
        y_transformed = self._transform_target(y_raw)
        X_all, y_transformed = self._filter_valid_data(X_all, y_transformed)

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_metrics = []
        best_overall = (float("inf"), None, None)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_all)):
            print(f"\n{'─' * 50}")
            print(f"  Cross-validation fold {fold_idx + 1}/{n_folds}")
            print(f"{'─' * 50}")

            X_tr, X_val = X_all[train_idx], X_all[val_idx]
            y_tr, y_val = y_transformed[train_idx], y_transformed[val_idx]

            from sklearn.preprocessing import StandardScaler
            fold_scaler_x = StandardScaler()
            fold_scaler_y = StandardScaler()
            X_tr_s = fold_scaler_x.fit_transform(X_tr)
            y_tr_s = fold_scaler_y.fit_transform(y_tr)
            X_val_s = fold_scaler_x.transform(X_val)
            y_val_s = fold_scaler_y.transform(y_val)

            X_tr_t = torch.from_numpy(X_tr_s).float().to(self.device)
            y_tr_t = torch.from_numpy(y_tr_s).float().to(self.device)
            X_val_t = torch.from_numpy(X_val_s).float().to(self.device)
            y_val_t = torch.from_numpy(y_val_s).float().to(self.device)

            train_ds = TensorDataset(X_tr_t, y_tr_t)
            train_loader = DataLoader(train_ds, batch_size=train_config['batch_size'], shuffle=True)
            val_ds = TensorDataset(X_val_t, y_val_t)
            val_loader = DataLoader(val_ds, batch_size=train_config['batch_size'], shuffle=False)

            fold_config = bayes_config.copy()
            fold_config['width'][0] = len(self.input_cols)

            model = BayesMultKAN(
                width=fold_config['width'], grid=fold_config['grid'], k=fold_config['k'],
                kl_weight=fold_config['kl_weight'], num_mc_samples=fold_config['num_mc_samples'],
                prior_mu=fold_config.get('prior_mu', 0.0),
                prior_log_sigma=fold_config.get('prior_log_sigma', 0.0),
                posterior_init_sigma=fold_config.get('posterior_init_sigma', 0.01),
                device=self.device, seed=random_state + fold_idx,
                base_fun=fold_config.get('base_fun', 'silu'),
            )

            exp = BayesianExperiment(
                model=model, train_loader=train_loader, val_loader=val_loader,
                optimizer_name='adam', lr=train_config['lr'],
                weight_decay=train_config['weight_decay'],
                device=self.device, likelihood=fold_config['likelihood'],
                kl_weight=fold_config['kl_weight'],
            )
            exp.train(num_epochs=train_config['num_epochs'], val_freq=train_config['val_freq'])

            # Evaluate on this fold's validation split
            with torch.no_grad():
                samples, _, _ = model.forward_sample(
                    X_val_t, num_samples=fold_config.get('num_mc_samples', 100)
                )
            samples_np = samples.cpu().numpy()
            mu_samples = samples_np[:, :, 0:1]
            mean_np = np.mean(mu_samples, axis=0)

            val_y_raw = df_data[self.output_col].values[val_idx].astype(np.float32)
            y_val_log = np.log10(np.maximum(val_y_raw, 1e-30))
            pred_log = fold_scaler_y.inverse_transform(mean_np).reshape(-1)

            valid = (y_val_log > self.y_bounds[0]) & (y_val_log < self.y_bounds[1])
            rmse = np.sqrt(np.mean((pred_log[valid] - y_val_log[valid]) ** 2))
            mae = np.mean(np.abs(pred_log[valid] - y_val_log[valid]))

            fold_metrics.append({"fold": fold_idx + 1, "RMSE_Log": rmse, "MAE_Log": mae})

            if exp.history["best_val_loss"] < best_overall[0]:
                best_overall = (exp.history["best_val_loss"], model, fold_scaler_x)

        # Restore best model and scaler for convenience
        if best_overall[1] is not None:
            self.model = best_overall[1]
            self.scaler_x = best_overall[2]

        rmse_vals = [m["RMSE_Log"] for m in fold_metrics]
        mae_vals = [m["MAE_Log"] for m in fold_metrics]
        print(f"\n K-fold CV summary ({n_folds} folds):")
        print(f"   RMSE_Log = {np.mean(rmse_vals):.4f} ± {np.std(rmse_vals):.4f}")
        print(f"   MAE_Log  = {np.mean(mae_vals):.4f} ± {np.std(mae_vals):.4f}")

        return {"folds": fold_metrics, "RMSE_Log_mean": np.mean(rmse_vals),
                "RMSE_Log_std": np.std(rmse_vals), "MAE_Log_mean": np.mean(mae_vals),
                "MAE_Log_std": np.std(mae_vals)}

    def _estimate_global_aleatoric_var(self):
        """Estimate a global aleatoric noise floor from training residuals.

        Only used when the output layer has a single channel (no predicted
        log-variance).  For heteroscedastic models (2-channel output) the
        per-sample aleatoric variance is predicted directly.
        """
        if self.bayes_config['width'][-1] == 2:
            self.global_aleatoric_var = 0.0
            return

        with torch.no_grad():
            samples, _, _ = self.model.forward_sample(
                self.X_train, num_samples=min(50, self.bayes_config.get('num_mc_samples', 100))
            )
            mean_pred = samples.mean(dim=0)
            residuals = self.y_train - mean_pred
            mse = (residuals ** 2).mean().item()
            self.global_aleatoric_var = max(mse, 1e-6)

    def estimate_residual_noise_var(self, df):
        """Estimate scaled target-space residual variance on a held-out frame."""
        if df is None or len(df) == 0:
            raise ValueError("Cannot estimate residual noise from an empty frame")

        X_np = df[self.input_cols].values.astype(np.float32)
        y_raw = df[self.output_col].values.astype(np.float32)
        y_transformed = self._transform_target(y_raw)
        X_np, y_transformed = self._filter_valid_data(X_np, y_transformed)
        if len(X_np) == 0:
            raise ValueError("No valid rows remain for residual noise estimation")

        X_scaled = self.scaler_x.transform(X_np).astype(np.float32)
        y_scaled = self.scaler_y.transform(y_transformed)
        X_tensor = torch.from_numpy(X_scaled).float().to(self.device)
        with torch.no_grad():
            pred, _ = self.model(X_tensor, sample=False)
        mu = pred[:, 0:1].detach().cpu().numpy()
        return float(np.mean((mu - y_scaled) ** 2))

    def calibrate_uncertainty(
        self,
        df_calib=None,
        target_picp=95.0,
        min_z=1.96,
        calibration_unit="point",
        group_cols=None,
    ):
        """Calibrate prediction intervals with normalized conformal scaling.

        Scores are computed in standardized model-target space as
        abs(y - mean) / max(std, sigma_floor).  ``calibration_unit="curve"``
        is the legacy name for grouped calibration: it reduces point scores to
        the maximum score per independent split group before taking the
        conformal quantile.  A capacitance group is a complete TCAD response
        surface rather than one of its frequency sub-curves.
        By default conformal calibration is not allowed to shrink the
        theoretical 95% Gaussian interval; it can only preserve or widen it.
        """
        if self.model is None:
            print("  Calibration skipped: model not trained")
            return 1.96

        if df_calib is not None:
            X_np = df_calib[self.input_cols].values.astype(np.float32)
            y_raw = df_calib[self.output_col].values.astype(np.float32)
            y_true = self._transform_target(y_raw).flatten()
        else:
            X_np = self.scaler_x.inverse_transform(
                self.X_train.detach().cpu().numpy()
            ).astype(np.float32)
            y_np = self.y_train.cpu().numpy()
            y_true = self.scaler_y.inverse_transform(y_np).flatten()

        pred = self.predict_with_uncertainty(X_np)
        log_mean = np.asarray(pred["log_mean"], dtype=np.float64)
        log_std = np.asarray(pred["log_std"], dtype=np.float64)

        # Compute normalized residual scores in standardized target space.
        # A fixed floor such as 1e-12 in physical target units is not scale
        # invariant and overwhelms capacitance uncertainties around 1e-19 F.
        target_center = float(self.scaler_y.mean_[0])
        target_scale = max(
            abs(float(self.scaler_y.scale_[0])),
            np.finfo(np.float64).tiny,
        )
        y_true_standardized = (y_true - target_center) / target_scale
        mean_standardized = (log_mean - target_center) / target_scale
        std_standardized = log_std / target_scale

        positive_std = std_standardized[
            np.isfinite(std_standardized) & (std_standardized > 0.0)
        ]
        if len(positive_std):
            sigma_floor_standardized = max(
                float(np.median(positive_std)) * 0.1,
                1e-8,
            )
        else:
            sigma_floor_standardized = 1e-8
        sigma_floor = sigma_floor_standardized * target_scale

        point_scores = np.abs(y_true_standardized - mean_standardized) / np.maximum(
            std_standardized,
            sigma_floor_standardized,
        )
        finite_scores = np.isfinite(point_scores)
        if not np.any(finite_scores):
            print("  Calibration: no valid z-scores, keeping z=1.96")
            self._calib_z = 1.96
            self._calib_method = "fallback_theoretical_normal"
            self._calib_sigma_floor = float(sigma_floor)
            self._calib_sigma_floor_standardized = float(
                sigma_floor_standardized
            )
            self._calib_raw_quantile = np.nan
            self._calibration_scores = np.array([], dtype=np.float64)
            self._calib_score_count = 0
            return 1.96

        calibration_unit = str(calibration_unit).lower()
        if calibration_unit not in {"point", "curve"}:
            raise ValueError(f"Unknown calibration unit: {calibration_unit}")

        if calibration_unit == "curve":
            if df_calib is None:
                raise ValueError("Curve-level conformal calibration requires df_calib")
            group_cols = list(group_cols or [])
            missing_group_cols = [col for col in group_cols if col not in df_calib.columns]
            if missing_group_cols:
                raise KeyError(f"Missing conformal group columns: {missing_group_cols}")
            if not group_cols:
                raise ValueError("Curve-level conformal calibration requires group_cols")
            score_frame = df_calib.reset_index(drop=True)[group_cols].copy()
            score_frame["_conformal_score"] = point_scores
            score_frame = score_frame.loc[finite_scores]
            calibration_scores = (
                score_frame.groupby(group_cols, dropna=False)["_conformal_score"]
                .max()
                .to_numpy(dtype=np.float64)
            )
        else:
            group_cols = []
            calibration_scores = point_scores[finite_scores].astype(np.float64, copy=False)

        q_raw = self._conformal_quantile(calibration_scores, target_picp)
        self._calib_z = float(q_raw if min_z is None else max(q_raw, float(min_z)))
        self._calib_method = f"normalized_conformal_{calibration_unit}"
        self._calib_sigma_floor = float(sigma_floor)
        self._calib_sigma_floor_standardized = float(
            sigma_floor_standardized
        )
        self._calib_raw_quantile = float(q_raw)
        self._calib_unit = calibration_unit
        self._calib_group_cols = list(group_cols)
        self._calibration_scores = np.asarray(calibration_scores, dtype=np.float64)
        self._calib_score_count = int(len(calibration_scores))
        src = "calibration set" if df_calib is not None else "training set"
        floor_text = "none" if min_z is None else f"{float(min_z):.2f}"
        print(
            f"  Calibration on {src}: conformal z = {self._calib_z:.3f} "
            f"(raw q={q_raw:.3f}, floor={floor_text}, unit={calibration_unit}, "
            f"n={len(calibration_scores)}) for {target_picp:.0f}% coverage"
        )

        calib_path = os.path.join(self.save_dir, "calibration_z.txt")
        with open(calib_path, "w", encoding="utf-8") as f:
            f.write(f"{self._calib_z:.6f}\n")
            f.write(f"method={self._calib_method}\n")
            f.write(f"raw_quantile={self._calib_raw_quantile:.6f}\n")
            f.write(f"sigma_floor={self._calib_sigma_floor:.12g}\n")
            f.write(
                "sigma_floor_standardized="
                f"{self._calib_sigma_floor_standardized:.12g}\n"
            )
            f.write(f"target_picp={target_picp:.6f}\n")
            f.write(f"min_z={'none' if min_z is None else f'{float(min_z):.6f}'}\n")
            f.write(f"calibration_unit={self._calib_unit}\n")
            f.write(f"score_count={self._calib_score_count}\n")
            f.write(f"group_cols={','.join(self._calib_group_cols)}\n")
        return self._calib_z

    def save_model(self, path=None):
        """保存模型权重、scaler 和配置。"""
        if path is None:
            path = os.path.join(self.save_dir, "model_checkpoint.pt")
        state = {
            'model_state': self.model.state_dict(),
            'scaler_x': self.scaler_x,
            'scaler_y': self.scaler_y,
            'data_stats': self.data_stats,
            'input_cols': self.input_cols,
            'output_col': self.output_col,
            'bayes_config': self.bayes_config,
            'task_name': self.task_name,
            'use_log_transform': self.use_log_transform,
            'calibration_z': getattr(self, '_calib_z', 1.96),
            'calibration_method': getattr(self, '_calib_method', 'theoretical_normal'),
            'calibration_sigma_floor': getattr(self, '_calib_sigma_floor', 1e-8),
            'calibration_sigma_floor_standardized': getattr(
                self, '_calib_sigma_floor_standardized', 1e-8
            ),
            'calibration_raw_quantile': getattr(self, '_calib_raw_quantile', 1.96),
            'calibration_unit': getattr(self, '_calib_unit', 'point'),
            'calibration_group_cols': getattr(self, '_calib_group_cols', []),
            'calibration_scores': getattr(
                self, '_calibration_scores', np.array([], dtype=np.float64)
            ),
            'calibration_score_count': getattr(self, '_calib_score_count', 0),
            'variance_parameterization': 'softplus_variance_v1',
            'inference_method': getattr(self, 'inference_method', 'vi'),
            'hmc_acceptance_rate': getattr(self, 'hmc_acceptance_rate', np.nan),
            'hmc_parameter_samples': getattr(self, 'hmc_parameter_samples', []),
            'hmc_chain_parameter_samples': getattr(self, 'hmc_chain_parameter_samples', []),
            'hmc_chain_acceptance_rates': getattr(self, 'hmc_chain_acceptance_rates', []),
            'hmc_diagnostics': getattr(self, 'hmc_diagnostics', {}),
            'hmc_mass_matrix_diagnostics': getattr(
                self, 'hmc_mass_matrix_diagnostics', {}
            ),
            'global_aleatoric_var': getattr(self, 'global_aleatoric_var', None),
        }
        # scaler 用 pickle 序列化
        scaler_data = {
            'scaler_x_mean': self.scaler_x.mean_,
            'scaler_x_scale': self.scaler_x.scale_,
            'scaler_y_mean': self.scaler_y.mean_,
            'scaler_y_scale': self.scaler_y.scale_,
        }
        state['scaler_data'] = scaler_data
        torch.save(state, path)
        print(f"模型已保存: {path}")
        return path

    @classmethod
    def load_model(cls, path, device='cpu'):
        """从保存的 checkpoint 加载模型。"""
        from sklearn.preprocessing import StandardScaler
        state = torch.load(path, map_location=device, weights_only=False)

        # 重建 modeler
        modeler = cls(
            task_name=state.get('task_name', 'loaded'),
            results_dir=os.path.dirname(path),
            device=device,
        )
        modeler.input_cols = state['input_cols']
        modeler.output_col = state['output_col']
        modeler.data_stats = state['data_stats']
        modeler.use_log_transform = state.get('use_log_transform', True)
        modeler.bayes_config = state.get('bayes_config', {})
        modeler._calib_z = float(state.get('calibration_z', 1.96))
        modeler._calib_method = state.get('calibration_method', 'theoretical_normal')
        modeler._calib_sigma_floor = float(state.get('calibration_sigma_floor', 1e-8))
        modeler._calib_sigma_floor_standardized = float(
            state.get('calibration_sigma_floor_standardized', 1e-8)
        )
        modeler._calib_raw_quantile = float(
            state.get('calibration_raw_quantile', modeler._calib_z)
        )
        modeler._calib_unit = state.get('calibration_unit', 'legacy_point')
        modeler._calib_group_cols = list(state.get('calibration_group_cols', []))
        modeler._calibration_scores = np.asarray(
            state.get('calibration_scores', []), dtype=np.float64
        )
        modeler._calib_score_count = int(
            state.get('calibration_score_count', len(modeler._calibration_scores))
        )
        modeler.inference_method = state.get(
            'inference_method',
            modeler.bayes_config.get('inference_method', 'vi'),
        )
        modeler.hmc_acceptance_rate = float(state.get('hmc_acceptance_rate', np.nan))
        modeler.hmc_parameter_samples = state.get('hmc_parameter_samples', [])
        modeler.hmc_chain_parameter_samples = state.get('hmc_chain_parameter_samples', [])
        modeler.hmc_chain_acceptance_rates = state.get('hmc_chain_acceptance_rates', [])
        modeler.hmc_diagnostics = state.get('hmc_diagnostics', {})
        modeler.hmc_mass_matrix_diagnostics = state.get(
            'hmc_mass_matrix_diagnostics', {}
        )
        if state.get('global_aleatoric_var') is not None:
            modeler.global_aleatoric_var = float(state['global_aleatoric_var'])

        # 重建 scaler
        sd = state.get('scaler_data', {})
        modeler.scaler_x = StandardScaler()
        modeler.scaler_x.mean_ = sd['scaler_x_mean']
        modeler.scaler_x.scale_ = sd['scaler_x_scale']
        modeler.scaler_y = StandardScaler()
        modeler.scaler_y.mean_ = sd['scaler_y_mean']
        modeler.scaler_y.scale_ = sd['scaler_y_scale']

        # 重建模型
        modeler.build_model(modeler.bayes_config)
        modeler.model.load_state_dict(state['model_state'])
        modeler.model.to(device)
        modeler.model.eval()

        print(f"模型已加载: {path}")
        return modeler

    def predict_with_uncertainty(self, X):
        if self.scaler_x is not None:
            X_scaled = self.scaler_x.transform(X)
        else:
            X_scaled = X
            
        X_tensor = torch.from_numpy(X_scaled).float().to(self.device)
        
        with self._fixed_prediction_seed():
            with torch.no_grad():
                if self.inference_method == "hmc":
                    samples = self._predict_hmc_samples(X_tensor)
                else:
                    samples = None
                if samples is None:
                    samples, _, _ = self.model.forward_sample(
                        X_tensor,
                        num_samples=self.bayes_config.get('num_mc_samples', 100)
                    )
            
        samples_np = samples.cpu().numpy()
        
        mu_samples = samples_np[:, :, 0:1]      
        epistemic_var = np.var(mu_samples, axis=0)
        
        if samples_np.shape[-1] > 1:
            variance_logits = samples[:, :, 1:2]
            aleatoric_samples = variance_from_logits(variance_logits)
            aleatoric_var = np.mean(aleatoric_samples.cpu().numpy(), axis=0)
            logvar_samples = np.log(np.maximum(aleatoric_samples.cpu().numpy(), 1e-12))
        else:
            if not hasattr(self, 'global_aleatoric_var'):
                self._estimate_global_aleatoric_var()
            global_var = self.global_aleatoric_var
            aleatoric_var = np.full_like(epistemic_var, global_var)
            logvar_samples = np.full_like(mu_samples, np.log(max(global_var, 1e-8)))

        mean_np = np.mean(mu_samples, axis=0)
        total_var_np = epistemic_var + 1e-8 + aleatoric_var
        
        log_mean = self.scaler_y.inverse_transform(mean_np).reshape(-1)
        log_std = (np.sqrt(total_var_np) * self.data_stats['y_std']).reshape(-1)
        
        epistemic_unc = (np.sqrt(epistemic_var) * self.data_stats['y_std']).reshape(-1)
        aleatoric_unc = (np.sqrt(aleatoric_var) * self.data_stats['y_std']).reshape(-1)
        
        log_samples = self.scaler_y.inverse_transform(mu_samples.reshape(-1, 1)).reshape(mu_samples.shape)
        
        clipped_log_samples = np.clip(log_samples, -30.0, 30.0)
        linear_samples = np.power(10.0, clipped_log_samples) if self.use_log_transform else log_samples
        
        linear_mean = np.mean(linear_samples, axis=0).reshape(-1) 
        linear_std = np.std(linear_samples, axis=0).reshape(-1)
        
        z = float(getattr(self, '_calib_z', 1.96))
        standard_z = float(getattr(self, '_calib_raw_quantile', z))
        if not np.isfinite(standard_z):
            standard_z = z
        raw_z = 1.96
        conformal_floor = (
            0.0
            if getattr(self, '_calib_unit', 'uncalibrated') == 'uncalibrated'
            else float(getattr(self, '_calib_sigma_floor', 0.0))
        )
        conformal_log_std = np.maximum(log_std, conformal_floor)
        raw_log_lower = log_mean - raw_z * log_std
        raw_log_upper = log_mean + raw_z * log_std
        standard_log_lower = log_mean - standard_z * conformal_log_std
        standard_log_upper = log_mean + standard_z * conformal_log_std
        log_lower = log_mean - z * conformal_log_std
        log_upper = log_mean + z * conformal_log_std
        raw_linear_lower = np.power(10.0, np.clip(raw_log_lower, -30.0, 30.0)) if self.use_log_transform else raw_log_lower
        raw_linear_upper = np.power(10.0, np.clip(raw_log_upper, -30.0, 30.0)) if self.use_log_transform else raw_log_upper
        linear_lower = np.power(10.0, np.clip(log_lower, -30.0, 30.0)) if self.use_log_transform else log_lower
        linear_upper = np.power(10.0, np.clip(log_upper, -30.0, 30.0)) if self.use_log_transform else log_upper

        return {
            'mean': linear_mean, 'std': linear_std, 'lower': linear_lower, 'upper': linear_upper,
            'model_mean': log_mean, 'model_std': log_std,
            'model_lower': log_lower, 'model_upper': log_upper,
            'log_mean': log_mean, 'log_std': log_std, 'log_lower': log_lower, 'log_upper': log_upper,
            'raw_lower': raw_linear_lower, 'raw_upper': raw_linear_upper,
            'raw_model_lower': raw_log_lower, 'raw_model_upper': raw_log_upper,
            'raw_log_lower': raw_log_lower, 'raw_log_upper': raw_log_upper,
            'standard_conformal_model_lower': standard_log_lower,
            'standard_conformal_model_upper': standard_log_upper,
            'standard_conformal_log_lower': standard_log_lower,
            'standard_conformal_log_upper': standard_log_upper,
            'epistemic': epistemic_unc, 'aleatoric': aleatoric_unc,
            'samples': mu_samples, 'log_samples': log_samples, 'linear_samples': linear_samples,
        }

    def predict_derivative(self, X, diff_col):
        """B-spline 解析导数: dI/d(diff_col) 在物理空间。

        通过 autograd 精确计算 KAN B-spline 曲面对指定输入列的导数，
        绕过符号公式的近似误差。

        标准化链:
          dI/dX_raw = dI/d(log10_I) * d(log10_I)/d(y_norm)
                      * d(y_norm)/d(X_norm) * d(X_norm)/d(X_raw)
                    = ln(10) * I * std_y * model_deriv / std_X

        参数:
            X: (n_samples, n_inputs) — 原始物理单位
            diff_col: str — 求导的输入列名

        返回:
            dI_dX: (n_samples,) — 线性空间导数 dI/d(diff_col)
            dlogI_dX: (n_samples,) — log10 空间导数 d(log10_I)/d(diff_col)
        """
        from kan.bspline_derivative import kan_model_derivative

        diff_idx = self.input_cols.index(diff_col)
        X_scaled = self.scaler_x.transform(X.astype(np.float32))
        x_tensor = torch.from_numpy(X_scaled).float().to(self.device)

        model_deriv = kan_model_derivative(
            self.model, x_tensor, diff_idx, sample_mode=False
        )  # (n_samples,) — d(y_norm)/d(X_norm[:, diff_idx])

        # 标准化还原
        std_y = self.data_stats['y_std']
        std_x = self.data_stats['X_std'][diff_idx]

        dlogI_dX = std_y * model_deriv / std_x  # d(log10_I)/dX_raw

        # 预测 log10_I 用于链式法则 (仅取均值通道 channel 0)
        X_tensor = torch.from_numpy(X_scaled).float().to(self.device)
        with torch.no_grad():
            y_scaled, _ = self.model(X_tensor, sample=False)
        y_scaled_np = y_scaled[:, 0].cpu().numpy().reshape(-1)  # channel 0 = mean
        log10_I = self.scaler_y.inverse_transform(y_scaled_np.reshape(-1, 1)).reshape(-1)
        I_linear = np.power(10.0, log10_I)

        dI_dX = np.log(10) * I_linear * dlogI_dX  # dI/dX = ln(10) * I * d(log10_I)/dX

        return dI_dX, dlogI_dX

    def predict_derivative_with_uncertainty(self, X, diff_col, n_mc=100):
        """贝叶斯 B-spline 导数 + 不确定性。

        MC 采样后验权重 → 每组计算 B-spline 解析导数 → 统计均值±2σ。

        参数:
            X: (n_samples, n_inputs) — 原始物理单位
            diff_col: str — 求导的输入列名
            n_mc: MC 采样次数 (默认 100)

        返回:
            dict: {
                'dI_dX_mean': (n_samples,), 'dI_dX_std': (n_samples,),
                'dI_dX_lower': (n_samples,), 'dI_dX_upper': (n_samples,),
                'dlogI_dX_mean': (n_samples,), 'dlogI_dX_std': (n_samples,),
                'dlogI_dX_samples': (n_mc, n_samples),
            }
        """
        from kan.bspline_derivative import kan_model_derivative_bayesian

        diff_idx = self.input_cols.index(diff_col)
        X_scaled = self.scaler_x.transform(X.astype(np.float32))
        x_tensor = torch.from_numpy(X_scaled).float().to(self.device)

        # 贝叶斯导数 (在标准化空间)
        bayes_result = kan_model_derivative_bayesian(
            self.model, x_tensor, diff_idx, n_samples=n_mc
        )

        # 标准化还原
        std_y = self.data_stats['y_std']
        std_x = self.data_stats['X_std'][diff_idx]

        dlogI_dX_samples = std_y * bayes_result['samples'] / std_x  # (n_mc, n_samples)
        dlogI_dX_mean = dlogI_dX_samples.mean(axis=0)
        dlogI_dX_std = dlogI_dX_samples.std(axis=0, ddof=1)

        # 线性空间导数需要 I 值 (仅取均值通道 channel 0)
        X_tensor = torch.from_numpy(X_scaled).float().to(self.device)
        with torch.no_grad():
            y_scaled, _ = self.model(X_tensor, sample=False)
        y_scaled_np = y_scaled[:, 0].cpu().numpy().reshape(-1)  # channel 0 = mean
        log10_I = self.scaler_y.inverse_transform(y_scaled_np.reshape(-1, 1)).reshape(-1)
        I_linear = np.power(10.0, log10_I)
        ln10 = np.log(10)

        dI_dX_samples = ln10 * I_linear[None, :] * dlogI_dX_samples
        dI_dX_mean = dI_dX_samples.mean(axis=0)
        dI_dX_std = dI_dX_samples.std(axis=0, ddof=1)

        return {
            'dI_dX_mean': dI_dX_mean,
            'dI_dX_std': dI_dX_std,
            'dI_dX_lower': dI_dX_mean - 2 * dI_dX_std,
            'dI_dX_upper': dI_dX_mean + 2 * dI_dX_std,
            'dlogI_dX_mean': dlogI_dX_mean,
            'dlogI_dX_std': dlogI_dX_std,
            'dlogI_dX_lower': dlogI_dX_mean - 2 * dlogI_dX_std,
            'dlogI_dX_upper': dlogI_dX_mean + 2 * dlogI_dX_std,
            'dlogI_dX_samples': dlogI_dX_samples,
        }

    def evaluate_on_test_set(self, df_test):
        if df_test is None:
            raise ValueError("请传入 df_test 进行物理空间评估")
            
        X_test = df_test[self.input_cols].values
        y_test_raw = df_test[self.output_col].values
        y_test_transformed = self._transform_target(y_test_raw).flatten()
        
        valid_test = (y_test_transformed > self.y_bounds[0]) & (y_test_transformed < self.y_bounds[1])
        X_test = X_test[valid_test]
        y_test_raw = y_test_raw[valid_test]
        y_test_transformed = y_test_transformed[valid_test]
        test_frame = df_test.loc[valid_test].reset_index(drop=True)

        pred = self.predict_with_uncertainty(X_test)
        
        mae_log = np.mean(np.abs(pred['log_mean'] - y_test_transformed))
        rmse_log = np.sqrt(np.mean((pred['log_mean'] - y_test_transformed)**2))
        mae_linear = np.mean(np.abs(pred['mean'] - y_test_raw.flatten()))
        residual = pred['log_mean'] - y_test_transformed
        ss_res = np.sum(residual ** 2)
        ss_tot = np.sum((y_test_transformed - np.mean(y_test_transformed)) ** 2)
        r2_model_space = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan
        
        raw_in_ci = (
            (y_test_transformed >= pred['raw_log_lower'])
            & (y_test_transformed <= pred['raw_log_upper'])
        )
        standard_in_ci = (
            (y_test_transformed >= pred['standard_conformal_log_lower'])
            & (y_test_transformed <= pred['standard_conformal_log_upper'])
        )
        no_shrink_in_ci = (
            (y_test_transformed >= pred['log_lower'])
            & (y_test_transformed <= pred['log_upper'])
        )
        raw_picp = np.mean(raw_in_ci) * 100.0
        standard_picp = np.mean(standard_in_ci) * 100.0
        no_shrink_picp = np.mean(no_shrink_in_ci) * 100.0
        raw_mpiw = np.mean(pred['raw_log_upper'] - pred['raw_log_lower'])
        standard_mpiw = np.mean(
            pred['standard_conformal_log_upper']
            - pred['standard_conformal_log_lower']
        )
        no_shrink_mpiw = np.mean(pred['log_upper'] - pred['log_lower'])
        picp = no_shrink_picp
        mpiw = no_shrink_mpiw
        mean_std_log = np.mean(pred['log_std'])

        # Proper scoring rules are evaluated in standardized target space so
        # they are invariant to whether a task is measured in dB, log-current,
        # or capacitance values around 1e-17 F.
        target_scale = max(
            abs(float(self.scaler_y.scale_[0])),
            np.finfo(np.float64).tiny,
        )
        residual_standardized = residual / target_scale
        predicted_std = np.asarray(pred['log_std'], dtype=np.float64)
        predicted_std_standardized = predicted_std / target_scale
        score_floor_standardized = float(
            getattr(self, '_calib_sigma_floor_standardized', 1e-8)
        )
        raw_std_standardized = np.maximum(
            predicted_std_standardized,
            score_floor_standardized,
        )
        standard_z = float(
            getattr(
                self,
                '_calib_raw_quantile',
                getattr(self, '_calib_z', 1.96),
            )
        )
        if not np.isfinite(standard_z):
            standard_z = float(getattr(self, '_calib_z', 1.96))
        standard_std_standardized = np.maximum(
            raw_std_standardized * standard_z / 1.96,
            score_floor_standardized,
        )
        no_shrink_std_standardized = np.maximum(
            raw_std_standardized
            * float(getattr(self, '_calib_z', 1.96))
            / 1.96,
            score_floor_standardized,
        )
        raw_gaussian_nll = float(
            np.mean(
                0.5 * np.log(2.0 * np.pi * raw_std_standardized ** 2)
                + 0.5 * residual_standardized ** 2 / raw_std_standardized ** 2
            )
        )
        standard_gaussian_nll = float(
            np.mean(
                0.5 * np.log(2.0 * np.pi * standard_std_standardized ** 2)
                + 0.5
                * residual_standardized ** 2
                / standard_std_standardized ** 2
            )
        )
        no_shrink_gaussian_nll = float(
            np.mean(
                0.5 * np.log(2.0 * np.pi * no_shrink_std_standardized ** 2)
                + 0.5
                * residual_standardized ** 2
                / no_shrink_std_standardized ** 2
            )
        )
        alpha = 0.05

        def interval_score(lower, upper):
            width = upper - lower
            below = np.maximum(lower - y_test_transformed, 0.0)
            above = np.maximum(y_test_transformed - upper, 0.0)
            return float(np.mean(width + 2.0 * (below + above) / alpha))

        raw_interval_score = interval_score(
            pred['raw_log_lower'],
            pred['raw_log_upper'],
        )
        standard_interval_score = interval_score(
            pred['standard_conformal_log_lower'],
            pred['standard_conformal_log_upper'],
        )
        no_shrink_interval_score = interval_score(
            pred['log_lower'],
            pred['log_upper'],
        )

        def grouped_coverage(columns):
            columns = [column for column in columns if column in test_frame.columns]
            if not columns:
                return (
                    np.nan,
                    np.nan,
                    np.nan,
                    np.full(len(test_frame), np.nan),
                    np.full(len(test_frame), np.nan),
                    np.full(len(test_frame), np.nan),
                    0,
                )
            coverage_frame = test_frame[columns].copy()
            coverage_frame["_raw_covered"] = raw_in_ci
            coverage_frame["_standard_covered"] = standard_in_ci
            coverage_frame["_no_shrink_covered"] = no_shrink_in_ci
            grouped = coverage_frame.groupby(columns, dropna=False).agg(
                raw_covered=("_raw_covered", "all"),
                standard_covered=("_standard_covered", "all"),
                no_shrink_covered=("_no_shrink_covered", "all"),
            )
            raw_flags = (
                coverage_frame.groupby(columns, dropna=False)["_raw_covered"]
                .transform("all")
                .to_numpy(dtype=bool)
            )
            standard_flags = (
                coverage_frame.groupby(columns, dropna=False)["_standard_covered"]
                .transform("all")
                .to_numpy(dtype=bool)
            )
            no_shrink_flags = (
                coverage_frame.groupby(columns, dropna=False)["_no_shrink_covered"]
                .transform("all")
                .to_numpy(dtype=bool)
            )
            return (
                float(grouped["raw_covered"].mean() * 100.0),
                float(grouped["standard_covered"].mean() * 100.0),
                float(grouped["no_shrink_covered"].mean() * 100.0),
                raw_flags,
                standard_flags,
                no_shrink_flags,
                int(len(grouped)),
            )

        group_cols = [
            col for col in getattr(self, '_calib_group_cols', [])
            if col in test_frame.columns
        ]
        (
            raw_group_coverage,
            standard_group_coverage,
            no_shrink_group_coverage,
            raw_group_flags,
            standard_group_flags,
            no_shrink_group_flags,
            test_group_count,
        ) = grouped_coverage(group_cols)

        # For capacitance, retain the nested C--V sub-curve diagnostic
        # (TCAD x frequency) while keeping TCAD as the conformal unit.
        subcurve_cols = []
        if "_curve_id" in test_frame.columns and "log_frequency_ghz" in test_frame.columns:
            subcurve_cols = ["_curve_id", "log_frequency_ghz"]
        (
            raw_subcurve_coverage,
            standard_subcurve_coverage,
            no_shrink_subcurve_coverage,
            raw_subcurve_flags,
            standard_subcurve_flags,
            no_shrink_subcurve_flags,
            test_subcurve_count,
        ) = grouped_coverage(subcurve_cols)

        # Backward-compatible curve fields now refer to the independent grouped
        # split unit.  New group/device/subcurve fields make the hierarchy explicit.
        raw_curve_coverage = raw_group_coverage
        calibrated_curve_coverage = no_shrink_group_coverage
        raw_curve_flags = raw_group_flags
        calibrated_curve_flags = no_shrink_group_flags

        prediction_frame = test_frame.copy()
        prediction_frame["actual_model_space"] = y_test_transformed
        prediction_frame["prediction_mean_model_space"] = pred['log_mean']
        prediction_frame["prediction_std_model_space"] = predicted_std
        prediction_frame["raw_lower_model_space"] = pred['raw_log_lower']
        prediction_frame["raw_upper_model_space"] = pred['raw_log_upper']
        prediction_frame["standard_conformal_lower_model_space"] = (
            pred['standard_conformal_log_lower']
        )
        prediction_frame["standard_conformal_upper_model_space"] = (
            pred['standard_conformal_log_upper']
        )
        prediction_frame["calibrated_lower_model_space"] = pred['log_lower']
        prediction_frame["calibrated_upper_model_space"] = pred['log_upper']
        prediction_frame["raw_point_covered"] = raw_in_ci
        prediction_frame["standard_conformal_point_covered"] = standard_in_ci
        prediction_frame["no_shrink_point_covered"] = no_shrink_in_ci
        prediction_frame["calibrated_point_covered"] = no_shrink_in_ci
        prediction_frame["raw_curve_covered"] = raw_curve_flags
        prediction_frame["calibrated_curve_covered"] = calibrated_curve_flags
        prediction_frame["raw_group_covered"] = raw_group_flags
        prediction_frame["standard_conformal_group_covered"] = standard_group_flags
        prediction_frame["no_shrink_group_covered"] = no_shrink_group_flags
        prediction_frame["calibrated_group_covered"] = no_shrink_group_flags
        prediction_frame["raw_subcurve_covered"] = raw_subcurve_flags
        prediction_frame["standard_conformal_subcurve_covered"] = (
            standard_subcurve_flags
        )
        prediction_frame["no_shrink_subcurve_covered"] = no_shrink_subcurve_flags
        prediction_frame["calibrated_subcurve_covered"] = no_shrink_subcurve_flags
        prediction_frame.to_csv(
            os.path.join(self.save_dir, "uq_test_predictions.csv"),
            index=False,
        )

        return {
            'MAE_Log': mae_log, 'RMSE_Log': rmse_log, 'MAE_Linear': mae_linear,
            'MAE_Model_Space': mae_log, 'RMSE_Model_Space': rmse_log,
            'R2_Model_Space': r2_model_space,
            'PICP_95': picp, 'MPIW_Log': mpiw, 'Mean_Std_Log': mean_std_log,
            'Raw_PICP_95': raw_picp, 'Calibrated_PICP_95': picp,
            'Raw_MPIW_Log': raw_mpiw, 'Calibrated_MPIW_Log': mpiw,
            'Standard_Conformal_PICP_95': standard_picp,
            'No_Shrink_Conformal_PICP_95': no_shrink_picp,
            'Standard_Conformal_MPIW_Log': standard_mpiw,
            'No_Shrink_Conformal_MPIW_Log': no_shrink_mpiw,
            'Raw_Gaussian_NLL_Model_Space': raw_gaussian_nll,
            'Standard_Conformal_Gaussian_NLL_Model_Space': standard_gaussian_nll,
            'No_Shrink_Conformal_Gaussian_NLL_Model_Space': no_shrink_gaussian_nll,
            'Calibrated_Gaussian_NLL_Model_Space': no_shrink_gaussian_nll,
            'Raw_Interval_Score_95': raw_interval_score,
            'Standard_Conformal_Interval_Score_95': standard_interval_score,
            'No_Shrink_Conformal_Interval_Score_95': no_shrink_interval_score,
            'Calibrated_Interval_Score_95': no_shrink_interval_score,
            'Raw_Curve_Coverage_95': raw_curve_coverage,
            'Standard_Conformal_Curve_Coverage_95': standard_group_coverage,
            'No_Shrink_Conformal_Curve_Coverage_95': no_shrink_group_coverage,
            'Calibrated_Curve_Coverage_95': calibrated_curve_coverage,
            'Raw_Group_Coverage_95': raw_group_coverage,
            'Standard_Conformal_Group_Coverage_95': standard_group_coverage,
            'No_Shrink_Conformal_Group_Coverage_95': no_shrink_group_coverage,
            'Calibrated_Group_Coverage_95': no_shrink_group_coverage,
            'Raw_Device_Coverage_95': (
                raw_group_coverage if group_cols == ["_curve_id"] else np.nan
            ),
            'Standard_Conformal_Device_Coverage_95': (
                standard_group_coverage
                if group_cols == ["_curve_id"] else np.nan
            ),
            'No_Shrink_Conformal_Device_Coverage_95': (
                no_shrink_group_coverage
                if group_cols == ["_curve_id"] else np.nan
            ),
            'Calibrated_Device_Coverage_95': (
                no_shrink_group_coverage
                if group_cols == ["_curve_id"] else np.nan
            ),
            'Raw_Subcurve_Coverage_95': raw_subcurve_coverage,
            'Standard_Conformal_Subcurve_Coverage_95': standard_subcurve_coverage,
            'No_Shrink_Conformal_Subcurve_Coverage_95': no_shrink_subcurve_coverage,
            'Calibrated_Subcurve_Coverage_95': no_shrink_subcurve_coverage,
            'Test_Group_Count': test_group_count,
            'Test_Device_Count': (
                test_group_count if group_cols == ["_curve_id"] else 0
            ),
            'Test_Subcurve_Count': test_subcurve_count,
            'Test_Point_Count': int(len(test_frame)),
            'MPIW_Model_Space': mpiw, 'Mean_Std_Model_Space': mean_std_log,
            'Calibration_Z': getattr(self, '_calib_z', 1.96),
            'Calibration_Raw_Quantile': getattr(self, '_calib_raw_quantile', np.nan),
            'Raw_Z': 1.96,
            'Standard_Conformal_Z': standard_z,
            'No_Shrink_Conformal_Z': getattr(self, '_calib_z', 1.96),
            'Calibration_Std_Scale_Vs_Normal': getattr(self, '_calib_z', 1.96) / 1.96,
            'Calibration_Unit': getattr(self, '_calib_unit', 'unknown'),
            'Calibration_Score_Count': getattr(self, '_calib_score_count', 0),
            'Calibration_Group_Columns': ','.join(
                getattr(self, '_calib_group_cols', [])
            ),
            'Calibration_Sigma_Floor_Standardized': getattr(
                self, '_calib_sigma_floor_standardized', np.nan
            ),
            'Calibration_Sigma_Floor_Model_Space': getattr(
                self, '_calib_sigma_floor', np.nan
            ),
        }
        
    def ood_detection(self, X_test, X_train=None, threshold=None):
        if X_train is not None:
            X_train_scaled = self.scaler_x.transform(X_train)
            X_train_tensor = torch.from_numpy(X_train_scaled).float().to(self.device)
            with torch.no_grad():
                train_samples, _, _ = self.model.forward_sample(X_train_tensor, num_samples=self.bayes_config.get('num_mc_samples', 100))
            train_entropy = predictive_entropy(train_samples).cpu().numpy()
            threshold = np.percentile(train_entropy, 95)
        
        X_test_scaled = self.scaler_x.transform(X_test)
        X_test_tensor = torch.from_numpy(X_test_scaled).float().to(self.device)
        with torch.no_grad():
            test_samples, _, _ = self.model.forward_sample(X_test_tensor, num_samples=self.bayes_config.get('num_mc_samples', 100))
        
        test_entropy = predictive_entropy(test_samples).cpu().numpy()
        if threshold is None:
            threshold = np.mean(test_entropy) + 2 * np.std(test_entropy)
        
        ood_flags = test_entropy > threshold
        return {
            'entropy': test_entropy, 'ood_flags': ood_flags,
            'threshold': threshold, 'ood_ratio': np.mean(ood_flags),
        }


def _delegate_fit_ac_response_compact_formula(self, verbose=True):
    from device_modeling.photodetector.bayesian_symbolic import _fit_ac_response_compact_formula

    return _fit_ac_response_compact_formula(self, verbose=verbose)


def _delegate_extract_symbolic_formula(
    self,
    weight_simple=0.8,
    r2_threshold=0.0,
    verbose=1,
    simplify=True,
):
    from device_modeling.photodetector.bayesian_symbolic import extract_symbolic_formula

    return extract_symbolic_formula(
        self,
        weight_simple=weight_simple,
        r2_threshold=r2_threshold,
        verbose=verbose,
        simplify=simplify,
    )


def _delegate_plot_formula_verification(self, df_test=None, voltage_col=None, df_curves=None):
    from device_modeling.photodetector.bayesian_symbolic import plot_formula_verification

    return plot_formula_verification(
        self,
        df_test=df_test,
        voltage_col=voltage_col,
        df_curves=df_curves,
    )


def _delegate_plot_loss_curve(self):
    from device_modeling.photodetector.diagnostics import plot_loss_curve

    return plot_loss_curve(self)


# Keep the historical BayesKANDeviceModeler methods available while making the
# split-out modules the single maintenance point for symbolic extraction and
# diagnostics.
BayesKANDeviceModeler._fit_ac_response_compact_formula = _delegate_fit_ac_response_compact_formula
BayesKANDeviceModeler.extract_symbolic_formula = _delegate_extract_symbolic_formula
BayesKANDeviceModeler.plot_formula_verification = _delegate_plot_formula_verification
BayesKANDeviceModeler.plot_loss_curve = _delegate_plot_loss_curve
