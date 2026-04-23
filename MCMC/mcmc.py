from sklearn.metrics import roc_auc_score
import math
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


class BayesianGLM:

    def __init__(self,
                 link=None,
                 inverse_link=None,
                 beta_prior=None,
                 burn_in=100,
                 num_samples=1000,
                 num_chains=1,
                 step_size=0.05,
                 seed=0,
                 device=None,
                 **kwargs):

        # funciones de link
        self.link = link
        self.inverse_link = inverse_link if inverse_link is not None else (lambda x: x)

        # prior
        self.beta_prior = beta_prior

        # MCMC params
        self.burn_in = burn_in
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.step_size = step_size

        # device
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # seed
        torch.manual_seed(seed)

        # contenedor de muestras
        self.samples = None

        # kwargs dinámicos
        for k, v in kwargs.items():
            setattr(self, k, v)

    # =========================================================
    # ABSTRACT METHODS
    # =========================================================

    def log_posterior(self, X, y, params):
        raise NotImplementedError

    def init_params(self, p):
        raise NotImplementedError

    def sample_likelihood(self, mu, params):
        raise NotImplementedError

    def neg_log_likelihood(self, X, y, params):
        raise NotImplementedError

    # =========================================================
    # FIT (Metropolis-Hastings)
    # =========================================================

    def fit(self, X, y):

        X = X.to(self.device)
        y = y.to(self.device)

        n, p = X.shape

        params = self.init_params(p)

        samples = {
            k: torch.zeros((self.num_samples, self.num_chains, *v.shape[1:]),
                           device=self.device)
            for k, v in params.items()
        }

        total_iter = self.burn_in + self.num_samples

        for i in range(total_iter):

            # propuesta
            props = {
                k: v + self.step_size * torch.randn_like(v)
                for k, v in params.items()
            }

            logp_current = self.log_posterior(X, y, params)
            logp_prop = self.log_posterior(X, y, props)

            log_alpha = logp_prop - logp_current
            accept = torch.log(torch.rand_like(log_alpha)) < log_alpha

            # update vectorizado por cadenas
            for k in params:
                if params[k].dim() == 2:
                    params[k] = torch.where(accept[:, None], props[k], params[k])
                else:
                    params[k] = torch.where(accept, props[k], params[k])

            # guardar después de burn-in
            if i >= self.burn_in:
                idx = i - self.burn_in
                for k in samples:
                    samples[k][idx] = params[k]

        self.samples = samples

    # =========================================================
    # POSTERIOR SAMPLES
    # =========================================================

    def get_params_samples(self):
        return self.samples

    def get_beta_samples(self):
        return self.samples["beta"]

    def get_beta_hat(self):
        return self.get_beta_samples().mean(dim=(0, 1))

    def get_plugin_params(self):
        return {
            k: v.mean(dim=(0, 1))
            for k, v in self.samples.items()
        }

    # =========================================================
    # SUMMARY
    # =========================================================

    def summary(self):

        if self.samples is None:
            raise ValueError("Debes llamar a fit() antes.")

        print("\nPosterior Summary")
        print("=" * 50)

        for name, values in self.samples.items():

            flat = values.reshape(-1, *values.shape[2:])

            mean = flat.mean(dim=0)
            std = flat.std(dim=0)

            print(f"\n{name}:")
            print(f"mean = {mean}")
            print(f"std  = {std}")

    # =========================================================
    # PREDICTIONS
    # =========================================================

    def predict_plugin(self, X):

        X = X.to(self.device)
        beta = self.get_beta_hat()

        eta = X @ beta
        return self.inverse_link(eta)

    def predict_posterior_samples(self, X):

        X = X.to(self.device)

        beta = self.get_beta_samples()  # (S, C, d)

        eta = torch.einsum("nd,scd->scn", X, beta)

        return self.inverse_link(eta)

    def predict_posterior_mean(self, X):

        y = self.predict_posterior_samples(X)
        return y.mean(dim=(0, 1))

    def predict_credible_interval(self, X, alpha=0.05):

        y = self.predict_posterior_samples(X)
        y_flat = y.reshape(-1, X.shape[0])

        lower = torch.quantile(y_flat, alpha / 2, dim=0)
        upper = torch.quantile(y_flat, 1 - alpha / 2, dim=0)

        return lower, upper

    def forward(self, X):
        return self.predict_posterior_samples(X)

    # =========================================================
    # POSTERIOR PREDICTIVE
    # =========================================================

    def sample_posterior_predictive(self, X):

        params = self.get_params_samples()
        mu = self.predict_posterior_samples(X)

        return self.sample_likelihood(mu, params)

    # =========================================================
    # NLL
    # =========================================================

    def normalized_neg_log_likelihood_plugin(self, X, y):

        params = self.get_plugin_params()
        return self.neg_log_likelihood(X, y, params) / X.shape[0]

    def normalized_neg_log_likelihood_samples(self, X, y):

        params = self.get_params_samples()

        values = []

        for i in range(self.num_samples):
            for c in range(self.num_chains):

                p = {k: v[i, c] for k, v in params.items()}
                values.append(self.neg_log_likelihood(X, y, p))

        return torch.stack(values) / X.shape[0]

    # =========================================================
    # METRICS
    # =========================================================

    def summary_metrics(self, X, y, return_dict=False):

        nnll = self.normalized_neg_log_likelihood_samples(X, y)

        stats = {
            "mean": nnll.mean(),
            "std": nnll.std(),
            "median": nnll.median(),
            "5%": torch.quantile(nnll, 0.05),
            "95%": torch.quantile(nnll, 0.95),
            "plugin": self.normalized_neg_log_likelihood_plugin(X, y)
        }

        if return_dict:
            return stats

        print("\nSummary Normalized Negative Log-Likelihood")
        print("-" * 50)

        for k, v in stats.items():
            print(f"{k:10}: {v:.4f}")

    # =========================================================
    # VISUALIZATIONS
    # =========================================================

    def plot_nll_density(self, X, y, figsize=(8, 4)):

        nnll = self.normalized_neg_log_likelihood_samples(X, y).cpu().numpy()

        kde = gaussian_kde(nnll)
        x_grid = np.linspace(nnll.min(), nnll.max(), 200)

        plt.figure(figsize=figsize)
        plt.plot(x_grid, kde(x_grid))
        plt.title("Normalized Negative Log-Likelihood")
        plt.xlabel("NLL")
        plt.ylabel("Density")
        plt.show()

class BayesianLinearRegression(BayesianGLM):

    def __init__(self,
                 sigma_prior=None,
                 **kwargs):

        super().__init__(**kwargs)

        # prior para sigma > 0
        self.sigma_prior = sigma_prior if sigma_prior is not None else torch.distributions.HalfNormal(1.0)

    # =========================================================
    # INIT PARAMS (para MH)
    # =========================================================

    def init_params(self, p):

        return {
            "beta": torch.zeros((self.num_chains, p), device=self.device),
            "log_sigma": torch.zeros(self.num_chains, device=self.device)  # trabajamos en log
        }

    # =========================================================
    # LOG POSTERIOR
    # =========================================================

    def log_posterior(self, X, y, params):

        beta = params["beta"]                  # (C, p)
        log_sigma = params["log_sigma"]        # (C,)
        sigma = torch.exp(log_sigma)           # (C,)

        # likelihood
        mu = X @ beta.T                        # (n, C)

        dist = torch.distributions.Normal(mu, sigma)
        log_like = dist.log_prob(y[:, None]).sum(0)  # (C,)

        # prior beta (Normal isotrópico si no se definió)
        if self.beta_prior is None:
            log_prior_beta = -0.5 * (beta ** 2).sum(dim=1)
        else:
            log_prior_beta = self.beta_prior.log_prob(beta).sum(dim=1)

        # prior sigma
        log_prior_sigma = self.sigma_prior.log_prob(sigma)

        # jacobiano (por cambio log -> sigma)
        return log_like + log_prior_beta + log_prior_sigma + log_sigma

    # =========================================================
    # NEG LOG LIKELIHOOD
    # =========================================================

    def neg_log_likelihood(self, X, y, params):

        beta = params["beta"]
        sigma = params.get("sigma", None)

        if sigma is None:
            sigma = torch.exp(params["log_sigma"])

        mu = X @ beta

        dist = torch.distributions.Normal(mu, sigma)
        return -dist.log_prob(y).sum()

    # =========================================================
    # POSTERIOR PREDICTIVE
    # =========================================================

    def sample_likelihood(self, mu, params):

        # mu: (S, C, n)
        log_sigma = params["log_sigma"]        # (S, C)
        sigma = torch.exp(log_sigma)           # (S, C)

        sigma = sigma[:, :, None]              # (S, C, 1)

        return mu + sigma * torch.randn_like(mu)

    # =========================================================
    # SUMMARY METRICS
    # =========================================================

    def summary_metrics(self, X, y, return_data=False):

        if self.samples is None:
            raise ValueError("Debes llamar a fit() antes.")

        X = X.to(self.device)
        y = y.to(self.device)

        # =====================================================
        # PREDICCIONES
        # =====================================================
        y_samples = self.predict_posterior_samples(X)   # (S, C, n)
        y_plugin = self.predict_plugin(X)               # (n,)

        # =====================================================
        # NNLL
        # =====================================================
        nnll = self.normalized_neg_log_likelihood_samples(X, y)
        nnll_plugin = self.normalized_neg_log_likelihood_plugin(X, y)

        # =====================================================
        # MSE / MAE (posterior)
        # =====================================================
        y_expanded = y[None, None, :]  # (1,1,n)

        mse_samples = ((y_samples - y_expanded) ** 2).mean(dim=2)  # (S,C)
        mae_samples = torch.abs(y_samples - y_expanded).mean(dim=2)

        # flatten
        mse_samples = mse_samples.reshape(-1)
        mae_samples = mae_samples.reshape(-1)

        # =====================================================
        # MSE / MAE (plugin)
        # =====================================================
        mse_plugin = ((y_plugin - y) ** 2).mean()
        mae_plugin = torch.abs(y_plugin - y).mean()

        # =====================================================
        # TABLA
        # =====================================================
        data = {
            "NNLL": [
                nnll.mean(),
                nnll.std(),
                nnll.median(),
                torch.quantile(nnll, 0.05),
                torch.quantile(nnll, 0.95),
                nnll_plugin,
            ],
            "MSE": [
                mse_samples.mean(),
                mse_samples.std(),
                mse_samples.median(),
                torch.quantile(mse_samples, 0.05),
                torch.quantile(mse_samples, 0.95),
                mse_plugin,
            ],
            "MAE": [
                mae_samples.mean(),
                mae_samples.std(),
                mae_samples.median(),
                torch.quantile(mae_samples, 0.05),
                torch.quantile(mae_samples, 0.95),
                mae_plugin,
            ],
        }

        columns = ["mean", "std", "median", "5%", "95%", "plugin"]

        df = pd.DataFrame(
            {k: [v_i.item() for v_i in v] for k, v in data.items()},
            index=columns
        ).T


        return df




class BayesianLaplaceRegression(BayesianGLM):

    def __init__(self,
                 b_prior=None,
                 **kwargs):

        super().__init__(**kwargs)

        # prior para b > 0
        self.b_prior = b_prior if b_prior is not None else torch.distributions.HalfNormal(1.0)

    # =========================================================
    # INIT PARAMS
    # =========================================================

    def init_params(self, p):

        return {
            "beta": torch.zeros((self.num_chains, p), device=self.device),
            "log_b": torch.zeros(self.num_chains, device=self.device)
        }

    # =========================================================
    # LOG POSTERIOR
    # =========================================================

    def log_posterior(self, X, y, params):

        beta = params["beta"]            # (C, p)
        log_b = params["log_b"]          # (C,)
        b = torch.exp(log_b)             # (C,)

        # likelihood
        mu = X @ beta.T                 # (n, C)

        dist = torch.distributions.Laplace(mu, b)
        log_like = dist.log_prob(y[:, None]).sum(0)  # (C,)

        # prior beta
        if self.beta_prior is None:
            log_prior_beta = -0.5 * (beta ** 2).sum(dim=1)
        else:
            log_prior_beta = self.beta_prior.log_prob(beta).sum(dim=1)

        # prior b
        log_prior_b = self.b_prior.log_prob(b)

        # jacobiano
        return log_like + log_prior_beta + log_prior_b + log_b

    # =========================================================
    # NEG LOG LIKELIHOOD
    # =========================================================

    def neg_log_likelihood(self, X, y, params):

        beta = params["beta"]

        if "b" in params:
            b = params["b"]
        else:
            b = torch.exp(params["log_b"])

        mu = X @ beta

        dist = torch.distributions.Laplace(mu, b)
        return -dist.log_prob(y).sum()

    # =========================================================
    # POSTERIOR PREDICTIVE
    # =========================================================

    def sample_likelihood(self, mu, params):

        # mu: (S, C, n)
        log_b = params["log_b"]          # (S, C)
        b = torch.exp(log_b)             # (S, C)

        b = b[:, :, None]                # (S, C, 1)

        dist = torch.distributions.Laplace(mu, b)
        return dist.sample()

    # =========================================================
    # PREDICT (PLUGIN)
    # =========================================================

    def predict_plugin(self, X):

        X = X.to(self.device)
        beta = self.get_beta_hat()

        return X @ beta

    # =========================================================
    # PREDICT (POSTERIOR)
    # =========================================================

    def predict_posterior_samples(self, X):

        X = X.to(self.device)
        beta = self.get_beta_samples()  # (S, C, d)

        return torch.einsum("nd,scd->scn", X, beta)

    # =========================================================
    # SUMMARY METRICS
    # =========================================================

    def summary_metrics(self, X, y):

        if self.samples is None:
            raise ValueError("Debes llamar a fit() antes.")

        X = X.to(self.device)
        y = y.to(self.device)

        # =====================================================
        # PREDICCIONES
        # =====================================================
        y_samples = self.predict_posterior_samples(X)  # (S,C,n)
        y_plugin = self.predict_plugin(X)              # (n,)

        # =====================================================
        # NNLL
        # =====================================================
        nnll = self.normalized_neg_log_likelihood_samples(X, y)
        nnll_plugin = self.normalized_neg_log_likelihood_plugin(X, y)

        # =====================================================
        # MSE / MAE
        # =====================================================
        y_expanded = y[None, None, :]

        mse_samples = ((y_samples - y_expanded) ** 2).mean(dim=2)
        mae_samples = torch.abs(y_samples - y_expanded).mean(dim=2)

        mse_samples = mse_samples.reshape(-1)
        mae_samples = mae_samples.reshape(-1)

        mse_plugin = ((y_plugin - y) ** 2).mean()
        mae_plugin = torch.abs(y_plugin - y).mean()

        # =====================================================
        # TABLA
        # =====================================================
        data = {
            "NNLL": [
                nnll.mean(),
                nnll.std(),
                nnll.median(),
                torch.quantile(nnll, 0.05),
                torch.quantile(nnll, 0.95),
                nnll_plugin,
            ],
            "MSE": [
                mse_samples.mean(),
                mse_samples.std(),
                mse_samples.median(),
                torch.quantile(mse_samples, 0.05),
                torch.quantile(mse_samples, 0.95),
                mse_plugin,
            ],
            "MAE": [
                mae_samples.mean(),
                mae_samples.std(),
                mae_samples.median(),
                torch.quantile(mae_samples, 0.05),
                torch.quantile(mae_samples, 0.95),
                mae_plugin,
            ],
        }

        columns = ["mean", "std", "median", "5%", "95%", "plugin"]

        df = pd.DataFrame(
            {k: [v_i.item() for v_i in v] for k, v in data.items()},
            index=columns
        ).T

        return df

class BayesianBinaryGLM(BayesianGLM):

    def __init__(self,
                 inverse_link=None,
                 **kwargs):

        super().__init__(inverse_link=inverse_link, **kwargs)

    # =========================================================
    # LOG POSTERIOR
    # =========================================================
    def log_posterior(self, beta):

        z = self.X @ beta.T
        mu = self.inverse_link(z)

        log_like = torch.distributions.Bernoulli(probs=mu).log_prob(
            self.y[:, None]
        ).sum(0)

        log_prior = self.beta_prior.log_prob(beta)

        return log_like + log_prior

    # =========================================================
    # INIT PARAMS
    # =========================================================
    def init_params(self, n_chains):

        return {
            "beta": torch.zeros((n_chains, self.d))
        }

    # =========================================================
    # STORE SAMPLES
    # =========================================================
    def _store_samples(self, samples):

        self.beta_samples = samples["beta"]
        self.beta = self.beta_samples.mean(dim=(0, 1))

    # =========================================================
    # PREDICTIONS
    # =========================================================
    def predict_proba_samples(self, X):

        z = self.beta_samples @ X.T
        return self.inverse_link(z)

    def predict_samples(self, X, threshold=0.5):

        probs = self.predict_proba_samples(X)
        return (probs >= threshold).float()

    def predict_plugin(self, X, threshold=0.5):

        z = X @ self.beta
        probs = self.inverse_link(z)

        return (probs >= threshold).float()

    # =========================================================
    # METRICS
    # =========================================================
    def accuracy_samples(self, X, y, threshold=0.5):

        y_pred = self.predict_samples(X, threshold)
        return (y[None, None, :] == y_pred).float().mean(dim=2)

    def accuracy_plugin(self, X, y, threshold=0.5):

        y_pred = self.predict_plugin(X, threshold)
        return (y == y_pred).float().mean()

    def auc_samples(self, X, y):

        probs = self.predict_proba_samples(X)
        probs_flat = probs.reshape(-1, probs.shape[-1])

        y_np = y.numpy()

        auc_list = [
            roc_auc_score(y_np, p.numpy())
            for p in probs_flat
        ]

        return torch.tensor(auc_list).reshape(
            probs.shape[0], probs.shape[1]
        )

    def auc_plugin(self, X, y):

        probs = self.inverse_link(X @ self.beta)
        return roc_auc_score(y.numpy(), probs.numpy())

    # =========================================================
    # VISUALIZACIONES
    # =========================================================
    def plot_accuracy_density(self, X, y, threshold=0.5, figsize=(10, 6)):

        acc = self.accuracy_samples(X, y, threshold)
        acc_flat = acc.reshape(-1).numpy()

        df = pd.DataFrame(acc_flat, columns=['accuracy'])

        df.plot.density(figsize=figsize, title='Accuracy')
        plt.xlabel('Accuracy')
        plt.show()

    def plot_auc_density(self, X, y, figsize=(10, 6)):

        auc = self.auc_samples(X, y)
        auc_flat = auc.reshape(-1).numpy()

        df = pd.DataFrame(auc_flat, columns=['AUC'])

        df.plot.density(figsize=figsize, title='AUC Score')
        plt.xlabel('AUC')
        plt.show()

    # =========================================================
    # SUMMARY
    # =========================================================
    def summary_metrics(self, X, y, threshold=0.5):

        acc_samples = self.accuracy_samples(X, y, threshold)
        acc_plugin = self.accuracy_plugin(X, y, threshold)

        auc_samples = self.auc_samples(X, y)
        auc_plugin = self.auc_plugin(X, y)

        nnll = self.normalized_neg_log_likelihood_samples(X, y)
        nnll_plugin = self.normalized_neg_log_likelihood_plugin(X, y)

        data = {
            "NNLL": [
                nnll.mean(),
                nnll.std(),
                nnll.median(),
                torch.quantile(nnll, 0.05),
                torch.quantile(nnll, 0.95),
                nnll_plugin,
            ],
            "AUC": [
                auc_samples.mean(),
                auc_samples.std(),
                auc_samples.median(),
                torch.quantile(auc_samples, 0.05),
                torch.quantile(auc_samples, 0.95),
                auc_plugin,
            ],
            "ACC": [
                acc_samples.mean(),
                acc_samples.std(),
                acc_samples.median(),
                torch.quantile(acc_samples, 0.05),
                torch.quantile(acc_samples, 0.95),
                acc_plugin,
            ],
        }

        columns = ["mean", "std", "median", "5%", "95%", "plugin"]

        df = pd.DataFrame.from_dict(data, orient="index", columns=columns)

        return df

class BayesianLogisticRegression(BayesianBinaryGLM):

    def __init__(self, **kwargs):

        super().__init__(inverse_link=torch.sigmoid, **kwargs)

    # =========================================================
    # LOG POSTERIOR (con logits estable)
    # =========================================================
    def log_posterior(self, beta):

        z = self.X @ beta.T

        log_like = torch.distributions.Bernoulli(logits=z).log_prob(
            self.y[:, None]
        ).sum(0)

        log_prior = self.beta_prior.log_prob(beta)

        return log_like + log_prior

class BayesianProbitRegression(BayesianBinaryGLM):

    def __init__(self, **kwargs):

        normal = torch.distributions.Normal(0.0, 1.0)

        super().__init__(
            inverse_link=lambda x: normal.cdf(x),
            **kwargs
        )

        self._normal = normal  # para reutilizar

    # =========================================================
    # LOG POSTERIOR
    # =========================================================
    def log_posterior(self, beta):

        z = self.X @ beta.T

        # probit: Φ(z)
        mu = self._normal.cdf(z)

        # evitar log(0)
        eps = 1e-8
        mu = torch.clamp(mu, eps, 1 - eps)

        log_like = torch.distributions.Bernoulli(probs=mu).log_prob(
            self.y[:, None]
        ).sum(0)

        log_prior = self.beta_prior.log_prob(beta)

        return log_like + log_prior

class BayesianPoissonRegression(BayesianGLM):

    def __init__(self, inverse_link=torch.exp, **kwargs):

        super().__init__(inverse_link=inverse_link, **kwargs)

    # =========================================================
    # LOG POSTERIOR (equivalente a model en numpyro)
    # =========================================================
    def log_posterior(self, beta):

        eta = self.X @ beta.T
        mu = self.inverse_link(eta)

        log_like = torch.distributions.Poisson(mu).log_prob(
            self.y[:, None]
        ).sum(0)

        log_prior = self.beta_prior.log_prob(beta)

        return log_like + log_prior

    # =========================================================
    # INIT PARAMS
    # =========================================================
    def init_params(self, n_chains):

        return {
            "beta": torch.zeros((n_chains, self.d))
        }

    # =========================================================
    # STORE
    # =========================================================
    def _store_samples(self, samples):

        self.beta_samples = samples["beta"]
        self.beta = self.beta_samples.mean(dim=(0, 1))

    # =========================================================
    # NEG LOG LIKELIHOOD (idéntico a tu versión JAX)
    # =========================================================
    def neg_log_likelihood(self, X, y, params):

        beta = params["beta"]

        eta = X @ beta
        mu = self.inverse_link(eta)

        return -torch.distributions.Poisson(mu).log_prob(y).sum()

    # =========================================================
    # SAMPLE LIKELIHOOD (posterior predictive)
    # =========================================================
    def sample_likelihood(self, mu, params):

        return torch.distributions.Poisson(mu).sample()

class BayesianNegBinomRegression(BayesianGLM):

    def __init__(self, inverse_link=torch.exp, alpha_prior=None, **kwargs):

        super().__init__(inverse_link=inverse_link, **kwargs)

        self.alpha_prior = alpha_prior if alpha_prior is not None else torch.distributions.HalfNormal(1.0)

    # =========================================================
    # LOG POSTERIOR (equivalente a model en numpyro)
    # =========================================================
    def log_posterior(self, beta, alpha):

        eta = self.X @ beta.T
        mu = self.inverse_link(eta)

        # conversión NB2 -> torch
        probs = alpha[:, None] / (alpha[:, None] + mu)

        log_like = torch.distributions.NegativeBinomial(
            total_count=alpha[:, None],
            probs=probs
        ).log_prob(self.y[:, None]).sum(0)

        log_prior_beta = self.beta_prior.log_prob(beta)
        log_prior_alpha = self.alpha_prior.log_prob(alpha)

        return log_like + log_prior_beta + log_prior_alpha

    # =========================================================
    # INIT PARAMS
    # =========================================================
    def init_params(self, n_chains):

        return {
            "beta": torch.zeros((n_chains, self.d)),
            "alpha": torch.ones(n_chains)  # positivo desde inicio
        }

    # =========================================================
    # STORE
    # =========================================================
    def _store_samples(self, samples):

        self.beta_samples = samples["beta"]
        self.alpha_samples = samples["alpha"]

        self.beta = self.beta_samples.mean(dim=(0, 1))
        self.alpha = self.alpha_samples.mean()

    # =========================================================
    # NEG LOG LIKELIHOOD (idéntico concepto que Poisson)
    # =========================================================
    def neg_log_likelihood(self, X, y, params):

        beta = params["beta"]
        alpha = params["alpha"]

        eta = X @ beta
        mu = self.inverse_link(eta)

        probs = alpha / (alpha + mu)

        return -torch.distributions.NegativeBinomial(
            total_count=alpha,
            probs=probs
        ).log_prob(y).sum()

    # =========================================================
    # SAMPLE LIKELIHOOD (posterior predictive)
    # =========================================================
    def sample_likelihood(self, mu, params):

        alpha = params["alpha"]

        probs = alpha / (alpha + mu)

        return torch.distributions.NegativeBinomial(
            total_count=alpha,
            probs=probs
        ).sample()

class BayesianGammaRegression(BayesianGLM):

    def __init__(self, inverse_link=torch.exp, alpha_prior=None, **kwargs):

        super().__init__(inverse_link=inverse_link, **kwargs)

        self.alpha_prior = alpha_prior if alpha_prior is not None else torch.distributions.HalfNormal(1.0)

    # =========================================================
    # LOG POSTERIOR
    # =========================================================
    def log_posterior(self, beta, alpha):

        eta = self.X @ beta.T
        mu = self.inverse_link(eta)

        # Gamma: concentration=alpha, rate=alpha/mu
        log_like = torch.distributions.Gamma(
            concentration=alpha[:, None],
            rate=alpha[:, None] / mu
        ).log_prob(self.y[:, None]).sum(0)

        log_prior_beta = self.beta_prior.log_prob(beta)
        log_prior_alpha = self.alpha_prior.log_prob(alpha)

        return log_like + log_prior_beta + log_prior_alpha

    # =========================================================
    # INIT PARAMS
    # =========================================================
    def init_params(self, n_chains):

        return {
            "beta": torch.zeros((n_chains, self.d)),
            "alpha": torch.ones(n_chains)
        }

    # =========================================================
    # STORE
    # =========================================================
    def _store_samples(self, samples):

        self.beta_samples = samples["beta"]
        self.alpha_samples = samples["alpha"]

        self.beta = self.beta_samples.mean(dim=(0, 1))
        self.alpha = self.alpha_samples.mean()

    # =========================================================
    # NEG LOG LIKELIHOOD
    # =========================================================
    def neg_log_likelihood(self, X, y, params):

        beta = params["beta"]
        alpha = params["alpha"]

        eta = X @ beta
        mu = self.inverse_link(eta)

        return -torch.distributions.Gamma(
            concentration=alpha,
            rate=alpha / mu
        ).log_prob(y).sum()

    # =========================================================
    # SAMPLE LIKELIHOOD
    # =========================================================
    def sample_likelihood(self, mu, params):

        alpha = params["alpha"]

        return torch.distributions.Gamma(
            concentration=alpha,
            rate=alpha / mu
        ).sample()

class CallOptionPrice:
    """
    Valoración de Opciones de Compra (Call) mediante el Modelo Black-Scholes

    Implementa la fórmula analítica de Black-Scholes y estimaciones mediante
    Monte Carlo estándar, muestreo por importancia y métodos MCMC (Metropolis-Hastings).
    """

    def __init__(self, s_0, r, sigma, k, t):
        """
        Inicializa los parámetros de la opción call.

        Parámetros
        ----------
        s_0 : float
            Precio inicial del activo subyacente.
        r : float
            Tasa de interés libre de riesgo (continua).
        sigma : float
            Volatilidad del activo subyacente.
        k : float
            Precio de ejercicio (strike price).
        t : float
            Tiempo hasta vencimiento (en años).
        """
        self.s_0 = torch.Tensor([s_0])
        self.r = torch.Tensor([r])
        self.sigma = torch.Tensor([sigma])
        self.k = torch.Tensor([k])
        self.t = torch.Tensor([t])

        # Valor crítico theta: punto donde el payoff se vuelve positivo
        # Corresponde a z tal que S_T = K
        self.theta = (torch.log(self.k / self.s_0) - (self.r - self.sigma ** 2 / 2) * self.t) / (self.sigma * torch.sqrt(self.t))

        # Distribución normal estándar
        self.Z = torch.distributions.Normal(0.0, 1.0)

        # Precio analítico (se calcula cuando se necesita)
        self.price = None

    def payoff(self, z):
        """
        Calcula el payoff de la opción call en función de la variable normal Z.

        Parámetros
        ----------
        z : torch.Tensor
            Variable aleatoria normal estándar.

        Retorna
        -------
        torch.Tensor
            Payoff de la opción call: max(S_T - K, 0)
            donde S_T = S_0 * exp((r - σ²/2)t + σ√t Z)

        Notas
        -----
        El payoff es positivo solo cuando Z > θ, donde:
        θ = (log(K/S_0) - (r - σ²/2)t) / (σ√t)
        """
        payoff = torch.where(
            z <= self.theta,
            torch.zeros_like(z),
            self.s_0 * torch.exp((self.r - self.sigma**2/2) * self.t + self.sigma * torch.sqrt(self.t) * z) - self.k
        )
        return payoff

    def g_min_var(self, z):
        """
        Calcula la función de densidad de la distribución de mínima varianza.

        Esta distribución es óptima para muestreo por importancia, ya que
        minimiza la varianza del estimador.

        Parámetros
        ----------
        z : torch.Tensor
            Variable aleatoria normal estándar.

        Retorna
        -------
        torch.Tensor
            Densidad de la distribución de mínima varianza evaluada en z.

        Notas
        -----
        La distribución de mínima varianza tiene densidad:
        g(z) ∝ payoff(z) * φ(z)
        donde φ(z) es la densidad normal estándar.
        La constante de normalización es:
        c = √(2π) [S_0 e^{rt} Φ(θ - σ√t) - K Φ(θ)]
        """
        c = math.sqrt(2 * math.pi) * (
            self.s_0 * torch.exp(self.r * self.t) * (1 - self.Z.cdf(self.theta - self.sigma * torch.sqrt(self.t))) -
            self.k * (1 - self.Z.cdf(self.theta))
        )
        return self.payoff(z) * torch.exp(-z**2 / 2) / c

    def compute_price(self):
        """
        Calcula el precio analítico de la opción call mediante Black-Scholes.

        Utiliza la fórmula cerrada de Black-Scholes:
        C = S_0 Φ(d1) - K e^{-rt} Φ(d2)

        Parámetros
        ----------
        t : float
            Tiempo hasta vencimiento (en años).
        """
        # Parámetros d1 y d2 de Black-Scholes
        d1 = (torch.log(self.s_0 / self.k) + (self.r + self.sigma ** 2 / 2) * self.t) / (self.sigma * torch.sqrt(self.t))
        d2 = d1 - self.sigma * torch.sqrt(self.t)

        # Fórmula de Black-Scholes
        self.price = self.s_0 * self.Z.cdf(d1) - self.k * torch.exp(-self.r * self.t) * self.Z.cdf(d2)

    def compute_price_mc(self, n=10000, decimals=10):
        """
        Estima el precio mediante Monte Carlo estándar.

        Simula trayectorias del activo bajo la medida neutral al riesgo:
        S_T = S_0 * exp((r - σ²/2)T + σ√T Z)

        Parámetros
        ----------
        n : int, opcional
            Número de simulaciones. Por defecto 10,000.
        decimals : int, opcional
            Número de decimales para mostrar. Por defecto 10.

        Retorna
        -------
        tuple
            (precio_estimado, error_absoluto, varianza)

        Notas
        -----
        El estimador Monte Carlo estándar es:
        Ĉ = e^{-rt} * (1/n) * Σ payoff(Z_i)
        """
        # Calcular precio analítico para referencia
        self.compute_price()

        # Simular variable normal estándar
        z = self.Z.sample((n,))

        # Payoff de la opción call
        payoff = self.payoff(z)

        # Precio estimado (valor esperado descontado)
        price_hat = torch.exp(-self.r * self.t) * payoff.mean()

        # Varianza del estimador
        var = torch.exp(-2 * self.r * self.t) * payoff.var() / n

        # Error absoluto respecto al precio analítico
        error = torch.abs(self.price - price_hat)

        print("----- Monte Carlo -----")
        print(f"Precio  : {self.price.item():.{decimals}f}")
        print(f"Precio estimado : {price_hat.item():.{decimals}f}")
        print(f"Error: {error.item():.{decimals}f}")
        print(f"Varianza: {var.item():.{decimals}f}")

        return price_hat, error, var

    def compute_price_is(self, theta=None, n=10000, decimals=10):
        """
        Estima el precio mediante muestreo por importancia.

        Cambia la media de la distribución de Z para muestrear más eficientemente
        en regiones donde el payoff es positivo.

        Parámetros
        ----------
        theta : float, opcional
            Desplazamiento para la media en el muestreo por importancia.
            Si es None, se calcula automáticamente como el valor que hace
            que el activo esté en dinero (at-the-money).
        n : int, opcional
            Número de simulaciones. Por defecto 10,000.
        decimals : int, opcional
            Número de decimales para mostrar. Por defecto 10.

        Retorna
        -------
        tuple
            (precio_estimado, error_absoluto, varianza)

        Notas
        -----
        La distribución propuesta es N(θ, 1) para Z.
        Los pesos de importancia se calculan como:
        w(z) = exp(-θz + θ²/2)

        El estimador por importancia es:
        Ĉ = e^{-rt} * (1/n) * Σ payoff(Z_i) * w(Z_i)

        donde Z_i ~ N(θ, 1).
        """
        # Calcular theta óptimo si no se proporciona
        if theta is None:
            theta = self.theta
        else:
            theta = torch.Tensor([theta])

        # Calcular precio analítico
        self.compute_price()

        # Muestrear de la distribución propuesta (normal desplazada)
        z = self.Z.sample((n,)) + theta

        # Payoff de la opción
        payoff = self.payoff(z)

        # Pesos de importancia (razón de densidades)
        # f(z)/g(z) donde f es N(0,1) y g es N(θ,1)
        weights = torch.exp(-theta * z + theta ** 2 / 2)

        # Estimador por importancia
        estimator = payoff * weights
        price_hat = torch.exp(-self.r * self.t) * estimator.mean()

        # Varianza del estimador
        var = torch.exp(-2 * self.r * self.t) * estimator.var() / n

        # Error absoluto
        error = torch.abs(self.price - price_hat)

        print("----- Importance Sampling -----")
        print(f"Precio  : {self.price.item():.{decimals}f}")
        print(f"Precio estimado : {price_hat.item():.{decimals}f}")
        print(f"Error : {error.item():.{decimals}f}")
        print(f"Varianza : {var.item():.{decimals}f}")

        return price_hat, error, var

    def compute_price_mh(self, n=10000, decimals=10, n_chains=1, step_size=1.0, burn_in=None):
        """
        Estima el precio mediante el algoritmo Metropolis-Hastings (MCMC).

        Este método utiliza una cadena de Markov para muestrear de la distribución
        objetivo proporcional a payoff(z) * φ(z), que es óptima para el muestreo
        por importancia.

        Parámetros
        ----------
        n : int, opcional
            Número total de iteraciones de la cadena. Por defecto 10,000.
        decimals : int, opcional
            Número de decimales para mostrar. Por defecto 10.
        n_chains : int, opcional
            Número de cadenas paralelas. Por defecto 1.
        step_size : float, opcional
            Tamaño de paso para la distribución de propuesta. Por defecto 1.0.
        burn_in : int, opcional
            Número de iteraciones iniciales a descartar (periodo de calentamiento).
            Si es None, no se descartan iteraciones.

        Retorna
        -------
        tuple
            (precio_estimado, error_absoluto, varianza, muestras)

        Notas
        -----
        El algoritmo Metropolis-Hastings se utiliza para muestrear de:
        π(z) ∝ payoff(z) * φ(z)

        Luego, el precio se estima como:
        Ĉ = e^{-rt} * (1/m) * Σ payoff(z_i) * φ(z_i) / g(z_i)

        donde g es la distribución de mínima varianza y φ es la densidad normal.
        """
        # Inicializar cadenas
        x_n = torch.zeros((n, n_chains))

        # Algoritmo Metropolis-Hastings
        for i in range(n - 1):
            # Generar propuesta desde la distribución actual más ruido
            y = step_size * self.Z.sample((n_chains,)) + x_n[i]
            u = torch.rand((n_chains,))

            # Calcular log-razón de aceptación
            log_alpha = (torch.log(self.payoff(y)) + self.Z.log_prob(y) -
                        torch.log(self.payoff(x_n[i])) - self.Z.log_prob(x_n[i]))

            # Aceptación según criterio de Metropolis
            alpha = torch.minimum(torch.zeros_like(log_alpha), log_alpha)
            accept = torch.log(u) <= alpha

            # Actualizar cadena
            x_n[i + 1] = torch.where(accept, y, x_n[i])

        # Descartar periodo de calentamiento si se especifica
        if burn_in is not None:
            x_n = x_n[burn_in:]

        # Aplanar todas las cadenas
        x_n_flat = x_n.flatten()

        # Calcular payoff, densidad normal y densidad de mínima varianza
        payoff = self.payoff(x_n_flat)
        norm = self.Z.log_prob(x_n_flat).exp()
        g = self.g_min_var(x_n_flat)

        # Estimador por importancia con la muestra MCMC
        # Se usa la distribución de mínima varianza como distribución de importancia
        price_hat = (torch.exp(-self.r * self.t) * payoff * norm / g).mean()

        # Varianza del estimador
        var = (torch.exp(-self.r * self.t) * payoff * norm / g).var() / x_n_flat.shape[0]

        # Calcular precio analítico y error
        self.compute_price()
        error = torch.abs(self.price - price_hat)

        print("----- Metropolis-Hastings -----")
        print(f"Precio  : {self.price.item():.{decimals}f}")
        print(f"Precio estimado : {price_hat.item():.{decimals}f}")
        print(f"Error: {error.item():.{decimals}f}")
        print(f"Varianza: {var.item():.{decimals}f}")

        return price_hat, error, var, x_n_flat


class BayesianNeuralNetwork1Hidden:

    def __init__(self, X, y, hidden_dim=10, tau2=1.0):
        self.X = X
        self.y = y
        self.n, self.d = X.shape
        self.hidden_dim = hidden_dim
        self.tau2 = tau2

        # Dimensiones de parámetros
        self.W1_dim = self.d * self.hidden_dim  # W1
        self.b1_dim = hidden_dim  # b1
        self.W2_dim = hidden_dim  # W2
        self.b2_dim = 1  # b2

        self.W1_prior = torch.distributions.MultivariateNormal(torch.zeros(self.W1_dim),
                                                               tau2 * torch.eye(self.W1_dim))
        self.b1_prior = torch.distributions.MultivariateNormal(torch.zeros(self.b1_dim),
                                                               tau2 * torch.eye(self.b1_dim))

        self.W2_prior = torch.distributions.MultivariateNormal(torch.zeros(self.W2_dim),
                                                               tau2 * torch.eye(self.W2_dim))

        self.b2_prior = torch.distributions.MultivariateNormal(torch.zeros(self.b2_dim),
                                                               tau2 * torch.eye(self.b2_dim))
        self.W1_samples = None
        self.W2_samples = None
        self.b1_samples = None
        self.b2_samples = None

        self.n_chains = None
        self.weigths = None
        self.bias = None

    def forward(self, X, W1, W2, b1, b2):
        a = torch.tanh(X @ W1.reshape(-1, self.d, self.hidden_dim) + b1[:, None, :])
        z = a @ W2.reshape(-1, self.hidden_dim, 1) + b2[:, None, :]

        return z.squeeze(-1)

    def log_posterior(self, W1, W2, b1, b2):
        z = self.forward(self.X, W1, W2, b1, b2)

        log_like = torch.distributions.Bernoulli(logits=z).log_prob(
            self.y[None, :]
        ).sum(1)

        log_W1prior = self.W1_prior.log_prob(W1)
        log_W2prior = self.W2_prior.log_prob(W2)
        log_b1prior = self.b1_prior.log_prob(b1)
        log_b2prior = self.b2_prior.log_prob(b2)

        log_prior = log_W1prior + log_b1prior + log_W2prior + log_b2prior

        return log_like + log_prior

    def fit(self, n_iter=1000, burn_in=0, n_chains=1, step_size=0.01):
        W1 = torch.zeros((n_chains, self.W1_dim))
        b1 = torch.zeros((n_chains, self.b1_dim))
        W2 = torch.zeros((n_chains, self.W2_dim))
        b2 = torch.zeros((n_chains, self.b2_dim))

        self.W1_samples = torch.zeros((n_iter, n_chains, self.W1_dim))
        self.W2_samples = torch.zeros((n_iter, n_chains, self.W2_dim))
        self.b1_samples = torch.zeros((n_iter, n_chains, self.b1_dim))
        self.b2_samples = torch.zeros((n_iter, n_chains, self.b2_dim))

        self.n_chains = n_chains

        for i in range(n_iter):
            W1_prop = W1 + step_size * torch.randn_like(W1)
            W2_prop = W2 + step_size * torch.randn_like(W2)
            b1_prop = b1 + step_size * torch.randn_like(b1)
            b2_prop = b2 + step_size * torch.randn_like(b2)

            logp_prop = self.log_posterior(W1_prop, W2_prop, b1_prop, b2_prop)
            logp = self.log_posterior(W1, W2, b1, b2)

            log_alpha = logp_prop - logp
            alpha = torch.minimum(torch.zeros_like(log_alpha), log_alpha)

            u = torch.log(torch.rand_like(alpha))
            accept = u <= alpha

            W1 = torch.where(accept[:, None], W1_prop, W1)
            W2 = torch.where(accept[:, None], W2_prop, W2)
            b1 = torch.where(accept[:, None], b1_prop, b1)
            b2 = torch.where(accept[:, None], b2_prop, b2)

            self.W1_samples[i] = W1
            self.W2_samples[i] = W2
            self.b1_samples[i] = b1
            self.b2_samples[i] = b2

        self.W1_samples = self.W1_samples[burn_in:]
        self.W2_samples = self.W2_samples[burn_in:]
        self.b1_samples = self.b1_samples[burn_in:]
        self.b2_samples = self.b2_samples[burn_in:]

        self.weigths = [self.W1_samples.mean(dim=(0, 1)).reshape(self.d, self.hidden_dim),
                        self.W2_samples.mean(dim=(0, 1)).reshape(self.hidden_dim, 1)]

        self.bias = [self.b1_samples.mean(dim=(0, 1)).reshape(1, self.hidden_dim),
                     self.b2_samples.mean(dim=(0, 1))]

    def predict_proba_samples(self, X):
        a = torch.tanh(
            X @ self.W1_samples.reshape(-1, self.n_chains, self.d, self.hidden_dim) + self.b1_samples[:, :, None, :])
        z = a @ self.W2_samples.reshape(-1, self.n_chains, self.hidden_dim, 1) + self.b2_samples[:, :, :, None]

        return torch.sigmoid(z).squeeze(-1)

    def predict_mean_proba(self, X):
        probs = self.predict_proba_samples(X)

        return probs.mean(dim=(0, 1))

    def predict_samples(self, X, threshold=0.5):
        probs = self.predict_proba_samples(X)

        return (probs >= threshold).float()

    def predict_mean(self, X, threshold=0.5):
        probs = self.predict_mean_proba(X)

        return (probs >= threshold).float()

    def accuracy_samples(self, X, y, threshold=0.5):
        probs = self.predict_proba_samples(X)

        y_pred = (probs >= threshold).float()

        acc = (y_pred == y[None, None, :]).float().mean(dim=2)

        return acc

    def accuracy_mean(self, X, y):
        acc = self.accuracy_samples(X, y)
        return acc.mean()

    def accuracy(self, X, y, threshold=0.5):
        y_pred = self.predict_mean(X, threshold)

        acc = (y_pred == y).float().mean()
        return acc.mean()

    def accuracy_ci(self, X, y, alpha=0.05):
        acc = self.accuracy_samples(X, y)

        lower = torch.quantile(acc, alpha / 2)
        upper = torch.quantile(acc, 1 - alpha / 2)

        return lower, upper


class BayesianNeuralNetwork1Hidden_MALA:

    def __init__(self, X, y, activation, hidden_dim=10, tau2=1.0):
        self.X = X
        self.y = y
        self.n, self.d = X.shape
        self.hidden_dim = hidden_dim
        self.tau2 = tau2

        self.W1_dim = self.d * hidden_dim
        self.b1_dim = hidden_dim
        self.W2_dim = hidden_dim
        self.b2_dim = 1

        self.W1_samples = None
        self.W2_samples = None
        self.b1_samples = None
        self.b2_samples = None
        self.n_chains = None
        self.activation = activation

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, X, W1, W2, b1, b2):
        a = self.activation(X @ W1.reshape(-1, self.d, self.hidden_dim) + b1[:, None, :])
        z = a @ W2.reshape(-1, self.hidden_dim, 1) + b2[:, None, :]

        return z.squeeze(-1)

    # -------------------------
    # Log posterior
    # -------------------------
    def log_posterior(self, W1, W2, b1, b2):
        z = self.forward(self.X, W1, W2, b1, b2)

        log_like = torch.distributions.Bernoulli(logits=z).log_prob(
            self.y[None, :]
        ).sum(1)

        log_prior = (
                torch.distributions.Normal(0, self.tau2 ** 0.5).log_prob(W1).sum(1)
                + torch.distributions.Normal(0, self.tau2 ** 0.5).log_prob(W2).sum(1)
                + torch.distributions.Normal(0, self.tau2 ** 0.5).log_prob(b1).sum(1)
                + torch.distributions.Normal(0, self.tau2 ** 0.5).log_prob(b2).sum(1)
        )

        return log_like + log_prior

    # -------------------------
    # Gradiente
    # -------------------------

    def grad_log_posterior(self, W1, W2, b1, b2):
        W1 = W1.detach().requires_grad_(True)
        W2 = W2.detach().requires_grad_(True)
        b1 = b1.detach().requires_grad_(True)
        b2 = b2.detach().requires_grad_(True)

        logp = self.log_posterior(W1, W2, b1, b2).sum()

        grads = torch.autograd.grad(
            logp, (W1, W2, b1, b2),
            retain_graph=False,
            create_graph=False,
            allow_unused=True
        )

        # proteger contra None
        grads = tuple(
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(grads, (W1, W2, b1, b2))
        )

        return grads

    # -------------------------
    # log q
    # -------------------------
    def log_q(self, x_from, x_to, grad_from, step):
        mean = x_from + 0.5 * step * grad_from
        return -((x_to - mean) ** 2).sum(dim=1) / (2 * step)

    # -------------------------
    # FIT (MALA)
    # -------------------------
    def fit(self, n_iter=2000, burn_in=1000, n_chains=1, step_size=1e-3):
        self.n_chains = n_chains

        W1 = torch.zeros((n_chains, self.W1_dim))
        W2 = torch.zeros((n_chains, self.W2_dim))
        b1 = torch.zeros((n_chains, self.b1_dim))
        b2 = torch.zeros((n_chains, self.b2_dim))

        self.W1_samples = torch.zeros((n_iter, n_chains, self.W1_dim))
        self.W2_samples = torch.zeros((n_iter, n_chains, self.W2_dim))
        self.b1_samples = torch.zeros((n_iter, n_chains, self.b1_dim))
        self.b2_samples = torch.zeros((n_iter, n_chains, self.b2_dim))

        logp = self.log_posterior(W1, W2, b1, b2)

        for i in range(n_iter):
            gW1, gW2, gb1, gb2 = self.grad_log_posterior(W1, W2, b1, b2)

            noise_scale = torch.sqrt(torch.tensor(step_size))

            W1_prop = W1 + 0.5 * step_size * gW1 + noise_scale * torch.randn_like(W1)
            W2_prop = W2 + 0.5 * step_size * gW2 + noise_scale * torch.randn_like(W2)
            b1_prop = b1 + 0.5 * step_size * gb1 + noise_scale * torch.randn_like(b1)
            b2_prop = b2 + 0.5 * step_size * gb2 + noise_scale * torch.randn_like(b2)

            logp_prop = self.log_posterior(W1_prop, W2_prop, b1_prop, b2_prop)

            gW1_p, gW2_p, gb1_p, gb2_p = self.grad_log_posterior(W1_prop, W2_prop, b1_prop, b2_prop)

            log_q_forward = (
                    self.log_q(W1, W1_prop, gW1, step_size)
                    + self.log_q(W2, W2_prop, gW2, step_size)
                    + self.log_q(b1, b1_prop, gb1, step_size)
                    + self.log_q(b2, b2_prop, gb2, step_size)
            )

            log_q_backward = (
                    self.log_q(W1_prop, W1, gW1_p, step_size)
                    + self.log_q(W2_prop, W2, gW2_p, step_size)
                    + self.log_q(b1_prop, b1, gb1_p, step_size)
                    + self.log_q(b2_prop, b2, gb2_p, step_size)
            )

            log_alpha = logp_prop - logp + log_q_backward - log_q_forward
            accept = torch.log(torch.rand_like(log_alpha)) < log_alpha

            W1 = torch.where(accept[:, None], W1_prop, W1)
            W2 = torch.where(accept[:, None], W2_prop, W2)
            b1 = torch.where(accept[:, None], b1_prop, b1)
            b2 = torch.where(accept[:, None], b2_prop, b2)

            logp = torch.where(accept, logp_prop, logp)

            self.W1_samples[i] = W1
            self.W2_samples[i] = W2
            self.b1_samples[i] = b1
            self.b2_samples[i] = b2

        # Burn-in
        self.W1_samples = self.W1_samples[burn_in:]
        self.W2_samples = self.W2_samples[burn_in:]
        self.b1_samples = self.b1_samples[burn_in:]
        self.b2_samples = self.b2_samples[burn_in:]

    # -------------------------
    # Predicción
    # -------------------------
    def predict_proba_samples(self, X):
        a = self.activation(
            X @ self.W1_samples.reshape(-1, self.n_chains, self.d, self.hidden_dim)
            + self.b1_samples[:, :, None, :]
        )

        z = (
                a @ self.W2_samples.reshape(-1, self.n_chains, self.hidden_dim, 1)
                + self.b2_samples[:, :, :, None]
        )

        return torch.sigmoid(z).squeeze(-1)

    def predict_mean_proba(self, X):
        probs = self.predict_proba_samples(X)
        return probs.mean(dim=(0, 1))

    def predict_samples(self, X, threshold=0.5):
        probs = self.predict_proba_samples(X)
        return (probs >= threshold).float()

    def predict_mean(self, X, threshold=0.5):
        probs = self.predict_mean_proba(X)
        return (probs >= threshold).float()

    # -------------------------
    # Métricas
    # -------------------------
    def accuracy_samples(self, X, y, threshold=0.5):
        probs = self.predict_proba_samples(X)
        y_pred = (probs >= threshold).float()

        acc = (y_pred == y[None, None, :]).float().mean(dim=2)
        return acc

    def accuracy_mean(self, X, y):
        return self.accuracy_samples(X, y).mean()

    def accuracy(self, X, y, threshold=0.5):
        y_pred = self.predict_mean(X, threshold)
        return (y_pred == y).float().mean()

    def accuracy_ci(self, X, y, alpha=0.05):
        acc = self.accuracy_samples(X, y)

        lower = torch.quantile(acc, alpha / 2)
        upper = torch.quantile(acc, 1 - alpha / 2)

        return lower, upper

