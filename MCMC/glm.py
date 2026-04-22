import jax.numpy as jnp
import jax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.random as random
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

class BayesianGLM():

    def __init__(self, link=None, inverse_link=None, beta_prior=None, burn_in=100, num_samples=1000, num_chains=1, seed=0, **kwargs):

        self.link = link
        self.inverse_link = inverse_link if inverse_link is not None else (lambda x: x)

        self.beta_prior = beta_prior if beta_prior is not None else dist.Normal(0.0, 1.0)

        self.burn_in = burn_in
        self.num_samples = num_samples
        self.num_chains = num_chains

        self.rng_key = random.PRNGKey(seed)

        self.kernel = None
        self.mcmc = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    # =========================================================
    # ABSTRACT METHODS
    # =========================================================

    def model(self, X, y=None):
        raise NotImplementedError

    def sample_likelihood(self, mu, params):
        raise NotImplementedError

    def neg_log_likelihood(self, X, y, params):
        raise NotImplementedError

    # =========================================================
    # FIT
    # =========================================================

    def fit(self, X, y):

        self.kernel = NUTS(self.model)

        self.mcmc = MCMC(
            self.kernel,
            num_warmup=self.burn_in,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            chain_method="parallel"
        )

        self.mcmc.run(self.rng_key, X, y)

    # =========================================================
    # POSTERIOR SAMPLES
    # =========================================================

    def get_params_samples(self):
        return self.mcmc.get_samples()

    def get_beta_samples(self):
        return self.mcmc.get_samples()["beta"]

    def get_beta_hat(self):
        return jnp.mean(self.get_beta_samples(), axis=0)

    def get_plugin_params(self):

        samples = self.get_params_samples()

        return {
            k: jnp.mean(v, axis=0)
            for k, v in samples.items()
        }

    # =========================================================
    # PREDICTIONS
    # =========================================================

    def predict_plugin(self, X):
        """Predicción usando beta_hat (modelo determinista)"""
        beta = self.get_beta_hat()
        eta = jnp.dot(X, beta)
        return self.inverse_link(eta)

    def predict_posterior_samples(self, X):
        """Predicciones (n, S) integrando sobre posterior"""

        beta = self.get_beta_samples()

        if beta.ndim == 1:
            beta = beta[None, :]

        eta = jnp.dot(X, beta.T)
        return self.inverse_link(eta)

    def predict_posterior_mean(self, X):
        """E[y|x,D] aproximado por Monte Carlo"""
        y_hat = self.predict_posterior_samples(X)
        return jnp.mean(y_hat, axis=1)

    def predict_credible_interval(self, X, alpha=0.05):
        """Intervalos creíbles punto a punto"""

        y_hat = self.predict_posterior_samples(X)

        lower = jnp.quantile(y_hat, alpha / 2, axis=1)
        upper = jnp.quantile(y_hat, 1 - alpha / 2, axis=1)

        return lower, upper

    def forward(self, X):
        """Alias opcional (compatibilidad)"""
        return self.predict_posterior_samples(X)

    # =========================================================
    # POSTERIOR PREDICTIVE (full likelihood sampling)
    # =========================================================

    def sample_posterior_predictive(self, X):

        samples = self.get_params_samples()
        beta = samples["beta"]

        if beta.ndim == 1:
            beta = beta[None, :]

        eta = jnp.dot(X, beta.T)
        mu = self.inverse_link(eta)

        return self.sample_likelihood(mu, samples)

    # =========================================================
    # NLL
    # =========================================================

    def normalized_neg_log_likelihood_plugin(self, X, y):

        beta = self.get_beta_hat()
        n = X.shape[0]

        params = self.get_plugin_params()

        return self.neg_log_likelihood(X, y, params) / n

    def normalized_neg_log_likelihood_samples(self, X, y):

        if self.mcmc is None:
            raise ValueError("Debes llamar a fit() antes.")

        samples = self.get_params_samples()
        n = X.shape[0]

        nnll = jax.vmap(
            self.neg_log_likelihood,
            in_axes=(None, None, 0)
        )(X, y, samples) / n

        return nnll

    # =========================================================
    # METRICS
    # =========================================================

    def summary(self):

        self.mcmc.print_summary()

    def summary_metrics(self, X, y, return_dict=False):

        nnll = self.normalized_neg_log_likelihood_samples(X, y)
        nnll = jnp.asarray(nnll)

        nnll_plugin = self.normalized_neg_log_likelihood_plugin(X, y)

        stats = {
            "mean": jnp.mean(nnll),
            "std": jnp.std(nnll),
            "median": jnp.median(nnll),
            "5%": jnp.quantile(nnll, 0.05),
            "95%": jnp.quantile(nnll, 0.95),
            "plugin_beta_hat": nnll_plugin
        }

        if return_dict:
            return stats

        print("\nSummary Normalized Negative Log-Likelihood")
        print("------------------------------------------")
        print(f"beta_hat NLL: {nnll_plugin:.4f}\n")

        for k, v in stats.items():
            if k != "plugin_beta_hat":
                print(f"{k:10}: {v:.4f}")

    # =========================================================
    # VISUALIZATIONS
    # =========================================================

    def plot_nll_density(self, X, y, color="blue", figsize=(10, 6)):

        nnll = self.normalized_neg_log_likelihood_samples(X, y)
        nnll = np.array(nnll)

        kde = gaussian_kde(nnll)
        x_grid = np.linspace(nnll.min(), nnll.max(), 200)

        plt.figure(figsize=figsize)
        plt.plot(x_grid, kde(x_grid), color=color)
        plt.title("Normalized Negative Log-Likelihood")
        plt.xlabel("NLL")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.show()

    def plot_parameter_densities(self, color="blue", figsize=(10, 6), layout=None):

        if self.mcmc is None:
            raise ValueError("Debes llamar a fit() antes.")

        samples = self.get_params_samples()

        for param_name, values in samples.items():

            values_flat = np.array(values).reshape(values.shape[0], -1)
            num_params = values_flat.shape[1]

            if layout is None:
                ncols = int(np.ceil(np.sqrt(num_params)))
                nrows = int(np.ceil(num_params / ncols))
            else:
                nrows, ncols = layout

            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
            axes = np.array(axes).reshape(-1)

            for i in range(num_params):

                kde = gaussian_kde(values_flat[:, i])
                x_grid = np.linspace(
                    values_flat[:, i].min(),
                    values_flat[:, i].max(),
                    200
                )

                axes[i].plot(x_grid, kde(x_grid), color=color)
                axes[i].set_title(f"{param_name}[{i}]")
                axes[i].grid(alpha=0.3)

            for j in range(num_params, len(axes)):
                axes[j].axis("off")

            plt.suptitle(f"Posterior densities: {param_name}")
            plt.tight_layout()
            plt.show()

    def plot_posterior_predictive_density(self, X, idx=0, color="blue", figsize=(8, 4)):

        if self.mcmc is None:
            raise ValueError("Debes llamar a fit() antes.")

        y_samples = np.array(self.predict_posterior_samples(X))

        y_i = y_samples[idx]

        kde = gaussian_kde(y_i)
        x_grid = np.linspace(y_i.min(), y_i.max(), 200)

        plt.figure(figsize=figsize)
        plt.plot(x_grid, kde(x_grid), color=color)

        plt.title(f"Posterior Predictive Density (obs {idx})")
        plt.xlabel("y")
        plt.ylabel("density")
        plt.tight_layout()
        plt.show()

class BayesianLinearRegression(BayesianGLM):

    def __init__(self,
                 sigma_prior=None,
                 **kwargs):

        super().__init__(**kwargs)

        # prior para ruido (desviación estándar)
        self.sigma_prior = sigma_prior if sigma_prior is not None else dist.HalfNormal(1.0)

    # =========================================================
    # MODEL
    # =========================================================

    def model(self, X, y=None):

        n, p = X.shape

        # priors
        beta = numpyro.sample(
            "beta",
            self.beta_prior.expand([p]).to_event(1)
        )

        sigma = numpyro.sample(
            "sigma",
            self.sigma_prior
        )


        with numpyro.plate("data", n):

            mu = jnp.dot(X, beta)

            numpyro.sample(
                "y",
                dist.Normal(mu, sigma),
                obs=y
            )

    # =========================================================
    # LIKELIHOOD (for NLL methods)
    # =========================================================

    def neg_log_likelihood(self, X, y, params):

        beta = params["beta"]
        sigma = params.get("sigma", 1.0)

        mu = jnp.dot(X, beta)

        return -jnp.sum(dist.Normal(mu, sigma).log_prob(y))

    # =========================================================
    # POSTERIOR PREDICTIVE (optional override)
    # =========================================================

    def sample_likelihood(self, mu, params):

        self.rng_key, subkay = random.split(self.rng_key)

        sigma = params["sigma"]

        return dist.Normal(mu, sigma).sample(subkay)

    def summary_metrics(self, X, y, return_data=False):

        if self.mcmc is None:
            raise ValueError("Debes llamar a fit() antes.")

        # =========================================================
        # PREDICCIONES
        # =========================================================
        y_samples = self.predict_posterior_samples(X)
        y_plugin = self.predict_plugin(X)

        # =========================================================
        # NNLL
        # =========================================================
        nnll = jnp.asarray(self.normalized_neg_log_likelihood_samples(X, y))
        nnll_plugin = self.normalized_neg_log_likelihood_plugin(X, y)

        # =========================================================
        # MSE / MAE (posterior)
        # =========================================================
        mse_samples = jnp.mean((y_samples - y[:, None]) ** 2, axis=0)
        mae_samples = jnp.mean(jnp.abs(y_samples - y[:, None]), axis=0)

        # =========================================================
        # MSE / MAE (plugin)
        # =========================================================
        mse_plugin = jnp.mean((y_plugin - y) ** 2)
        mae_plugin = jnp.mean(jnp.abs(y_plugin - y))

        # =========================================================
        # TABLA
        # =========================================================
        data = {
            "NNLL": [
                jnp.mean(nnll),
                jnp.std(nnll),
                jnp.median(nnll),
                jnp.quantile(nnll, 0.05),
                jnp.quantile(nnll, 0.95),
                nnll_plugin,
            ],
            "MSE": [
                jnp.mean(mse_samples),
                jnp.std(mse_samples),
                jnp.median(mse_samples),
                jnp.quantile(mse_samples, 0.05),
                jnp.quantile(mse_samples, 0.95),
                mse_plugin,
            ],
            "MAE": [
                jnp.mean(mae_samples),
                jnp.std(mae_samples),
                jnp.median(mae_samples),
                jnp.quantile(mae_samples, 0.05),
                jnp.quantile(mae_samples, 0.95),
                mae_plugin,
            ],
        }

        columns = ["mean", "std", "median", "5%", "95%", "plugin"]

        df = pd.DataFrame.from_dict(data, orient="index", columns=columns)
        print("\nModel Summary Metrics")
        print("=" * 70)
        print(df)
        print("=" * 70)

        if return_data:
            return df

class BayesianLaplaceRegression(BayesianGLM):

    def __init__(self,
                 b_prior=None,
                 **kwargs):

        super().__init__(**kwargs)

        # scale parameter of Laplace (b > 0)
        self.b_prior = b_prior if b_prior is not None else dist.HalfNormal(1.0)

    # =========================================================
    # MODEL
    # =========================================================
    def model(self, X, y=None):

        n, p = X.shape

        beta = numpyro.sample(
            "beta",
            self.beta_prior.expand([p]).to_event(1)
        )

        b = numpyro.sample(
            "b",
            self.b_prior
        )
        with numpyro.plate("data", n):

            mu = jnp.dot(X, beta)

            numpyro.sample(
                "y",
                dist.Laplace(mu, b),
                obs=y
            )

    # =========================================================
    # LIKELIHOOD
    # =========================================================
    def neg_log_likelihood(self, X, y, params):

        beta = params["beta"]
        b = params.get("b", 1.0)

        mu = jnp.dot(X, beta)

        return -jnp.sum(dist.Laplace(mu, b).log_prob(y))

    # =========================================================
    # POSTERIOR PREDICTIVE SAMPLING
    # =========================================================
    def sample_likelihood(self, mu, params):

        self.rng_key, subkay = random.split(self.rng_key)

        b = params["b"]

        return dist.Laplace(mu, b).sample(subkay)

    # =========================================================
    # PLUG-IN PREDICTION (β̂)
    # =========================================================
    def predict_plugin(self, X):

        beta = self.get_beta_hat()
        return jnp.dot(X, beta)

    # =========================================================
    # POSTERIOR PREDICTIONS
    # =========================================================
    def predict_posterior_samples(self, X):

        beta = self.get_beta_samples()

        if beta.ndim == 1:
            beta = beta[None, :]

        return jnp.dot(X, beta.T)

    # =========================================================
    # METRICS SUMMARY
    # =========================================================
    def summary_metrics(self, X, y, return_data=False):

        if self.mcmc is None:
            raise ValueError("Debes llamar a fit() antes.")

        # =====================================================
        # PREDICCIONES
        # =====================================================
        y_samples = self.predict_posterior_samples(X)
        y_plugin = self.predict_plugin(X)

        # =====================================================
        # NNLL
        # =====================================================
        nnll = jnp.asarray(self.normalized_neg_log_likelihood_samples(X, y))
        nnll_plugin = self.normalized_neg_log_likelihood_plugin(X, y)

        # =====================================================
        # MSE / MAE (posterior samples)
        # =====================================================
        mse_samples = jnp.mean((y_samples - y[:, None]) ** 2, axis=0)
        mae_samples = jnp.mean(jnp.abs(y_samples - y[:, None]), axis=0)

        # =====================================================
        # MSE / MAE (plugin)
        # =====================================================
        mse_plugin = jnp.mean((y_plugin - y) ** 2)
        mae_plugin = jnp.mean(jnp.abs(y_plugin - y))

        # =====================================================
        # TABLA
        # =====================================================
        data = {
            "NNLL": [
                jnp.mean(nnll),
                jnp.std(nnll),
                jnp.median(nnll),
                jnp.quantile(nnll, 0.05),
                jnp.quantile(nnll, 0.95),
                nnll_plugin,
            ],
            "MSE": [
                jnp.mean(mse_samples),
                jnp.std(mse_samples),
                jnp.median(mse_samples),
                jnp.quantile(mse_samples, 0.05),
                jnp.quantile(mse_samples, 0.95),
                mse_plugin,
            ],
            "MAE": [
                jnp.mean(mae_samples),
                jnp.std(mae_samples),
                jnp.median(mae_samples),
                jnp.quantile(mae_samples, 0.05),
                jnp.quantile(mae_samples, 0.95),
                mae_plugin,
            ],
        }

        columns = ["mean", "std", "median", "5%", "95%", "plugin"]

        df = pd.DataFrame.from_dict(data, orient="index", columns=columns)
        df = df.round(4)

        print("\nLaplace Regression Summary Metrics")
        print("=" * 70)
        print(df)
        print("=" * 70)

        if return_data:
            return df

class BayesianBinaryGLM(BayesianGLM):

    def __init__(self,
                 inverse_link=None,
                 **kwargs):

        super().__init__(inverse_link=inverse_link, **kwargs)

    def model(self, X, y=None):

        n, p = X.shape

        beta = numpyro.sample(
            "beta",
            self.beta_prior.expand([p]).to_event(1)
        )


        with numpyro.plate("data", n):

            mu = self.inverse_link(jnp.dot(X, beta))

            numpyro.sample(
                "y",
                dist.Bernoulli(mu),
                obs=y
            )

    def neg_log_likelihood(self, X, y, params):


        beta = params["beta"]

        mu = self.inverse_link(jnp.dot(X, beta))

        return -jnp.sum(dist.Bernoulli(mu).log_prob(y))

    # =========================================================
    # POSTERIOR PREDICTIVE (optional override)
    # =========================================================

    def sample_likelihood(self, mu, params):

        self.rng_key, subkay = random.split(self.rng_key)

        return dist.Bernoulli(mu).sample(self.rng_key)

    def predict_posterior_class_samples(self, X, threshold=0.5):

        return (self.predict_posterior_samples(X) > threshold).astype(int)

    def predict_class_plugin(self, X, threshold=0.5):

        return (self.predict_plugin(X) > threshold).astype(int)

    def accuracy_samples(self, X, y, threshold=0.5):

        y_pred = self.predict_posterior_class_samples(X, threshold)

        return (y[:, None] == y_pred).mean(axis=0)

    def accuracy_plugin(self, X, y, threshold=0.5):

        y_pred = self.predict_class_plugin(X, threshold)

        return (y == y_pred).mean()


    def auc_samples(self, X, y):

        probs_samples = self.predict_posterior_samples(X)

        auc_samples = jnp.array([roc_auc_score(y, probs_samples[:, i]) for i in range(probs_samples.shape[1])])


        return auc_samples

    def auc_plugin(self, X, y):

        probs = self.predict_plugin(X)

        return roc_auc_score(y, probs)


    def plot_accuracy_density(self, X, y, threshold=0.5, figsize=(10,6), color="blue"):

        acc_samples = np.array(self.accuracy_samples(X, y, threshold))
        kde = gaussian_kde(acc_samples)
        x_grid = np.linspace(acc_samples.min(), acc_samples.max(), 200)

        plt.figure(figsize=figsize)
        plt.plot(x_grid, kde(x_grid), color=color)
        plt.title("Accuracy")
        plt.xlabel("Accuracy")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.show()


    def plot_auc_density(self, X, y, figsize=(10,6), color="blue"):

        auc_samples = np.array(self.auc_samples(X, y))
        kde = gaussian_kde(auc_samples)
        x_grid = np.linspace(auc_samples.min(), auc_samples.max(), 200)

        plt.figure(figsize=figsize)
        plt.plot(x_grid, kde(x_grid), color=color)
        plt.title("AUC Score")
        plt.xlabel("AUC Score")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.show()

    def summary_metrics(self, X, y, threshold=0.5):

        if self.mcmc is None:
            raise ValueError("Debes llamar a fit() antes.")

        acc_samples = self.accuracy_samples(X, y, threshold)
        acc_plugin = self.accuracy_plugin(X, y, threshold)

        # =====================================================
        # AUC (usamos sklearn)
        # =====================================================

        auc_samples = self.auc_samples(X, y)
        auc_plugin = self.auc_plugin(X, y)


        nnll_samples = self.normalized_neg_log_likelihood_samples(X, y)
        nnll_plugin = self.normalized_neg_log_likelihood_plugin(X, y)


        data = {
            "NNLL": [
                jnp.mean(nnll_samples),
                jnp.std(nnll_samples),
                jnp.median(nnll_samples),
                jnp.quantile(nnll_samples, 0.05),
                jnp.quantile(nnll_samples, 0.95),
                nnll_plugin,
            ],
            "AUC": [
                jnp.mean(auc_samples),
                jnp.std(auc_samples),
                jnp.median(auc_samples),
                jnp.quantile(auc_samples, 0.05),
                jnp.quantile(auc_samples, 0.95),
                auc_plugin,
            ],
            "ACC": [
                jnp.mean(acc_samples),
                jnp.std(acc_samples),
                jnp.median(acc_samples),
                jnp.quantile(acc_samples, 0.05),
                jnp.quantile(acc_samples, 0.95),
                acc_plugin,
            ],
        }

        columns = ["mean", "std", "median", "5%", "95%", "plugin"]

        df = pd.DataFrame.from_dict(data, orient="index", columns=columns)

        return df

class BayesianLogisticRegression(BayesianBinaryGLM):

    def __init__(self, **kwargs):

        super().__init__(inverse_link=jax.nn.sigmoid, **kwargs)

    def model(self, X, y=None):

        n, p = X.shape

        beta = numpyro.sample(
            "beta",
            self.beta_prior.expand([p]).to_event(1)
        )


        with numpyro.plate("data", n):

            eta = jnp.dot(X, beta)

            numpyro.sample(
                "y",
                dist.Bernoulli(logits=eta),
                obs=y
            )

class BayesianPoissonRegression(BayesianGLM):

    def __init__(self, inverse_link=jnp.exp, **kwargs):

        super().__init__(inverse_link=inverse_link, **kwargs)

    def model(self, X, y=None):

        n, p = X.shape

        beta = numpyro.sample(
            "beta",
            self.beta_prior.expand([p]).to_event(1)
        )


        with numpyro.plate("data", n):

            eta = jnp.dot(X, beta)
            mu = self.inverse_link(eta)
            numpyro.sample(
                "y",
                dist.Poisson(mu),
                obs=y
            )

    def neg_log_likelihood(self, X, y, params):

        beta = params["beta"]

        eta = jnp.dot(X, beta)
        mu = self.inverse_link(eta)

        return -jnp.sum(dist.Poisson(mu).log_prob(y))

    def sample_likelihood(self, mu, params):

        self.rng_key, subkay = random.split(self.rng_key)

        return dist.Poisson(mu).sample(self.rng_key)

class BayesianNegBinomRegression(BayesianGLM):

    def __init__(self, inverse_link=jnp.exp, alpha_prior = None,**kwargs):

        super().__init__(inverse_link=inverse_link, **kwargs)
        self.alpha_prior = alpha_prior if alpha_prior is not None else dist.HalfNormal(1.0)

    def model(self, X, y=None):

        n, p = X.shape

        beta = numpyro.sample(
            "beta",
            self.beta_prior.expand([p]).to_event(1)
        )

        alpha = numpyro.sample(
            "alpha",
            self.alpha_prior
        )

        with numpyro.plate("data", n):

            eta = jnp.dot(X, beta)
            mu = self.inverse_link(eta)

            numpyro.sample(
                "y",
                dist.NegativeBinomial2(mu, alpha),
                obs=y
            )

    def neg_log_likelihood(self, X, y, params):

        beta = params["beta"]
        alpha = params["alpha"]

        eta = jnp.dot(X, beta)
        mu = self.inverse_link(eta)

        return -jnp.sum(dist.NegativeBinomial2(mu, alpha).log_prob(y))

    def sample_likelihood(self, mu, params):

        self.rng_key, subkey = random.split(self.rng_key)

        return dist.NegativeBinomial2(mu, params["alpha"]).sample(subkey)

class BayesianGammaRegression(BayesianGLM):

    def __init__(self, inverse_link=jnp.exp, alpha_prior = None,**kwargs):

        super().__init__(inverse_link=inverse_link, **kwargs)
        self.alpha_prior = alpha_prior if alpha_prior is not None else dist.HalfNormal(1.0)

    def model(self, X, y=None):

        n, p = X.shape

        beta = numpyro.sample(
            "beta",
            self.beta_prior.expand([p]).to_event(1)
        )

        alpha = numpyro.sample(
            "alpha",
            self.alpha_prior
        )

        eta = jnp.dot(X, beta)
        mu = self.inverse_link(eta)

        with numpyro.plate("data", n):

            numpyro.sample(
                "y",
                dist.Gamma(alpha, alpha/mu),
                obs=y
            )

    def neg_log_likelihood(self, X, y, params):

        beta = params["beta"]
        alpha = params["alpha"]

        eta = jnp.dot(X, beta)
        mu = self.inverse_link(eta)

        return -jnp.sum(dist.Gamma(mu, alpha/mu).log_prob(y))

    def sample_likelihood(self, mu, params):

        self.rng_key, subkey = random.split(self.rng_key)
        alpha = params["alpha"]

        return dist.Gamma(alpha, alpha/mu).sample(subkey)

class BayesianMultinomialRegression(BayesianGLM):

    def __init__(self, K, **kwargs):

        self.K = K

        super().__init__(
            inverse_link=None,   # no aplica
            **kwargs
        )

        self.inverse_link = None

    # =========================================================
    # MODEL
    # =========================================================

    def model(self, X, y=None):

        n, p = X.shape

        beta = numpyro.sample(
            "beta",
            self.beta_prior.expand([self.K, p]).to_event(2)
        )

        logits = jnp.dot(X, beta.T)

        with numpyro.plate("data", n):
            numpyro.sample(
                "y",
                dist.Categorical(logits=logits),
                obs=y
            )

    # =========================================================
    # NEG LOG LIKELIHOOD
    # =========================================================

    def neg_log_likelihood(self, X, y, params):

        beta = params["beta"]

        logits = jnp.dot(X, beta.T)

        return -jnp.sum(
            dist.Categorical(logits=logits).log_prob(y)
        )

    # =========================================================
    # PREDICT (PLUGIN)
    # =========================================================

    def predict_plugin(self, X):

        beta = self.get_beta_hat()

        logits = jnp.dot(X, beta.T)

        return jax.nn.softmax(logits, axis=-1)

    # =========================================================
    # PREDICT (POSTERIOR)
    # =========================================================

    def predict_posterior_samples(self, X):

        beta = self.get_beta_samples()  # (S, K, p)

        logits = jnp.einsum("np,skp->snk", X, beta)

        return jax.nn.softmax(logits, axis=-1)

    # =========================================================
    # POSTERIOR PREDICTIVE
    # =========================================================

    def sample_posterior_predictive(self, X):

        beta = self.get_beta_samples()

        logits = jnp.einsum("np,skp->snk", X, beta)

        self.rng_key, subkey = random.split(self.rng_key)

        return dist.Categorical(logits=logits).sample(subkey)

    # =========================================================
    # CLASS PREDICTIONS
    # =========================================================

    def predict_class_plugin(self, X):

        probs = self.predict_plugin(X)
        return jnp.argmax(probs, axis=-1)

    def predict_class_posterior(self, X):

        probs = self.predict_posterior_samples(X)
        return jnp.argmax(probs, axis=-1)

    def accuracy_samples(self, X, y):

        y_pred = self.predict_class_posterior(X)  # (S, n)

        return (y[None, :] == y_pred).mean(axis=1)

    def accuracy_plugin(self, X, y):

        y_pred = self.predict_class_plugin(X)

        return (y == y_pred).mean()

    def plot_accuracy_density(self, X, y, figsize=(10,6), color="blue"):

        acc_samples = np.array(self.accuracy_samples(X, y))

        kde = gaussian_kde(acc_samples)
        x_grid = np.linspace(acc_samples.min(), acc_samples.max(), 200)

        plt.figure(figsize=figsize)
        plt.plot(x_grid, kde(x_grid), color=color)
        plt.title("Accuracy")
        plt.xlabel("Accuracy")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.show()

    def summary_metrics(self, X, y):

        if self.mcmc is None:
            raise ValueError("Debes llamar a fit() antes.")

        acc_samples = self.accuracy_samples(X, y)
        acc_plugin = self.accuracy_plugin(X, y)

        nnll_samples = self.normalized_neg_log_likelihood_samples(X, y)
        nnll_plugin = self.normalized_neg_log_likelihood_plugin(X, y)

        data = {
            "NNLL": [
                jnp.mean(nnll_samples),
                jnp.std(nnll_samples),
                jnp.median(nnll_samples),
                jnp.quantile(nnll_samples, 0.05),
                jnp.quantile(nnll_samples, 0.95),
                nnll_plugin,
            ],
            "ACC": [
                jnp.mean(acc_samples),
                jnp.std(acc_samples),
                jnp.median(acc_samples),
                jnp.quantile(acc_samples, 0.05),
                jnp.quantile(acc_samples, 0.95),
                acc_plugin,
            ],
        }

        columns = ["mean", "std", "median", "5%", "95%", "plugin"]

        df = pd.DataFrame.from_dict(data, orient="index", columns=columns)

        return df

