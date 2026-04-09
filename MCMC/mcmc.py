import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import math


class BayesianGLM:

    def __init__(self, X, y):

        self.X = X
        self.y = y

    def log_posterior(self, **params):
        raise NotImplementedError

    def init_params(self, n_chains):
        raise NotImplementedError

    def _store_samples(self, samples):
        raise NotImplementedError

    def fit_mh(self, n_iter=1000, burn_in=0, n_chains=1, step_size=0.1):

        params = self.init_params(n_chains)

        samples = {
            k: torch.zeros((n_iter, *v.shape))
            for k, v in params.items()
        }

        for i in range(n_iter):

            props = {
                k: v + step_size * torch.randn_like(v)
                for k, v in params.items()
            }

            logp_current = self.log_posterior(**params)
            logp_prop = self.log_posterior(**props)

            log_alpha = logp_prop - logp_current
            alpha = torch.minimum(torch.zeros_like(log_alpha), log_alpha)

            u = torch.log(torch.rand_like(alpha))
            accept = u <= alpha

            for k in params:

                if params[k].dim() == 2:
                    params[k] = torch.where(accept[:, None], props[k], params[k])
                else:
                    params[k] = torch.where(accept, props[k], params[k])

                samples[k][i] = params[k]

        for k in samples:
            samples[k] = samples[k][burn_in:]

        return samples

    def fit(self, n_iter=1000, burn_in=0, n_chains=1, step_size=0.1):

        samples = self.fit_mh(n_iter, burn_in, n_chains, step_size)
        self._store_samples(samples)

    def deviance_samples(self, X, y, **params):
        raise NotImplementedError

    def deviance_mean(self, X, y):
        return self.deviance_samples(X, y).mean()

    def deviance_ci(self, X, y, alpha=0.05):

        dev = self.deviance_samples(X, y)
        dev_flat = dev.reshape(-1)

        lower = torch.quantile(dev_flat, alpha/2)
        upper = torch.quantile(dev_flat, 1 - alpha/2)

        return lower, upper

    def plot_deviance_density(self, X, y, figsize=(6, 4)):

        dev = self.deviance_samples(X, y)
        dev_flat = dev.reshape(-1).numpy()

        df = pd.DataFrame(dev_flat, columns=['deviance'])

        df.plot.density(figsize=figsize, title='Posterior Density of Deviance')
        plt.xlabel('Deviance')
        plt.show()

    def plot_beta_traces(self, figsize=(10, 6), layout=None):

        n_iter, n_chains, d = self.beta_samples.shape

        if layout is None:
            layout = (d, 1)

        fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize, sharex=True)
        axes = np.array(axes).reshape(-1)

        for j in range(d):

            data = {}

            for c in range(n_chains):
                data[f'chain_{c}'] = self.beta_samples[:, c, j].numpy()

            df = pd.DataFrame(data)
            df.plot(ax=axes[j], title=f'Beta_{j}')

        plt.tight_layout()
        plt.show()

    def plot_betas_densities(self, figsize=(6, 8), layout=None):

        if layout is None:
            layout = (self.d, 1)

        columns = [f'Beta_{i}' for i in range(self.d)]

        data = pd.DataFrame(
            self.beta_samples.reshape(-1, self.d).numpy(),
            columns=columns
        )

        data.plot.density(
            figsize=figsize,
            subplots=True,
            title='Betas Densities',
            sharey=False,
            sharex=False,
            layout=layout
        )

        plt.show()


class BayesianLinearRegression(BayesianGLM):

    def __init__(self, X, y, tau2=1.0, a=1.0, b=1.0):

        super().__init__(X, y)

        self.tau2 = tau2
        self.a = a
        self.b = b

        self.n, self.d = self.X.shape

        self.beta_prior = torch.distributions.MultivariateNormal(
            torch.zeros(self.d),
            self.tau2 * torch.eye(self.d)
        )

    def init_params(self, n_chains):
        return {
            "beta": torch.zeros((n_chains, self.d)),
            "eta": torch.zeros(n_chains)
        }

    def log_posterior(self, beta, eta):

        sigma2 = torch.exp(eta)

        z = self.X @ beta.mT

        like = torch.distributions.Normal(z, torch.sqrt(sigma2))
        log_like = like.log_prob(self.y[:, None]).sum(0)

        log_beta_prior = self.beta_prior.log_prob(beta)

        sigma2_prior = torch.distributions.InverseGamma(self.a, self.b)
        log_sigma2_prior = sigma2_prior.log_prob(sigma2)

        return log_like + log_beta_prior + log_sigma2_prior + eta

    def _store_samples(self, samples):

        self.beta_samples = samples["beta"]
        self.eta_samples = samples["eta"]
        self.sigma2_samples = torch.exp(self.eta_samples)

        self.beta = self.beta_samples.mean(dim=(0, 1))
        self.sigma2 = self.sigma2_samples.mean()

    def predict_samples(self, X):

        mu = self.beta_samples @ X.mT
        noise = torch.randn_like(mu) * torch.sqrt(self.sigma2_samples[:, :, None])

        return mu + noise

    def deviance_samples(self, X, y):

        y_pred = self.predict_samples(X)
        y = y[None, None, :]

        dev = (y - y_pred) ** 2

        return dev.mean(dim=2)

    def plot_sigma2_traces(self, figsize=(10, 6)):

        n_iter, n_chains = self.sigma2_samples.shape

        data = {}

        for c in range(n_chains):
            data[f'chain_{c}'] = self.sigma2_samples[:, c].numpy()

        df = pd.DataFrame(data)

        df.plot(figsize=figsize, title='Trace sigma2')
        plt.show()

    def plot_sigma2_density(self, figsize=(6, 4)):

        sigma2_flat = self.sigma2_samples.reshape(-1).numpy()

        df = pd.DataFrame(sigma2_flat, columns=['sigma2'])

        df.plot.density(figsize=figsize, title='Posterior Density of sigma2')
        plt.xlabel('sigma2')
        plt.show()


class BayesianLogisticRegression(BayesianGLM):

    def __init__(self, X, y, tau2=1.0):

        super().__init__(X, y)

        self.tau2 = tau2

        self.n, self.d = self.X.shape

        self.beta_prior = torch.distributions.MultivariateNormal(
            torch.zeros(self.d),
            self.tau2 * torch.eye(self.d)
        )

    def init_params(self, n_chains):
        return {
            "beta": torch.zeros((n_chains, self.d))
        }

    def log_posterior(self, beta):

        z = self.X @ beta.mT

        log_like = torch.distributions.Bernoulli(logits=z).log_prob(
            self.y[:, None]
        ).sum(0)

        log_prior = self.beta_prior.log_prob(beta)

        return log_like + log_prior

    def _store_samples(self, samples):

        self.beta_samples = samples["beta"]
        self.beta = self.beta_samples.mean(dim=(0, 1))

    def predict_proba_samples(self, X):

        z = self.beta_samples @ X.mT
        return torch.sigmoid(z)

    def predict_samples(self, X, threshold=0.5):

        probs = self.predict_proba_samples(X)
        return (probs >= threshold).float()

    def accuracy_samples(self, X, y, threshold=0.5):

        y_pred = self.predict_samples(X, threshold)
        return (y[None, None, :] == y_pred).float().mean(dim=2)

    def plot_accuracy_density(self, X, y, figsize=(6, 4)):
        acc = self.accuracy_samples(X, y)
        acc_flat = acc.reshape(-1).numpy()

        df = pd.DataFrame(acc_flat, columns=['accuracy'])

        df.plot.density(figsize=figsize, title='Posterior Density of Accuracy')
        plt.xlabel('Accuracy')
        plt.show()

    def accuracy_ci(self, X, y, alpha=0.05):
        acc = self.accuracy_samples(X, y)

        lower = torch.quantile(acc, alpha / 2)
        upper = torch.quantile(acc, 1 - alpha / 2)

        return lower, upper

    def accuracy_summary(self, X, y, alpha=0.05):
        mean = self.accuracy_mean(X, y)
        lower, upper = self.accuracy_ci(X, y, alpha)

        print(f'Accuracy mean: {mean:.4f}')
        print(f'{100 * (1 - alpha):.1f}% CI: [{lower:.4f}, {upper:.4f}]')

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

    def plot_auc_density(self, X, y, figsize=(6, 4)):
        auc = self.auc_samples(X, y)
        auc_flat = auc.reshape(-1).numpy()

        df = pd.DataFrame(auc_flat, columns=['AUC'])

        df.plot.density(figsize=figsize, title='Posterior Density of AUC')
        plt.xlabel('AUC')
        plt.show()

    def auc_ci(self, X, y, alpha=0.05):
        auc = self.auc_samples(X, y)

        lower = torch.quantile(auc, alpha / 2)
        upper = torch.quantile(auc, 1 - alpha / 2)

        return lower, upper

    def auc_summary(self, X, y, alpha=0.05):
        mean = self.auc_mean(X, y)
        lower, upper = self.auc_ci(X, y, alpha)

        print(f'AUC mean: {mean:.4f}')
        print(f'{100 * (1 - alpha):.1f}% CI: [{lower:.4f}, {upper:.4f}]')

    def deviance_samples(self, X, y, eps=1e-8):

        probs = self.predict_proba_samples(X)
        y = y[None, None, :]

        probs = torch.clamp(probs, eps, 1 - eps)

        dev = -(y * torch.log(probs) + (1 - y) * torch.log(1 - probs))

        return dev.mean(dim=2)


class BayesianPoissonRegression(BayesianGLM):

    def __init__(self, X, y, tau2=1.0):

        super().__init__(X, y)

        self.tau2 = tau2

        self.n, self.d = self.X.shape

        self.beta_prior = torch.distributions.MultivariateNormal(
            torch.zeros(self.d),
            self.tau2 * torch.eye(self.d)
        )

    def init_params(self, n_chains):
        return {
            "beta": torch.zeros((n_chains, self.d))
        }

    def log_posterior(self, beta):

        z = self.X @ beta.mT

        log_like = torch.distributions.Poisson(
            rate=torch.exp(z)
        ).log_prob(self.y[:, None]).sum(0)

        log_prior = self.beta_prior.log_prob(beta)

        return log_like + log_prior

    def _store_samples(self, samples):

        self.beta_samples = samples["beta"]
        self.beta = self.beta_samples.mean(dim=(0, 1))

    def predict_samples(self, X):

        z = self.beta_samples @ X.mT
        return torch.exp(z)

    def deviance_samples(self, X, y, eps=1e-8):

        y_pred = self.predict_samples(X)
        y = y[None, None, :]

        y_pred = torch.clamp(y_pred, eps)
        z = torch.clamp(y / y_pred, eps)

        dev = (y * torch.log(z) - (y - y_pred))

        return dev.mean(dim=2)


class BayesianGammaRegression(BayesianGLM):

    def __init__(self, X, y, tau2=1.0, a=1.0, b=1.0):

        super().__init__(X, y)

        self.tau2 = tau2

        self.n, self.d = self.X.shape

        self.beta_prior = torch.distributions.MultivariateNormal(
            torch.zeros(self.d),
            self.tau2 * torch.eye(self.d)
        )

        self.alpha_prior = torch.distributions.Gamma(a, b)

    def init_params(self, n_chains):
        return {
            "beta": torch.zeros((n_chains, self.d)),
            "eta": torch.zeros(n_chains)
        }

    def log_posterior(self, beta, eta):

        alpha = torch.exp(eta)

        z = beta @ self.X.T
        mu = torch.exp(z)

        rate = alpha[:, None] / mu

        log_like = torch.distributions.Gamma(
            concentration=alpha[:, None],
            rate=rate
        ).log_prob(self.y[None, :]).sum(1)

        log_prior_beta = self.beta_prior.log_prob(beta)
        log_prior_alpha = self.alpha_prior.log_prob(alpha)

        return log_like + log_prior_beta + log_prior_alpha + eta

    def _store_samples(self, samples):

        self.beta_samples = samples["beta"]
        self.eta_samples = samples["eta"]
        self.alpha_samples = torch.exp(self.eta_samples)

        self.beta = self.beta_samples.mean(dim=(0, 1))
        self.alpha = self.alpha_samples.mean()

    def predict_samples(self, X):

        z = self.beta_samples @ X.mT
        return torch.exp(z)

    def deviance_samples(self, X, y, eps=1e-8):

        y_pred = self.predict_samples(X)
        y = y[None, None, :]

        y_pred = torch.clamp(y_pred, eps)
        z = torch.clamp(y / y_pred, eps)

        dev = ((y - y_pred) / y_pred - torch.log(z))

        return dev.mean(dim=2)

    def plot_alpha_traces(self, figsize=(10, 6)):

        n_iter, n_chains = self.alpha_samples.shape

        data = {}

        for c in range(n_chains):
            data[f'chain_{c}'] = self.alpha_samples[:, c].numpy()

        df = pd.DataFrame(data)

        df.plot(figsize=figsize, title='Trace alpha')
        plt.show()

    def plot_alpha_density(self, figsize=(6, 4)):

        alpha_flat = self.alpha_samples.reshape(-1).numpy()

        df = pd.DataFrame(alpha_flat, columns=['alpha'])

        df.plot.density(figsize=figsize, title='Posterior Density of alpha')
        plt.xlabel('alpha')
        plt.show()

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

