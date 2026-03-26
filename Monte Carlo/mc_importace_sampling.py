"""
Módulo de Muestreo por Importancia y Valoración de Opciones

Este módulo proporciona clases para implementar técnicas de muestreo por importancia
en el contexto de probabilidades de cola, funciones generadoras de momentos,
valoración de opciones financieras y regresión logística Bayesiana.
"""

import torch


class RestrictedExponential(torch.distributions.Distribution):
    """
    Distribución Exponencial Restringida

    Implementa una distribución exponencial desplazada que solo toma valores
    mayores o iguales a un parámetro de ubicación (loc). Útil para muestreo
    por importancia en cálculos de probabilidades de cola.
    """

    def __init__(self, loc, scale):
        """
        Inicializa la distribución exponencial restringida.

        Parámetros
        ----------
        loc : float
            Parámetro de ubicación (valor mínimo de la distribución).
        scale : float
            Parámetro de escala de la exponencial.
        """
        self.loc = loc
        self.scale = scale

    def sample(self, sample_shape=torch.Size()):
        """
        Genera muestras de la distribución exponencial restringida.

        Parámetros
        ----------
        sample_shape : torch.Size, opcional
            Forma de las muestras a generar.

        Retorna
        -------
        torch.Tensor
            Muestras de la distribución, todas mayores o iguales a loc.
        """
        # Transformada inversa para la exponencial desplazada
        u = torch.rand(sample_shape)
        x = -torch.log(u) + self.loc
        return x

    def log_prob(self, x):
        """
        Calcula la densidad logarítmica de la distribución.

        Parámetros
        ----------
        x : torch.Tensor
            Puntos donde evaluar la densidad.

        Retorna
        -------
        torch.Tensor
            Log-verosimilitud de la distribución evaluada en x.
        """
        # Densidad es cero para x < loc
        y = torch.where(x < self.loc, torch.zeros_like(x), torch.log(torch.exp(self.loc - x)))
        return y


class TailNormal:
    """
    Estimación de Probabilidades de Cola Normal mediante Muestreo por Importancia

    Estima P(Z > x) para una distribución normal estándar utilizando muestreo
    por importancia con una distribución exponencial restringida como propuesta.
    """

    def __init__(self):
        """Inicializa el estimador de probabilidades de cola normal."""
        self.Z = torch.distributions.Normal(0.0, 1.0)  # Distribución normal estándar

    def importance_sampling(self, x, n=1000000, decimals=10):
        """
        Estima P(Z > x) mediante muestreo por importancia.

        Utiliza una distribución exponencial restringida con loc = x como
        distribución propuesta, que asigna mayor peso a la región de cola.

        Parámetros
        ----------
        x : float
            Umbral para la probabilidad de cola.
        n : int, opcional
            Número de muestras. Por defecto 1,000,000.
        decimals : int, opcional
            Número de decimales para mostrar. Por defecto 10.

        Retorna
        -------
        tuple
            (probabilidad_estimada, error_absoluto, varianza)

        Notas
        -----
        El estimador se calcula como: μ = (1/n) Σ h(y_i) * f(y_i)/g(y_i)
        donde h(y) = 1{y > x}, f es la densidad normal, g es la densidad exponencial.
        """
        # Distribución propuesta: exponencial desplazada a x
        re = RestrictedExponential(x, 1)
        y = re.sample((n,))

        # Indicador de cola
        h = torch.where(y < x, torch.zeros_like(y), torch.ones_like(y))

        # Densidad objetivo (normal) y densidad propuesta (exponencial)
        f = self.Z.log_prob(y).exp()
        g = re.log_prob(y).exp()

        # Estimador por importancia
        mu = torch.mean(h * f / g)

        # Valor verdadero (analítico)
        real = 1 - self.Z.cdf(x)

        # Cálculo de error y varianza
        error = torch.abs(real - mu)
        var = torch.var(h * f / g) / n

        if mu.numel() == 1:
            print(f"Probabilidad estimada : {mu.item():.{decimals}f}")
            print(f"Error: {error.item():.{decimals}f}")
            print(f"Varianza: {var.item():.{decimals}f}")

        return mu, error, var

    def monte_carlo(self, x, n=1000, decimals=10):
        """
        Estima P(Z > x) mediante Monte Carlo estándar.

        Parámetros
        ----------
        x : float
            Umbral para la probabilidad de cola.
        n : int, opcional
            Número de muestras. Por defecto 1000.
        decimals : int, opcional
            Número de decimales para mostrar. Por defecto 10.

        Retorna
        -------
        tuple
            (probabilidad_estimada, error_absoluto, varianza)
        """
        # Generar muestras normales estándar
        y = self.Z.sample((n,))

        # Indicador de cola
        h = torch.where(y < x, torch.zeros_like(x), torch.ones_like(x))

        # Estimador Monte Carlo
        mu = torch.mean(h)

        # Valor verdadero
        real = 1 - self.Z.cdf(x)

        # Cálculo de error y varianza
        error = torch.abs(real - mu)
        var = torch.var(h, unbiased=True) / n

        if mu.numel() == 1:
            print(f"Probabilidad estimada : {mu.item():.{decimals}f}")
            print(f"Error: {error.item():.{decimals}f}")
            print(f"Varianza: {var.item():.{decimals}f}")

        return mu, error, var


class NormalMGF:
    """
    Estimación de la Función Generadora de Momentos (FGM) de una Normal

    Estima M(t) = E[e^{tX}] para X ~ N(μ, σ²) mediante Monte Carlo estándar
    y muestreo por importancia.
    """

    def __init__(self, loc, scale):
        """
        Inicializa el estimador de la función generadora de momentos.

        Parámetros
        ----------
        loc : float
            Media de la distribución normal.
        scale : float
            Desviación estándar de la distribución normal.
        """
        self.loc = loc
        self.scale = scale
        self.var = scale ** 2

        self.Z_f = torch.distributions.Normal(loc, scale)  # Distribución objetivo
        self.Z_g = None  # Distribución propuesta (se define en importance_sampling)

    def importance_sampling(self, t, n=1000000, decimals=10):
        """
        Estima M(t) mediante muestreo por importancia.

        Utiliza una distribución normal con media desplazada como propuesta:
        g(x) ~ N(μ + σ²t, σ²)

        Parámetros
        ----------
        t : float
            Punto donde evaluar la FGM.
        n : int, opcional
            Número de muestras. Por defecto 1,000,000.
        decimals : int, opcional
            Número de decimales para mostrar. Por defecto 10.

        Retorna
        -------
        tuple
            (fgm_estimada, varianza, error_absoluto)

        Notas
        -----
        La distribución propuesta óptima para este problema es una normal
        con media μ + σ²t, que hace que el estimador tenga varianza cero.
        """
        # Distribución propuesta óptima
        self.Z_g = torch.distributions.Normal(self.loc + self.var * t, self.scale)

        # Generar muestras de la propuesta
        x = self.Z_g.sample((n,))

        # Función a integrar y pesos de importancia
        h = torch.exp(x * t)
        f = self.Z_f.log_prob(x).exp()
        g = self.Z_g.log_prob(x).exp()

        # Estimador por importancia
        mu = torch.mean(h * f / g)
        var = torch.var(h * f / g) / n

        # Valor verdadero analítico: M(t) = exp(μt + σ²t²/2)
        real = torch.exp(torch.Tensor([self.loc * t + self.var * (t ** 2) / 2]))

        error = torch.abs(real - mu)

        if mu.numel() == 1:
            print(f"FGM estimada : {mu.item():.{decimals}f}")
            print(f"Error: {error.item():.{decimals}f}")
            print(f"Varianza: {var.item():.{decimals}f}")

        return mu, var, error

    def monte_carlo(self, t, n=1000000, decimals=10):
        """
        Estima M(t) mediante Monte Carlo estándar.

        Parámetros
        ----------
        t : float
            Punto donde evaluar la FGM.
        n : int, opcional
            Número de muestras. Por defecto 1,000,000.
        decimals : int, opcional
            Número de decimales para mostrar. Por defecto 10.

        Retorna
        -------
        tuple
            (fgm_estimada, varianza, error_absoluto)
        """
        # Generar muestras de la distribución objetivo
        x = self.Z_f.sample((n,))

        # Función a integrar
        h = torch.exp(x * t)

        # Estimador Monte Carlo
        mu = torch.mean(h)
        var = torch.var(h) / n

        # Valor verdadero
        real = torch.exp(torch.Tensor([self.loc * t + self.var * (t ** 2) / 2]))

        error = torch.abs(real - mu)

        if mu.numel() == 1:
            print(f"FGM estimada : {mu.item():.{decimals}f}")
            print(f"Error: {error.item():.{decimals}f}")
            print(f"Varianza: {var.item():.{decimals}f}")

        return mu, var, error


class CallOptionPrice:
    """
    Valoración de Opciones de Compra (Call) mediante el Modelo Black-Scholes

    Implementa la fórmula analítica de Black-Scholes y estimaciones mediante
    Monte Carlo estándar y muestreo por importancia.
    """

    def __init__(self, s_0, r, sigma, k):
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
        """
        self.s_0 = torch.Tensor([s_0])
        self.r = torch.Tensor([r])
        self.sigma = torch.Tensor([sigma])
        self.k = torch.Tensor([k])
        self.Z = torch.distributions.Normal(0.0, 1.0)  # Normal estándar
        self.price = None  # Almacena el precio analítico

    def compute_price(self, t):
        """
        Calcula el precio analítico de la opción call mediante Black-Scholes.

        Parámetros
        ----------
        t : float
            Tiempo hasta vencimiento (en años).
        """
        t = torch.Tensor([t])

        # Parámetros d1 y d2 de Black-Scholes
        d1 = (torch.log(self.s_0 / self.k) + (self.r + self.sigma ** 2 / 2) * t) / (self.sigma * torch.sqrt(t))
        d2 = d1 - self.sigma * torch.sqrt(t)

        # Fórmula de Black-Scholes
        self.price = self.s_0 * self.Z.cdf(d1) - self.k * torch.exp(-self.r * t) * self.Z.cdf(d2)

    def compute_price_mc(self, t, n=10000, decimals=10):
        """
        Estima el precio mediante Monte Carlo estándar.

        Simula trayectorias del activo bajo la medida neutral al riesgo:
        S_T = S_0 * exp((r - σ²/2)T + σ√T Z)

        Parámetros
        ----------
        t : float
            Tiempo hasta vencimiento.
        n : int, opcional
            Número de simulaciones. Por defecto 10,000.
        decimals : int, opcional
            Número de decimales para mostrar. Por defecto 10.

        Retorna
        -------
        tuple
            (precio_estimado, error_absoluto, varianza)
        """
        # Calcular precio analítico para referencia
        self.compute_price(t)
        t = torch.Tensor([t])

        # Simular el activo al vencimiento
        z = self.Z.sample((n,))
        s_t = self.s_0 * torch.exp((self.r - self.sigma ** 2 / 2) * t + self.sigma * torch.sqrt(t) * z)

        # Payoff de la opción call
        payoff = torch.relu(s_t - self.k)

        # Precio estimado (valor esperado descontado)
        price_hat = torch.exp(-self.r * t) * payoff.mean()

        # Varianza del estimador
        var = torch.exp(-2 * self.r * t) * payoff.var(unbiased=False) / n

        # Error absoluto respecto al precio analítico
        error = torch.abs(self.price - price_hat)

        print("----- Monte Carlo -----")
        print(f"Precio  : {self.price.item():.{decimals}f}")
        print(f"Precio estimado : {price_hat.item():.{decimals}f}")
        print(f"Error: {error.item():.{decimals}f}")
        print(f"Varianza: {var.item():.{decimals}f}")

        return price_hat, error, var

    def compute_price_is(self, t, theta=None, n=10000, decimals=10):
        """
        Estima el precio mediante muestreo por importancia.

        Cambia la media de la distribución de Z para muestrear más eficientemente
        en regiones donde el payoff es positivo.

        Parámetros
        ----------
        t : float
            Tiempo hasta vencimiento.
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
        La distribución propuesta es N(θ, 1) para Z. Los pesos de importancia
        se calculan como w = exp(-θz + θ²/2).
        """
        t = torch.Tensor([t])

        # Calcular theta óptimo si no se proporciona
        if theta is None:
            theta = (torch.log(self.k / self.s_0) - (self.r - self.sigma ** 2 / 2) * t) / (self.sigma * torch.sqrt(t))
        else:
            theta = torch.Tensor([theta])

        # Calcular precio analítico
        d1 = (torch.log(self.s_0 / self.k) + (self.r + self.sigma ** 2 / 2) * t) / (self.sigma * torch.sqrt(t))
        d2 = d1 - self.sigma * torch.sqrt(t)
        self.price = self.s_0 * self.Z.cdf(d1) - self.k * torch.exp(-self.r * t) * self.Z.cdf(d2)

        # Muestrear de la distribución propuesta (normal desplazada)
        z = self.Z.sample((n,)) + theta

        # Simular el activo con las muestras desplazadas
        s_t = self.s_0 * torch.exp((self.r - self.sigma ** 2 / 2) * t + self.sigma * torch.sqrt(t) * z)

        # Payoff de la opción
        payoff = torch.relu(s_t - self.k)

        # Pesos de importancia
        weights = torch.exp(-theta * z + theta ** 2 / 2)

        # Estimador por importancia
        estimator = payoff * weights
        price_hat = torch.exp(-self.r * t) * estimator.mean()

        # Varianza del estimador
        var = torch.exp(-2 * self.r * t) * estimator.var(unbiased=False) / n

        # Error absoluto
        error = torch.abs(self.price - price_hat)

        print("----- Importance Sampling -----")
        print(f"Precio  : {self.price.item():.{decimals}f}")
        print(f"Precio estimado : {price_hat.item():.{decimals}f}")
        print(f"Error : {error.item():.{decimals}f}")
        print(f"Varianza : {var.item():.{decimals}f}")

        return price_hat, error, var


class NISLogisticRegression:
    """
    Regresión Logística Bayesiana mediante Muestreo por Importancia

    Implementa inferencia Bayesiana para regresión logística utilizando
    muestreo por importancia (Normal Importance Sampling - NIS) con una
    propuesta normal multivariante.
    """

    def __init__(self, X, y, prior_var=1.0):
        """
        Inicializa el modelo de regresión logística Bayesiana.

        Parámetros
        ----------
        X : torch.Tensor
            Matriz de características de forma (n_samples, n_features).
        y : torch.Tensor
            Vector de etiquetas binarias de forma (n_samples,).
        prior_var : float, opcional
            Varianza a priori para los coeficientes (distribución normal
            multivariante con media cero). Por defecto 1.0.
        """
        self.X = X
        self.y = y
        self.prior_var = prior_var
        self.n, self.d = self.X.shape  # n: muestras, d: características

        # Prior normal multivariante para los coeficientes β
        self.prior = torch.distributions.MultivariateNormal(
            torch.zeros(self.d),
            prior_var * torch.eye(self.d)
        )

        self.beta_hat = None  # Estimación posterior de los coeficientes
        self.samples = None   # Muestras de la distribución propuesta
        self.weights = None   # Pesos de importancia

    def log_likelihood(self, beta, epsilon=1e-12):
        """
        Calcula la log-verosimilitud de los datos dado el vector de coeficientes β.

        Parámetros
        ----------
        beta : torch.Tensor
            Vector de coeficientes de forma (d,) o (n_samples, d).
        epsilon : float, opcional
            Pequeña constante para evitar log(0). Por defecto 1e-12.

        Retorna
        -------
        torch.Tensor
            Log-verosimilitud de los datos.
        """
        # Calcular el logit: z = Xβ
        z = self.X @ beta.T

        # Probabilidades mediante función sigmoide
        p = torch.sigmoid(z)

        # Log-verosimilitud: Σ [y_i log(p_i) + (1-y_i) log(1-p_i)]
        ll = (self.y[:, None] * torch.log(p + epsilon) +
              (1 - self.y[:, None]) * torch.log(1 - p + epsilon)).sum(dim=0)

        return ll

    def log_f_kernel(self, beta, epsilon=1e-12):
        """
        Calcula el logaritmo del núcleo de la distribución posterior
        (no normalizado): log(prior) + log(verosimilitud).

        Parámetros
        ----------
        beta : torch.Tensor
            Vector de coeficientes.
        epsilon : float, opcional
            Constante para evitar log(0). Por defecto 1e-12.

        Retorna
        -------
        torch.Tensor
            Log del núcleo posterior.
        """
        return self.log_likelihood(beta, epsilon) - self.prior.log_prob(beta)

    def importance_sampling(self, n=10000, proposal_var=1.0, epsilon=1e-12):
        """
        Realiza inferencia Bayesiana mediante muestreo por importancia.

        Utiliza una distribución propuesta normal multivariante con media cero
        y varianza proposal_var.

        Parámetros
        ----------
        n : int, opcional
            Número de muestras. Por defecto 10,000.
        proposal_var : float, opcional
            Varianza de la distribución propuesta. Por defecto 1.0.
        epsilon : float, opcional
            Constante para evitar log(0). Por defecto 1e-12.

        Retorna
        -------
        torch.Tensor
            Estimación posterior de los coeficientes β (media ponderada).

        Notas
        -----
        Los pesos de importancia se calculan mediante la función softmax
        de los log-pesos para estabilidad numérica.
        """
        # Distribución propuesta: normal multivariante con media cero
        proposal = torch.distributions.MultivariateNormal(
            torch.zeros(self.d),
            proposal_var * torch.eye(self.d)
        )

        # Generar muestras de la propuesta
        samples = proposal.sample((n,))

        # Calcular log-pesos: log(posterior) - log(propuesta)
        log_weights = self.log_f_kernel(samples, epsilon) - proposal.log_prob(samples)

        # Normalizar pesos usando softmax para estabilidad numérica
        weights = torch.softmax(log_weights, dim=0)

        # Estimar la media posterior
        beta_hat = weights @ samples

        # Almacenar resultados
        self.beta_hat = beta_hat
        self.samples = samples
        self.weights = weights

        return beta_hat

    def predict_proba(self, X_new):
        """
        Predice probabilidades para nuevos datos utilizando la media posterior.

        Parámetros
        ----------
        X_new : torch.Tensor
            Nuevas características de forma (m, d).

        Retorna
        -------
        torch.Tensor
            Probabilidades predichas de pertenecer a la clase 1.
        """
        z = X_new @ self.beta_hat
        return torch.sigmoid(z)

    def predict_class(self, X_new, alpha=0.5):
        """
        Predice clases para nuevos datos utilizando un umbral.

        Parámetros
        ----------
        X_new : torch.Tensor
            Nuevas características de forma (m, d).
        alpha : float, opcional
            Umbral de decisión. Por defecto 0.5.

        Retorna
        -------
        torch.Tensor
            Clases predichas (0 o 1).
        """
        probs = self.predict_proba(X_new)
        return (probs >= alpha).int()

    def accuracy(self, X_test, y_test, alpha=0.5):
        """
        Calcula la precisión del modelo en un conjunto de prueba.

        Parámetros
        ----------
        X_test : torch.Tensor
            Características de prueba.
        y_test : torch.Tensor
            Etiquetas verdaderas de prueba.
        alpha : float, opcional
            Umbral de decisión. Por defecto 0.5.

        Retorna
        -------
        torch.Tensor
            Precisión del modelo (proporción de predicciones correctas).
        """
        y_pred = self.predict_class(X_test, alpha)
        return (y_pred == y_test).float().mean()