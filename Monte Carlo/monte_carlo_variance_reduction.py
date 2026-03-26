
"""
Módulo de Estimación de π y Reducción de Varianza

Este módulo proporciona clases para estimar π utilizando diferentes técnicas
de reducción de varianza: Monte Carlo básico, variable de control,
muestreo condicional, estratificado y variables antitéticas.
"""

import math
import torch
import matplotlib.pyplot as plt


class PiMonteCarlo:
    """
    Clase para la estimación de π mediante métodos Monte Carlo con reducción de varianza.

    Implementa cuatro técnicas diferentes para estimar π a partir del área de un círculo
    unitario inscrito en un cuadrado de lado 2:
    - Monte Carlo estándar
    - Variable de control
    - Monte Carlo condicional
    - Muestreo estratificado
    """

    def __init__(self, device="cpu"):
        """
        Inicializa el estimador de π.

        Parámetros
        ----------
        device : str, opcional
            Dispositivo para realizar los cálculos ('cpu' o 'cuda'). Por defecto 'cpu'.
        """
        self.device = device
        self.U = torch.distributions.Uniform(0, 1)  # Distribución uniforme en [0,1]

    def compute_pi(self, n=10000, decimals=4):
        """
        Estimador Monte Carlo básico para π.

        Utiliza la relación área del círculo / área del cuadrado = π/4,
        estimando π como: X = 4 * 1{U1² + U2² ≤ 1}

        Parámetros
        ----------
        n : int, opcional
            Número de muestras. Por defecto 10000.
        decimals : int, opcional
            Número de decimales para mostrar. Por defecto 4.

        Retorna
        -------
        tuple
            (estimación_pi, varianza_estimada)
        """
        # Generar muestras uniformes en [0,1]²
        U = self.U.sample((n, 2)).to(self.device)

        # Indicador de punto dentro del círculo de radio 1
        indicator = (U[:, 0] ** 2 + U[:, 1] ** 2 <= 1).float()
        X = 4 * indicator  # Estimador no sesgado de π
        pi_hat = torch.mean(X)
        var_hat = torch.var(X) / n

        print(f"Estimación de π (MC): {pi_hat.item():.{decimals}f}")
        print(f"Varianza estimada: {var_hat.item():.{decimals}f}")

        return pi_hat.item(), var_hat.item()

    def compute_pi_control(self, n=10000, decimals=4):
        """
        Estimador de π utilizando variable de control.

        Utiliza la variable de control W = 1{U1 + U2 > √2} cuya esperanza
        teórica es conocida: E[W] = 3 - 2√2.

        Parámetros
        ----------
        n : int, opcional
            Número de muestras. Por defecto 10000.
        decimals : int, opcional
            Número de decimales para mostrar. Por defecto 4.

        Retorna
        -------
        tuple
            (estimación_pi, varianza_estimada)

        Notas
        -----
        El estimador con variable de control tiene la forma:
        π_cv = X̄ - c*(W̄ - E[W]), donde c = Cov(X,W)/Var(W) es el coeficiente óptimo.
        """
        # Generar muestras uniformes en [0,1]²
        U = self.U.sample((n, 2)).to(self.device)

        # Variable de interés X = 4*1{U1²+U2² ≤ 1}
        indicator = (U[:, 0] ** 2 + U[:, 1] ** 2 <= 1).float()
        X = 4 * indicator

        # Variable de control W = 1{U1+U2 > √2}
        W = (U[:, 0] + U[:, 1] > math.sqrt(2)).float()

        # Esperanza teórica de W
        EW = 3 - 2 * math.sqrt(2)

        # Medias muestrales
        X_bar = torch.mean(X)
        W_bar = torch.mean(W)

        # Covarianza y varianza muestrales
        cov_XW = torch.mean((X - X_bar) * (W - W_bar))
        var_W = torch.var(W)

        # Coeficiente óptimo para la variable de control
        c = cov_XW / var_W

        # Estimador con variable de control
        pi_cv = X_bar - c * (W_bar - EW)

        # Varianza del estimador
        Y = X - c * (W - EW)
        var_hat = torch.var(Y) / n

        print(f"Estimación de π (Control Variate): {pi_cv.item():.{decimals}f}")
        print(f"Varianza estimada: {var_hat.item():.{decimals}f}")

        return pi_cv.item(), var_hat.item()

    def compute_pi_conditional(self, n=10000, decimals=4):
        """
        Estimador de π mediante Monte Carlo condicional.

        Utiliza la esperanza condicional E[X|U1] = 4√(1 - U1²), eliminando
        la varianza introducida por U2.

        Parámetros
        ----------
        n : int, opcional
            Número de muestras. Por defecto 10000.
        decimals : int, opcional
            Número de decimales para mostrar. Por defecto 4.

        Retorna
        -------
        tuple
            (estimación_pi, varianza_estimada)

        Notas
        -----
        El estimador condicional se basa en que:
        E[4·1{U1²+U2² ≤ 1} | U1] = 4·√(1 - U1²)·1{U1 ≤ 1}
        """
        # Generar muestras solo para U1
        U1 = self.U.sample((n,)).to(self.device)

        # Esperanza condicional E[X|U1] = 4*√(1 - U1²)
        X_cond = 4 * torch.sqrt(1 - U1 ** 2)

        pi_hat = torch.mean(X_cond)
        var_hat = torch.var(X_cond) / n

        print(f"Estimación de π (Conditional MC): {pi_hat.item():.{decimals}f}")
        print(f"Varianza estimada: {var_hat.item():.{decimals}f}")

        return pi_hat.item(), var_hat.item()

    def compute_pi_stratified(self, S=10, n=10000, decimals=4):
        """
        Estimación de π mediante muestreo estratificado en U1.

        Divide el intervalo [0,1] en S estratos y toma muestras uniformes
        dentro de cada estrato, reduciendo la varianza del estimador.

        Parámetros
        ----------
        S : int, opcional
            Número de estratos. Por defecto 10.
        n : int, opcional
            Número de muestras por estrato. Por defecto 10000.
        decimals : int, opcional
            Número de decimales para mostrar. Por defecto 4.

        Retorna
        -------
        tuple
            (estimación_pi, varianza_estimada)

        Notas
        -----
        El estimador estratificado se calcula como:
        π_hat = Σ p_s * μ_s, donde p_s = 1/S y μ_s es la media en el estrato s.
        """
        # Definir límites de los estratos
        a = torch.arange(0.0, 1.0, 1 / S).to(self.device)
        b = torch.arange(1 / S, 1.0 + 1 / S, 1 / S).to(self.device)

        # Distribución uniforme para cada estrato
        Unif_s = torch.distributions.Uniform(a, b)

        # Muestrear U1 de manera estratificada
        U_s = Unif_s.sample((n,)) ** 2

        # Muestrear U2 uniformemente
        U = self.U.sample((n, 1)).to(self.device) ** 2

        # Indicador de punto dentro del círculo
        X_S = (U_s + U) <= 1
        X = 4 * X_S.float()

        # Estimación de la media en cada estrato
        mu_s = torch.mean(X, dim=0)

        # Probabilidad de cada estrato
        p_s = 1 / S

        # Estimador final
        pi_hat = torch.sum(p_s * mu_s)

        # Varianza del estimador
        var_s = torch.var(X, dim=0)
        var_hat = torch.sum((p_s ** 2) * var_s / n)

        print(f"Estimación de π (Stratified MC): {pi_hat.item():.{decimals}f}")
        print(f"Varianza estimada: {var_hat.item():.{decimals}f}")

        return pi_hat.item(), var_hat.item()

    def compare_variances(self, n=10000, S=10):
        """
        Compara gráficamente las varianzas de los diferentes estimadores de π.

        Parámetros
        ----------
        n : int, opcional
            Número de muestras para cada estimador. Por defecto 10000.
        S : int, opcional
            Número de estratos para el muestreo estratificado. Por defecto 10.
        """
        # Calcular varianzas de cada método
        _, var_mc = self.compute_pi(n=n)
        _, var_cv = self.compute_pi_control(n=n)
        _, var_cond = self.compute_pi_conditional(n=n)
        _, var_strat = self.compute_pi_stratified(S=S, n=n)

        # Nombres de los métodos
        names = [
            "Monte Carlo",
            "Control Variate",
            "Conditional MC",
            "Stratified MC"
        ]

        vars_ = [var_mc, var_cv, var_cond, var_strat]

        # Crear gráfico de barras
        plt.figure(figsize=(8, 5))
        plt.bar(names, vars_)
        plt.ylabel("Varianza estimada")
        plt.title("Comparación de reducción de varianza en estimación de π")
        plt.xticks(rotation=20)
        plt.show()


class AntitheticMonteCarloIntegration:
    """
    Clase para integración Monte Carlo con variables antitéticas.

    Implementa la técnica de variables antitéticas para reducir la varianza
    en la estimación de integrales definidas. Utiliza pares de puntos
    correlacionados negativamente: x y x' = a + b - x.
    """

    def __init__(self, function, device="cpu"):
        """
        Inicializa el integrador con variables antitéticas.

        Parámetros
        ----------
        function : callable
            Función a integrar. Debe aceptar un tensor y devolver un tensor.
        device : str, opcional
            Dispositivo para los cálculos ('cpu' o 'cuda'). Por defecto 'cpu'.
        """
        self.device = device
        self.function = function
        self.Z_norm = torch.distributions.Normal(0, 1)  # Distribución normal para intervalos de confianza

    def compute_integral(self, a, b, n=100, alpha=0.05):
        """
        Calcula la integral de la función sobre [a, b] usando variables antitéticas.

        Parámetros
        ----------
        a : float
            Límite inferior de integración.
        b : float
            Límite superior de integración.
        n : int, opcional
            Número de pares antitéticos. Por defecto 100.
        alpha : float, opcional
            Nivel de significancia para el intervalo de confianza. Por defecto 0.05 (IC 95%).

        Retorna
        -------
        tuple
            (integral, intervalo_confianza, error_estándar)

        Notas
        -----
        La técnica de variables antitéticas utiliza:
        - Muestra original: x = a + (b-a)u
        - Muestra antitética: x' = a + (b-a)(1-u)
        El estimador es: (f(x) + f(x'))/2, que tiene varianza reducida.
        """
        # Generar muestras uniformes en [0,1]
        u = torch.rand((n,)).to(self.device)

        # Par de puntos antitéticos
        x = a + (b - a) * u
        x_ = a + (b - a) * (1 - u)

        # Valor crítico para intervalo de confianza
        alpha_t = torch.tensor([1 - alpha / 2], device=self.device)
        z_alpha = self.Z_norm.icdf(alpha_t).item()

        # Evaluar la función en ambos puntos y promediar
        values = (self.function(x) + self.function(x_)) / 2

        # Estimar la integral
        integral = ((b - a) * values.mean()).item()

        # Calcular error estándar
        se = (b - a) * values.std().item() / math.sqrt(n)

        # Intervalo de confianza
        conf_int = (integral - z_alpha * se, integral + z_alpha * se)

        print(
            "integral: {:.4f}, Intervalo Confianza: ({:.4f}, {:.4f}), Var: {:.4f}"
            .format(integral, conf_int[0], conf_int[1], se ** 2)
        )

        return integral, conf_int, se


class StratifiedMonteCarloIntegration:
    """
    Clase para integración Monte Carlo con muestreo estratificado.

    Divide el dominio de integración en estratos y toma muestras
    proporcionales en cada estrato para reducir la varianza del estimador.
    """

    def __init__(self, function, device="cpu"):
        """
        Inicializa el integrador con muestreo estratificado.

        Parámetros
        ----------
        function : callable
            Función a integrar. Debe aceptar un tensor y devolver un tensor.
        device : str, opcional
            Dispositivo para los cálculos ('cpu' o 'cuda'). Por defecto 'cpu'.
        """
        self.device = device
        self.function = function
        self.Z_norm = torch.distributions.Normal(0, 1)
        self.U_s = None  # Distribuciones por estrato

    def compute_integral(self, a, b, S=10, n=10000, alpha=0.05):
        """
        Calcula la integral de la función sobre [a, b] usando muestreo estratificado.

        Parámetros
        ----------
        a : float
            Límite inferior de integración.
        b : float
            Límite superior de integración.
        S : int, opcional
            Número de estratos. Por defecto 10.
        n : int, opcional
            Número de muestras por estrato. Por defecto 10000.
        alpha : float, opcional
            Nivel de significancia para el intervalo de confianza. Por defecto 0.05.

        Retorna
        -------
        tuple
            (integral, intervalo_confianza, varianza)

        Notas
        -----
        El estimador estratificado se calcula como:
        Î = Σ_{s=1}^{S} p_s * (1/n) * Σ_{i=1}^{n} f(X_{s,i})
        donde p_s = 1/S es la probabilidad del estrato s.
        """
        # Definir límites de los estratos en el espacio uniforme
        a_u = torch.arange(0.0, 1.0, 1 / S).to(self.device)
        b_u = torch.arange(1 / S, 1.0 + 1 / S, 1 / S).to(self.device)

        # Distribución uniforme para cada estrato
        self.U_s = torch.distributions.Uniform(a_u, b_u)

        # Muestrear en cada estrato
        u_s = self.U_s.sample((n,))

        # Transformar al dominio original [a, b] y evaluar la función
        x_s = (b - a) * self.function((a + (b - a) * u_s))

        # Probabilidad de cada estrato
        p_s = 1 / S

        # Media en cada estrato
        mu_s = torch.mean(x_s, dim=0)

        # Estimador final
        integral = torch.sum(p_s * mu_s).item()

        # Varianza dentro de cada estrato
        var_s = torch.var(x_s, dim=0)

        # Varianza del estimador
        var = torch.sum((p_s ** 2) * var_s / n).item()

        # Error estándar
        se = math.sqrt(var)

        # Intervalo de confianza
        alpha_t = torch.tensor([1 - alpha / 2], device=self.device)
        z_alpha = self.Z_norm.icdf(alpha_t).item()
        conf_int = (integral - z_alpha * se, integral + z_alpha * se)

        print(
            "integral: {:.4f}, Intervalo Confianza: ({:.4f}, {:.4f}), Var: {:.4f}"
            .format(integral, conf_int[0], conf_int[1], var)
        )

        return integral, conf_int, var


class NormalMultivariateMonteCarlo:
    """
    Clase para integración Monte Carlo con distribuciones normales multivariantes.

    Implementa métodos para estimar esperanzas de funciones de vectores
    aleatorios normales multivariantes, utilizando:
    - Monte Carlo estándar
    - Variables antitéticas para reducción de varianza
    """

    def __init__(self, mu, Sigma, f, device="cpu"):
        """
        Inicializa el integrador para distribuciones normales multivariantes.

        Parámetros
        ----------
        mu : torch.Tensor
            Vector de medias de dimensión d.
        Sigma : torch.Tensor
            Matriz de covarianzas de dimensión d×d.
        f : callable
            Función a evaluar sobre las muestras. Debe aceptar un tensor de forma (n, d).
        device : str, opcional
            Dispositivo para los cálculos ('cpu' o 'cuda'). Por defecto 'cpu'.

        Notas
        -----
        Utiliza la descomposición de Cholesky de Σ = LLᵀ para generar muestras
        correlacionadas como X = μ + ZLᵀ, donde Z ~ N(0, I).
        """
        self.device = device

        self.mu = mu.to(device)
        self.Sigma = Sigma.to(device)

        # Descomposición de Cholesky para generar muestras correlacionadas
        self.L = torch.linalg.cholesky(self.Sigma)

        self.d = self.mu.shape[0]  # Dimensión del espacio

        self.f = f

    def monte_carlo(self, n=10000, decimals=6):
        """
        Monte Carlo estándar para estimar E[f(X)] con X ~ N(μ, Σ).

        Parámetros
        ----------
        n : int, opcional
            Número de muestras. Por defecto 10000.
        decimals : int, opcional
            Número de decimales para mostrar. Por defecto 6.

        Retorna
        -------
        tuple
            (estimación_esperanza, varianza_estimada)
        """
        # Generar muestras normales estándar independientes
        Z = torch.randn(n, self.d).to(self.device)

        # Transformar a muestras con la correlación deseada
        X = self.mu + Z @ self.L.T

        # Evaluar la función en las muestras
        FX = self.f(X)

        # Estimar la esperanza
        mean = torch.mean(FX)
        var = torch.var(FX) / n

        print(f"Esperanza estimada (MC): {mean.item():.{decimals}f}")
        print(f"Varianza estimada: {var.item():.{decimals}f}")

        return mean.item(), var.item()

    def antithetic(self, n=10000, decimals=6):
        """
        Monte Carlo con variables antitéticas para reducir varianza.

        Genera pares de muestras correlacionadas negativamente: Z y -Z,
        que se transforman en X = μ + ZLᵀ y X' = μ - ZLᵀ.

        Parámetros
        ----------
        n : int, opcional
            Número de pares antitéticos. Por defecto 10000.
        decimals : int, opcional
            Número de decimales para mostrar. Por defecto 6.

        Retorna
        -------
        tuple
            (estimación_esperanza, varianza_estimada)

        Notas
        -----
        El estimador antitético es (f(X) + f(X'))/2, que típicamente tiene
        varianza menor que el estimador estándar.
        """
        # Generar muestras normales estándar
        Z = torch.randn(n, self.d).to(self.device)

        # Par de muestras antitéticas
        X1 = self.mu + Z @ self.L.T
        X2 = self.mu - Z @ self.L.T

        # Evaluar la función en ambos puntos
        f1 = self.f(X1)
        f2 = self.f(X2)

        # Estimador antitético
        estimator = (f1 + f2) / 2

        # Estimar la esperanza
        mean = torch.mean(estimator)
        var = torch.var(estimator) / n

        print(f"Esperanza estimada (Antithetic MC): {mean.item():.{decimals}f}")
        print(f"Varianza estimada: {var.item():.{decimals}f}")

        return mean.item(), var.item()

    def compare_variances(self, n=10000, figsize=(10, 6)):
        """
        Compara gráficamente las varianzas de los estimadores estándar y antitético.

        Parámetros
        ----------
        n : int, opcional
            Número de muestras para cada método. Por defecto 10000.
        figsize : tuple, opcional
            Tamaño de la figura. Por defecto (10, 6).
        """
        # Calcular varianzas de ambos métodos
        _, var_mc = self.monte_carlo(n=n)
        _, var_ant = self.antithetic(n=n)

        # Crear gráfico de barras
        methods = ["Monte Carlo", "Antithetic MC"]
        variances = [var_mc, var_ant]

        plt.figure(figsize=figsize)
        plt.bar(methods, variances)
        plt.ylabel("Varianza estimada")
        plt.title("Comparación de varianza: Monte Carlo vs Antitético")
        plt.show()