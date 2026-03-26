"""
Módulo de Integración Monte Carlo y Estimación Estadística

Este módulo proporciona clases para integración Monte Carlo, estimación de
distribuciones empíricas y cálculo de medidas de riesgo utilizando tensores
de PyTorch con soporte para GPU.
"""

import torch
from math import sqrt
import math
import matplotlib.pyplot as plt

# Configurar dispositivo para operaciones con tensores (GPU si está disponible, sino CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MonteCarloIntegration:
    """
    Clase de Integración Monte Carlo

    Estima integrales definidas utilizando métodos Monte Carlo con intervalos
    de confianza. Utiliza muestreo uniforme entre los límites especificados
    para aproximar integrales de funciones.
    """

    def __init__(self, function):
        """
        Inicializa el integrador Monte Carlo.

        Parámetros
        ----------
        function : callable
            La función a integrar. Debe aceptar un tensor de entrada y devolver
            un tensor de salida.
        """
        self.function = function
        self.Z_norm = torch.distributions.Normal(0, 1)  # Distribución normal estándar para intervalos de confianza

    def compute_integral(self, a, b, n=100, alpha=0.05):
        """
        Calcula la integral de la función sobre [a, b] utilizando el método Monte Carlo.

        Parámetros
        ----------
        a : float
            Límite inferior de integración.
        b : float
            Límite superior de integración.
        n : int, opcional
            Número de puntos de muestreo. Por defecto es 100.
        alpha : float, opcional
            Nivel de significancia para el intervalo de confianza. Por defecto es 0.05 (IC 95%).

        Retorna
        -------
        tupla
            (valor_integral, intervalo_confianza, varianza) donde intervalo_confianza
            es una tupla de (límite_inferior, límite_superior).

        Notas
        -----
        La estimación de la integral se calcula como: (b-a) * (1/n) * Σ f(x_i)
        El error estándar es: (b-a) * σ(f(x)) / √n
        """
        # Generar muestras uniformes entre a y b
        dist = torch.distributions.Uniform(a, b)
        unif_sample = dist.sample((n,)).to(device)

        # Calcular el valor crítico para el intervalo de confianza
        alpha_t = torch.tensor([1 - alpha / 2], device=device)
        z_alpha = self.Z_norm.icdf(alpha_t).item()

        # Evaluar la función en los puntos de muestreo
        values = self.function(unif_sample)

        # Calcular la estimación de la integral
        integral = (b - a) * values.mean().item()

        # Calcular el error estándar
        se = (b - a) * values.std().item() / sqrt(n)

        # Construir el intervalo de confianza
        conf_int = (integral - z_alpha * se, integral + z_alpha * se)

        # Mostrar resultados
        print(
            "integral: {:.4f}, Intervalo Confianza: ({:.4f}, {:.4f}), Var: {:.4f}"
                .format(integral, conf_int[0], conf_int[1], se ** 2)
        )

        return integral, conf_int, se ** 2


class EmpiricalCDF:
    """
    Clase de Función de Distribución Acumulada Empírica

    Calcula y visualiza la función de distribución acumulada empírica con
    bandas de confianza de Dvoretzky-Kiefer-Wolfowitz (DKW) para un conjunto
    de datos dado.
    """

    def __init__(self, data):
        """
        Inicializa la función de distribución acumulada empírica con datos ordenados.

        Parámetros
        ----------
        data : torch.Tensor
            Datos de entrada para construir la función de distribución acumulada empírica.
        """
        self.data = torch.sort(data).values.to(device)
        self.n = len(data)

    def compute(self, x, alpha=0.05, print_=True):
        """
        Calcula la función de distribución acumulada empírica y la banda de confianza
        en los puntos especificados.

        Parámetros
        ----------
        x : torch.Tensor
            Puntos en los que evaluar la función de distribución acumulada empírica.
        alpha : float, opcional
            Nivel de significancia para la banda de confianza. Por defecto es 0.05 (banda 95%).
        print_ : bool, opcional
            Indica si se deben mostrar los resultados al evaluar en un solo punto.
            Por defecto es True.

        Retorna
        -------
        tupla
            (Fnx, (banda_inferior, banda_superior)) donde Fnx es el valor de la
            función de distribución acumulada empírica, y banda_inferior/banda_superior
            son los límites de la banda de confianza DKW.

        Notas
        -----
        Utiliza la desigualdad DKW para construir las bandas de confianza:
        ε = sqrt(log(2/α) / (2n))
        """
        x = x.to(device)

        # Contar el número de observaciones ≤ x
        count = torch.searchsorted(self.data, x, right=True)
        Fnx = count.float() / self.n

        # Ancho de la banda de confianza DKW
        ep = math.sqrt(math.log(2 / alpha) / (2 * self.n))
        low = torch.clamp(Fnx - ep, 0., 1.)
        up = torch.clamp(Fnx + ep, 0., 1.)

        # Mostrar resultados para evaluación en un solo punto
        if print_ and Fnx.numel() == 1:
            print(f"ECDF: {Fnx.item():.4f}, Banda: ({low.item():.4f}, {up.item():.4f})")

        return Fnx, (low, up)

    def plot(self, x_min, x_max, alpha=0.05, n_points=1000):
        """
        Grafica la función de distribución acumulada empírica con bandas de confianza DKW.

        Parámetros
        ----------
        x_min : float
            Valor mínimo del eje x para el gráfico.
        x_max : float
            Valor máximo del eje x para el gráfico.
        alpha : float, opcional
            Nivel de significancia para la banda de confianza. Por defecto es 0.05.
        n_points : int, opcional
            Número de puntos a evaluar para el gráfico. Por defecto es 1000.
        """
        # Generar puntos de evaluación
        points = torch.linspace(x_min, x_max, n_points, device=device)

        # Calcular la FDA empírica y las bandas de confianza
        Fnx, (low, up) = self.compute(points, alpha=alpha, print_=False)

        # Crear visualización
        plt.figure(figsize=(10, 6))
        plt.fill_between(points.cpu(), low.cpu(), up.cpu(), color='red',
                         alpha=0.2, label='Banda de Confianza')
        plt.step(points.cpu(), Fnx.cpu(), where='post', label='FDA Empírica', color='blue')
        plt.title(f"Función de Distribución Acumulada Empírica con Banda de Confianza (α={alpha})")
        plt.xlabel("x")
        plt.ylabel("F(x)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


class EmpiricalPDF:
    """
    Clase de Función de Densidad de Probabilidad Empírica

    Estima y visualiza la función de densidad de probabilidad utilizando
    un estimador kernel de densidad basado en muestreo de una distribución dada.
    """

    def __init__(self, distribution):
        """
        Inicializa el estimador de densidad empírica.

        Parámetros
        ----------
        distribution : torch.distributions.Distribution
            Distribución de probabilidad de la cual se generarán las muestras.
        """
        self.distribution = distribution

    def compute(self, x, n_sim=1000, n_groups=100, h=None):
        """
        Calcula la estimación de densidad empírica en los puntos especificados.

        Parámetros
        ----------
        x : torch.Tensor
            Puntos en los que evaluar la densidad.
        n_sim : int, opcional
            Número de simulaciones. Por defecto es 1000.
        n_groups : int, opcional
            Número de grupos por simulación. Por defecto es 100.
        h : float, opcional
            Ancho de banda para el estimador kernel. Si es None, se calcula como n_groups^(-1/3).

        Retorna
        -------
        torch.Tensor
            Estimación de la densidad en los puntos x.

        Notas
        -----
        Utiliza un estimador kernel simple basado en el método de ventana deslizante:
        f̂(x) = (1/(n·h)) * Σ I(x < X_i ≤ x + h)
        """
        if h is None:
            h = n_groups ** (-1 / 3)

        x = x.to(device)

        # Generar muestras de la distribución
        sample_ = self.distribution.sample((n_sim, n_groups, 1)).to(device)

        # Contar observaciones en el intervalo (x, x+h]
        mask = (sample_ > x) & (sample_ <= x + h)

        # Calcular la estimación de densidad
        fx = mask.sum(dim=(0, 1)) / (n_sim * n_groups * h)

        return fx

    def plot(self, x_min, x_max, n_points=200, n_sim=1000, n_groups=100, h=None):
        """
        Grafica la función de densidad empírica junto con la densidad verdadera.

        Parámetros
        ----------
        x_min : float
            Valor mínimo del eje x para el gráfico.
        x_max : float
            Valor máximo del eje x para el gráfico.
        n_points : int, opcional
            Número de puntos para evaluar. Por defecto es 200.
        n_sim : int, opcional
            Número de simulaciones. Por defecto es 1000.
        n_groups : int, opcional
            Número de grupos por simulación. Por defecto es 100.
        h : float, opcional
            Ancho de banda para el estimador kernel.
        """
        # Generar puntos de evaluación
        x = torch.linspace(x_min, x_max, n_points).to(device)

        # Calcular estimador empírico
        fx_est = self.compute(x, n_sim=n_sim, n_groups=n_groups, h=h)

        # Calcular densidad verdadera
        fx_true = torch.exp(self.distribution.log_prob(x))

        # Mover a CPU para graficar
        x = x.cpu()
        fx_est = fx_est.cpu()
        fx_true = fx_true.cpu()

        # Crear visualización
        plt.figure(figsize=(10, 6))
        plt.plot(x, fx_true, label="Densidad Verdadera", linewidth=2, color='red')
        plt.plot(x, fx_est, '--', label="Densidad Empírica", color='blue')
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Densidad Empírica vs Densidad Verdadera")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()


class ParametricVaRCVaR:
    """
    Clase de Valor en Riesgo (VaR) y Valor en Riesgo Condicional (CVaR) Paramétricos

    Calcula medidas de riesgo paramétricas utilizando muestreo de una distribución
    dada, con intervalos de confianza basados en aproximación asintótica.
    """

    def __init__(self, distribution):
        """
        Inicializa el calculador de VaR y CVaR.

        Parámetros
        ----------
        distribution : torch.distributions.Distribution
            Distribución de probabilidad para los retornos o pérdidas.
        """
        self.distribution = distribution
        self.Z = torch.distributions.Normal(0, 1)  # Distribución normal estándar

    def compute_VaR(self, n=1000, alpha=0.05, beta=0.05):
        """
        Calcula el Valor en Riesgo (VaR) paramétrico.

        Parámetros
        ----------
        n : int, opcional
            Número de muestras para la simulación. Por defecto es 1000.
        alpha : float, opcional
            Nivel de significancia para VaR (probabilidad de cola). Por defecto es 0.05.
        beta : float, opcional
            Nivel de significancia para el intervalo de confianza. Por defecto es 0.05.

        Retorna
        -------
        tupla
            (VaR, intervalo_confianza) donde intervalo_confianza es una tupla
            de (límite_inferior, límite_superior).

        Notas
        -----
        El error estándar se calcula utilizando la fórmula asintótica:
        SE = √(α(1-α)) / (f(VaR_hat)√n)
        """
        # Generar muestras de la distribución
        sample_ = self.distribution.sample((n,)).to(device)

        # Estimar VaR como el cuantil α
        VaR_hat = torch.quantile(sample_, alpha)

        # Calcular valor crítico para el intervalo de confianza
        beta2 = torch.Tensor([1 - beta / 2]).to(device)
        z_beta2 = self.Z.icdf(beta2)

        # Calcular error estándar asintótico
        se = math.sqrt(alpha * (1 - alpha)) / (self.distribution.log_prob(VaR_hat).exp() * math.sqrt(n))

        # Construir intervalo de confianza
        low = VaR_hat - z_beta2 * se
        up = VaR_hat + z_beta2 * se
        conf_int = (low, up)

        # Mostrar resultados
        print("VaR : {:.4f},\nIntervalo Confianza: ({:.4f}, {:.4f})".format(
            VaR_hat.item(), low.item(), up.item()))

        return VaR_hat, conf_int

    def compute_CVaR(self, n=1000, alpha=0.05, beta=0.05):
        """
        Calcula el Valor en Riesgo Condicional (CVaR) paramétrico.

        Parámetros
        ----------
        n : int, opcional
            Número de muestras para la simulación. Por defecto es 1000.
        alpha : float, opcional
            Nivel de significancia para CVaR (probabilidad de cola). Por defecto es 0.05.
        beta : float, opcional
            Nivel de significancia para el intervalo de confianza. Por defecto es 0.05.

        Retorna
        -------
        tupla
            (CVaR, intervalo_confianza) donde intervalo_confianza es una tupla
            de (límite_inferior, límite_superior).

        Notas
        -----
        CVaR se calcula como la pérdida esperada condicionada a que la pérdida
        supere el VaR: CVaR = E[X | X ≤ VaR_α]
        """
        # Generar muestras de la distribución
        sample_ = self.distribution.sample((n,)).to(device)

        # Estimar VaR como el cuantil α
        VaR_hat = torch.quantile(sample_, alpha)

        # Estimar CVaR como el promedio de las observaciones por debajo del VaR
        CVaR_hat = torch.where(sample_ <= VaR_hat, sample_, 0.0).mean() / alpha

        # Calcular error estándar
        se = (1 / alpha) * torch.where(sample_ <= VaR_hat, sample_, 0.0).std() / math.sqrt(n)

        # Calcular valor crítico para el intervalo de confianza
        beta2 = torch.Tensor([1 - beta / 2]).to(device)
        z_beta2 = self.Z.icdf(beta2)

        # Construir intervalo de confianza
        low = CVaR_hat - z_beta2 * se
        up = CVaR_hat + z_beta2 * se
        conf_int = (low, up)

        # Mostrar resultados
        print("CVaR : {:.4f},\nIntervalo Confianza: ({:.4f}, {:.4f})".format(
            CVaR_hat.item(), low.item(), up.item()))

        return CVaR_hat, conf_int

        
        
        
        

        
        