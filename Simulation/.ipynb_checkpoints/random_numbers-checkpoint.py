import math
import matplotlib.pyplot as plt
import random
from collections import Counter
import numpy as np

class RandomVariableSimulator:
    """
    Clase base para la simulación de variables aleatorias.

    Esta clase define la estructura general para generar muestras, calcular
    densidades/probabilidades y graficar resultados de variables aleatorias
    discretas o continuas.

    Métodos principales:
    -------------------
    generator()
        Debe ser implementado en la clase hija.
        Retorna una realización de la variable aleatoria.

    pdf(x)
        Debe ser implementado en la clase hija.
        Retorna la densidad (para continua) o probabilidad (para discreta)
        evaluada en x.

    sample(n)
        Genera una lista de n realizaciones de la variable aleatoria
        usando generator().

    plot(n=1000, rango=None, bins=50)
        Grafica la distribución de la variable aleatoria.
        - n: número de muestras a generar para el histograma/barplot.
        - rango: tupla (xmin, xmax) para limitar la gráfica continua.
        - bins: número de intervalos en el histograma (solo continua).

    Atributos:
    ----------
    kind : str
        Define si la variable es "discrete" o "continuous". Debe ser
        asignado en la clase hija antes de llamar a plot().
    """

    def __init__(self):
        self.kind = None  # Tipo de variable: 'discrete' o 'continuous'

    def generator(self):
        """
        Método abstracto que debe ser implementado en la clase hija.

        Retorna:
        --------
        float o int
            Una realización de la variable aleatoria.
        """
        raise NotImplementedError(
            "Debes implementar el método generator en la clase hija"
        )

    def pdf(self, x):
        """
        Método abstracto que debe ser implementado en la clase hija.

        Parámetros:
        -----------
        x : float o int
            Punto donde se evalúa la densidad o probabilidad.

        Retorna:
        --------
        float
            Valor de la densidad (continua) o probabilidad (discreta) en x.
        """
        raise NotImplementedError(
            "Debes implementar el método density en la clase hija"
        )

    def sample(self, n):
        """
        Genera n realizaciones de la variable aleatoria.

        Parámetros:
        -----------
        n : int
            Número de muestras a generar.

        Retorna:
        --------
        list
            Lista de n valores generados por generator().
        """
        return [self.generator() for _ in range(n)]

    def plot(self, n=1000, rango=None, bins=50):
        """
        Grafica la distribución de la variable aleatoria.

        Para variables discretas:
            - Se muestra un bar plot de la distribución empírica.
            - Se superpone la probabilidad teórica usando líneas y puntos.

        Para variables continuas:
            - Se muestra un histograma normalizado de la muestra.
            - Se superpone la densidad teórica usando una línea roja.

        Parámetros:
        -----------
        n : int, opcional
            Número de muestras para graficar (por defecto 1000).
        rango : tuple, opcional
            Tupla (xmin, xmax) para limitar la gráfica continua.
        bins : int, opcional
            Número de intervalos en el histograma (solo continua).

        Excepciones:
        ------------
        ValueError
            Si self.kind no está definido o tiene un valor inválido.
        """
        if self.kind is None:
            raise ValueError("Debes definir self.kind")

        data = self.sample(n)
        plt.figure()

        # -----------------------------
        # CASO DISCRETO
        # -----------------------------
        if self.kind == "discrete":
            
            counts = Counter(data)  # Conteo de ocurrencias
            x_vals = sorted(counts.keys())  # Valores únicos
            probs = [counts[x] / n for x in x_vals]  # Probabilidad empírica
            labels = [str(x) for x in x_vals]

            # Empírico — líneas verticales
            plt.bar(x_vals, probs, label="Empírica")

            # Teórica — probabilidad exacta
            theo = [self.pdf(x) for x in x_vals]
            plt.vlines(x_vals, [0], theo, colors='red', lw=2, alpha=0.9)
            plt.scatter(x_vals, theo, color='red', label="Teórica")

            plt.ylabel("Probabilidad")

        # -----------------------------
        # CASO CONTINUO
        # -----------------------------
        elif self.kind == "continuous":

            plt.hist(data, bins=bins, density=True,
                     range=rango, alpha=0.6, label="Empírica")

            # Determina rango de la curva teórica
            if rango is None:
                xmin = min(data)
                xmax = max(data)
            else:
                xmin, xmax = rango

            xs = np.linspace(xmin, xmax, 400)
            ys = [self.pdf(x) for x in xs]

            plt.plot(xs, ys, "r", lw=2, label="Teórica")
            plt.ylabel("Densidad")

        else:
            raise ValueError("self.kind debe ser 'discrete' o 'continuous'")

        plt.xlabel("Valores")
        plt.legend()
        plt.show()

class Bernoulli(RandomVariableSimulator):
    """
    Simulación de una variable aleatoria Bernoulli(p).

    La variable X toma solo dos valores:
        P(X = 1) = p
        P(X = 0) = 1 - p

    Hereda de RandomVariableSimulator y sobrescribe los métodos generator() y pdf().

    Atributos:
    ----------
    p : float
        Probabilidad de éxito (X=1), debe estar en [0,1].
    kind : str
        Tipo de variable aleatoria: "discrete".
    """

    def __init__(self, p):
        """
        Inicializa la variable Bernoulli con la probabilidad p.

        Parámetros:
        -----------
        p : float
            Probabilidad de éxito (X=1), debe estar entre 0 y 1.

        Excepciones:
        ------------
        ValueError
            Si p no está en el intervalo [0,1].
        """
        if not (0 <= p <= 1):
            raise ValueError("El parámetro p debe estar en [0,1]")

        self.p = p
        self.kind = "discrete"  # Variable discreta

    def pdf(self, x):
        """
        Función de probabilidad de la variable Bernoulli.

        Parámetros:
        -----------
        x : int
            Valor a evaluar (0 o 1).

        Retorna:
        --------
        float
            Probabilidad P(X=x). Retorna 0 si x no es 0 ni 1.
        """
        if x == 1:  
            return self.p  
        elif x == 0:
            return 1 - self.p
        else:
            return 0.0

    def generator(self):
        """
        Genera una realización de la variable Bernoulli(p).

        Usa un número uniforme aleatorio para decidir el resultado.

        Retorna:
        --------
        int
            1 con probabilidad p, 0 con probabilidad 1-p.
        """
        u = random.random()
        return 1 if u < self.p else 0

class GLC(RandomVariableSimulator):
    """
    Generador Lineal Congruencial (GLC) para números pseudoaleatorios continuos.

    Este generador produce una secuencia de números uniformes en (0,1) usando la 
    recurrencia lineal congruencial:

        X_{n+1} = (a * X_n + c) mod m
        U_n = X_n / m

    Donde:
    - m : módulo
    - a : multiplicador
    - c : incremento
    - X_0 : semilla inicial

    Hereda de RandomVariableSimulator y sobrescribe los métodos generator() y pdf().
    
    Atributos:
    ----------
    m : int
        Módulo del generador (por defecto 2**31 - 1).
    a : int
        Multiplicador (por defecto 7**5).
    c : int
        Incremento (por defecto 0).
    state : int
        Estado actual / semilla.
    kind : str
        Tipo de variable aleatoria: "continuous" (siempre 0 < U < 1).
    """

    def __init__(self, m=2**31 - 1, a=7**5, c=0, seed=1):
        """
        Inicializa el generador lineal congruencial con los parámetros dados.

        Parámetros:
        -----------
        m : int, opcional
            Módulo del generador (por defecto 2**31 - 1).
        a : int, opcional
            Multiplicador (por defecto 7**5).
        c : int, opcional
            Incremento (por defecto 0).
        seed : int, opcional
            Valor inicial del estado / semilla (por defecto 1).
        """
        super().__init__()

        self.m = m
        self.a = a
        self.c = c
        self.state = seed

        self.kind = "continuous"  # Siempre produce números en (0,1)

    def pdf(self, x):
        """
        Función de densidad teórica de la variable generada.

        Dado que el GLC genera números uniformes en (0,1), la densidad es:

            f(x) = 1  si 0 < x < 1
                 = 0  en otro caso

        Parámetros:
        -----------
        x : float
            Punto donde se evalúa la densidad.

        Retorna:
        --------
        float
            Valor de la densidad en x.
        """
        if 0 < x < 1:
            return 1
        else:
            return 0

    def generator(self):
        """
        Genera un número pseudoaleatorio uniforme en (0,1) usando la 
        recurrencia lineal congruencial.

        Actualiza el estado interno y retorna la realización normalizada.

        Retorna:
        --------
        float
            Número pseudoaleatorio en (0,1).
        """
        # Actualiza el estado según la fórmula del GLC
        self.state = (self.a * self.state + self.c) % self.m

        return self.state / self.m




        
        
