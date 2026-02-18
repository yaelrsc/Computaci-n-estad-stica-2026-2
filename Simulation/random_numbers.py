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



class DiscretaGeneral(RandomVariableSimulator):
    """
    Generador de una variable aleatoria discreta general.

    Esta clase permite simular variables aleatorias discretas con valores
    y probabilidades arbitrarias usando el método de probabilidades acumuladas.

    Sea:
        Valores posibles: x1, x2, ..., xn
        Probabilidades:  p1, p2, ..., pn   con sum(pi)=1

    Procedimiento de generación:
    ----------------------------
    1. Genera U ~ Uniforme(0,1)
    2. Devuelve el valor xi tal que la probabilidad acumulada hasta xi
       supera a U.

    Hereda de RandomVariableSimulator y sobrescribe los métodos generator() y pdf().

    Atributos:
    ----------
    valores : list
        Lista de valores posibles de la variable aleatoria.
    probabilidades : list
        Lista de probabilidades correspondientes a cada valor, deben sumar 1.
    kind : str
        Tipo de variable aleatoria: "discrete".
    pmf : dict
        Diccionario que asocia cada valor con su probabilidad.
    acumuladas : list
        Lista de probabilidades acumuladas para la generación.
    """

    def __init__(self, valores, probabilidades):
        """
        Inicializa la variable discreta general con los valores y probabilidades dadas.

        Parámetros:
        -----------
        valores : list
            Valores posibles de la variable aleatoria.
        probabilidades : list
            Probabilidades correspondientes a cada valor.

        Excepciones:
        ------------
        ValueError
            - Si valores y probabilidades no tienen la misma longitud.
            - Si las probabilidades no suman 1.
            - Si alguna probabilidad es negativa.
        """
       

        if len(valores) != len(probabilidades):
            raise ValueError("valores y probabilidades deben tener la misma longitud")

        if not abs(sum(probabilidades) - 1) < 1e-10:
            raise ValueError("Las probabilidades deben sumar 1")

        if any(p < 0 for p in probabilidades):
            raise ValueError("Las probabilidades deben ser no negativas")

        self.valores = valores
        self.probabilidades = probabilidades
        self.kind = "discrete"  # Variable discreta

        # Diccionario para la pmf
        self.pmf = dict(zip(valores, probabilidades))

        # Construcción de probabilidades acumuladas para la generación
        self.acumuladas = []
        acumulado = 0
        for p in probabilidades:
            acumulado += p
            self.acumuladas.append(acumulado)
            
    def pdf(self, x):
        """
        Función de probabilidad de la variable discreta.

        Parámetros:
        -----------
        x : valor
            Valor a evaluar.

        Retorna:
        --------
        float
            Probabilidad P(X=x). Retorna 0 si x no está en la lista de valores.
        """
        return self.pmf.get(x, 0.0)

    def generator(self):
        """
        Genera una realización de la variable discreta usando probabilidades acumuladas.

        Procedimiento:
        --------------
        1. Genera U ~ Uniforme(0,1)
        2. Devuelve el primer valor xi tal que U <= probabilidad acumulada hasta xi.

        Retorna:
        --------
        valor
            Una realización de la variable aleatoria discreta.
        """
        u = random.random()

        for valor, prob_acum in zip(self.valores, self.acumuladas):
            if u <= prob_acum:
                return valor
class UniformeAB(RandomVariableSimulator):
    """
    Generador de una variable aleatoria continua Uniforme(a, b).

    La variable X toma valores en el intervalo [a, b] con densidad constante:

        f(x) = 1 / (b - a)  para a < x < b
               0             en otro caso

    Hereda de RandomVariableSimulator y sobrescribe los métodos generator() y pdf().

    Atributos:
    ----------
    a : float
        Límite inferior del intervalo.
    b : float
        Límite superior del intervalo.
    kind : str
        Tipo de variable aleatoria: "continuous".
    """

    def __init__(self, a, b):
        """
        Inicializa la variable uniforme con los límites a y b.

        Parámetros:
        -----------
        a : float
            Límite inferior del intervalo.
        b : float
            Límite superior del intervalo (debe ser mayor que a).

        Excepciones:
        ------------
        ValueError
            Si b <= a.
        """
        

        if b <= a:
            raise ValueError("Se requiere a < b")

        self.a = a
        self.b = b
        self.kind = "continuous"  # Variable continua

    def pdf(self, x):
        """
        Función de densidad de la variable uniforme.

        Parámetros:
        -----------
        x : float
            Punto donde se evalúa la densidad.

        Retorna:
        --------
        float
            Valor de la densidad f(x). Retorna 0 si x no está en [a,b].
        """
        if self.a < x < self.b:
            return 1 / (self.b - self.a)
        else:
            return 0.0

    def generator(self):
        """
        Genera una realización de la variable uniforme.

        Procedimiento:
        --------------
        1. Genera U ~ Uniforme(0,1)
        2. Escala y desplaza para obtener un valor en [a, b]

        Retorna:
        --------
        float
            Número pseudoaleatorio en el intervalo [a, b].
        """
        u = random.random()
        return (self.b - self.a) * u + self.a
class Cauchy(RandomVariableSimulator):
    """
    Generador de una variable aleatoria continua Cauchy(mu, gamma).

    La variable X tiene densidad de probabilidad:

        f(x) = 1 / [pi * gamma * (1 + ((x - mu)/gamma)^2)]

    Donde:
    - mu : ubicación (mediana)
    - gamma : escala positiva

    Hereda de RandomVariableSimulator y sobrescribe los métodos generator() y pdf().

    Atributos:
    ----------
    mu : float
        Parámetro de ubicación de la distribución (mediana).
    gamma : float
        Parámetro de escala (debe ser > 0).
    kind : str
        Tipo de variable aleatoria: "continuous".
    """

    def __init__(self, mu=0, gamma=1):
        """
        Inicializa la variable Cauchy con parámetros mu y gamma.

        Parámetros:
        -----------
        mu : float, opcional
            Ubicación de la distribución (por defecto 0).
        gamma : float, opcional
            Escala positiva de la distribución (por defecto 1).

        Excepciones:
        ------------
        ValueError
            Si gamma <= 0.
        """
       

        if gamma <= 0:
            raise ValueError("gamma debe ser positiva")

        self.mu = mu
        self.gamma = gamma
        self.kind = "continuous"  # Variable continua

    def pdf(self, x):
        """
        Función de densidad de la distribución Cauchy.

        Parámetros:
        -----------
        x : float
            Punto donde se evalúa la densidad.

        Retorna:
        --------
        float
            Valor de la densidad f(x) en el punto x.
        """
        z = (x - self.mu) / self.gamma
        return 1 / (math.pi * self.gamma * (1 + z**2))

    def generator(self):
        """
        Genera una realización de la variable Cauchy usando la transformación
        inversa de la función de distribución.

        Procedimiento:
        --------------
        1. Genera U ~ Uniforme(0,1)
        2. Aplica la transformación inversa:
               X = mu + gamma * tan(pi * (U - 0.5))

        Retorna:
        --------
        float
            Número pseudoaleatorio con distribución Cauchy(mu, gamma).
        """
        u = random.random()
        return self.mu + self.gamma * math.tan(math.pi * (u - 0.5))
class Exponencial(RandomVariableSimulator):
    """
    Generador de una variable aleatoria continua Exponencial(lambda).

    La variable X tiene densidad de probabilidad:

        f(x) = lambda * exp(-lambda * x),  x >= 0

    Donde:
    - lambda : tasa positiva de la distribución

    Hereda de RandomVariableSimulator y sobrescribe los métodos generator() y pdf().

    Atributos:
    ----------
    lambd : float
        Parámetro de tasa de la distribución (debe ser > 0).
    kind : str
        Tipo de variable aleatoria: "continuous".
    """

    def __init__(self, lambd=1):
        """
        Inicializa la variable Exponencial con parámetro lambda.

        Parámetros:
        -----------
        lambd : float, opcional
            Parámetro de tasa (por defecto 1).

        Excepciones:
        ------------
        ValueError
            Si lambd <= 0.
        """
        

        if lambd <= 0:
            raise ValueError("lambda debe ser positiva")

        self.lambd = lambd
        self.kind = "continuous"  # Variable continua

    def pdf(self, x):
        """
        Función de densidad de la distribución Exponencial.

        Parámetros:
        -----------
        x : float
            Punto donde se evalúa la densidad.

        Retorna:
        --------
        float
            Valor de la densidad f(x). Retorna 0 si x < 0.
        """
        if x >= 0:
            return self.lambd * math.exp(-self.lambd * x)
        else:
            return 0.0

    def generator(self):
        """
        Genera una realización de la variable Exponencial usando la transformación inversa.

        Procedimiento:
        --------------
        1. Genera U ~ Uniforme(0,1)
        2. Aplica la transformación inversa:
               X = -ln(U) / lambda

        Retorna:
        --------
        float
            Número pseudoaleatorio con distribución Exponencial(lambda).
        """
        u = random.random()
        return -math.log(u) / self.lambd

class Erlang(RandomVariableSimulator):
    """
    Generador de una variable aleatoria Erlang (Gamma) con k entero positivo.

    Esta distribución es un caso especial de la distribución Gamma, con densidad:

        f(x) = (lambda^k / Gamma(k)) * x^(k-1) * exp(-lambda * x),  x >= 0

    Generación:
    -----------
    Se utiliza el método del producto de uniformes (suma de exponentiales):

        X = ( -ln(U1) - ln(U2) - ... - ln(Uk) ) / lambda
          = (sumatoria de k exponentiales) / lambda

    Hereda de RandomVariableSimulator y sobrescribe los métodos generator() y pdf().

    Atributos:
    ----------
    k : int
        Parámetro de forma (entero positivo).
    lam : float
        Parámetro de tasa (lambda > 0).
    kind : str
        Tipo de variable aleatoria: "continuous".
    """

    def __init__(self, k, lam):
        """
        Inicializa la variable Erlang con parámetros k y lambda.

        Parámetros:
        -----------
        k : int
            Número de etapas (entero positivo).
        lam : float
            Parámetro de tasa (lambda > 0).

        Excepciones:
        ------------
        ValueError
            Si k no es un entero positivo o lambda <= 0.
        """
    

        if k <= 0 or int(k) != k:
            raise ValueError("k debe ser entero positivo")

        if lam <= 0:
            raise ValueError("lambda debe ser positiva")

        self.k = int(k)
        self.lam = lam
        self.kind = "continuous"  # Variable continua

    def pdf(self, x):
        """
        Función de densidad de la distribución Erlang.

        Parámetros:
        -----------
        x : float
            Punto donde se evalúa la densidad.

        Retorna:
        --------
        float
            Valor de la densidad f(x). Retorna 0 si x < 0.
        """
        if x >= 0:
            coef = (self.lam ** self.k) / math.gamma(self.k)
            return coef * (x ** (self.k - 1)) * math.exp(-self.lam * x)
        else:
            return 0.0

    def generator(self):
        """
        Genera una realización de la variable Erlang usando suma de exponentiales.

        Procedimiento:
        --------------
        1. Genera k variables U_i ~ Uniforme(0,1)
        2. Suma -ln(U_i) para i=1..k
        3. Divide por lambda para obtener la realización

        Retorna:
        --------
        float
            Número pseudoaleatorio con distribución Erlang(k, lambda).
        """
        total = 0.0
        for _ in range(self.k):
            u = random.random()
            total += -math.log(u)
        return total / self.lam
class Binomial(RandomVariableSimulator):
    """
    Generador de una variable aleatoria Binomial(n, p).

    La variable X cuenta el número de éxitos en n ensayos independientes de
    Bernoulli(p):

        X = suma de n Bernoulli(p)

    Distribución de probabilidad:

        P(X = k) = C(n, k) * p^k * (1-p)^(n-k),   k = 0, 1, ..., n

    Hereda de RandomVariableSimulator y sobrescribe los métodos generator() y pdf().

    Atributos:
    ----------
    n : int
        Número de ensayos (entero positivo).
    p : float
        Probabilidad de éxito en cada ensayo (0 <= p <= 1).
    kind : str
        Tipo de variable aleatoria: "discrete".
    """

    def __init__(self, n, p):
        """
        Inicializa la variable Binomial con parámetros n y p.

        Parámetros:
        -----------
        n : int
            Número de ensayos (entero positivo).
        p : float
            Probabilidad de éxito en cada ensayo (0 <= p <= 1).

        Excepciones:
        ------------
        ValueError
            Si n no es un entero positivo o p no está en [0,1].
        """
    

        if n <= 0 or int(n) != n:
            raise ValueError("n debe ser entero positivo")

        if not (0 <= p <= 1):
            raise ValueError("p debe estar en [0,1]")

        self.n = int(n)
        self.p = p
        self.kind = "discrete"  # Variable discreta

    def pdf(self, k):
        """
        Función de probabilidad de la distribución Binomial.

        Parámetros:
        -----------
        k : int
            Número de éxitos a evaluar.

        Retorna:
        --------
        float
            Probabilidad P(X=k). Retorna 0 si k no es un entero entre 0 y n.
        """
        if k < 0 or k > self.n or int(k) != k:
            return 0.0
        else:
            comb = math.comb(self.n, int(k))
            return comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def generator(self):
        """
        Genera una realización de la variable Binomial(n, p) como suma de n Bernoulli(p).

        Procedimiento:
        --------------
        1. Para cada ensayo, genera un Bernoulli(p)
        2. Suma los éxitos para obtener X

        Retorna:
        --------
        int
            Número de éxitos observados en n ensayos.
        """
        total = 0
        for _ in range(self.n):
            if random.random() < self.p:
                total += 1
        return total
class Geometrica(RandomVariableSimulator):
    """
    Generador de una variable aleatoria Geométrica(p).

    La variable X representa el número de ensayos hasta el primer éxito
    en ensayos independientes de Bernoulli(p):

        P(X = k) = p * (1 - p)^(k - 1),   k = 1, 2, 3, ...

    Hereda de RandomVariableSimulator y sobrescribe los métodos generator() y pdf().

    Atributos:
    ----------
    p : float
        Probabilidad de éxito en cada ensayo (0 < p <= 1).
    kind : str
        Tipo de variable aleatoria: "discrete".
    """

    def __init__(self, p):
        """
        Inicializa la variable Geométrica con probabilidad p.

        Parámetros:
        -----------
        p : float
            Probabilidad de éxito en cada ensayo (0 < p <= 1).

        Excepciones:
        ------------
        ValueError
            Si p no está en (0,1].
        """
     

        if not (0 < p <= 1):
            raise ValueError("p debe estar en (0,1]")

        self.p = p
        self.kind = "discrete"  # Variable discreta

    def pdf(self, k):
        """
        Función de probabilidad de la distribución Geométrica.

        Parámetros:
        -----------
        k : int
            Número de ensayo hasta el primer éxito (k >= 1).

        Retorna:
        --------
        float
            Probabilidad P(X=k). Retorna 0 si k < 1 o no es entero.
        """
        if k < 1 or int(k) != k:
            return 0.0
        else:
            return self.p * ((1 - self.p) ** (k - 1))

    def generator(self):
        """
        Genera una realización de la variable Geométrica usando la transformación inversa.

        Procedimiento:
        --------------
        1. Genera U ~ Uniforme(0,1)
        2. Aplica la transformación inversa:
               X = floor( log(U) / log(1 - p) ) + 1

        Retorna:
        --------
        int
            Número de ensayos hasta el primer éxito.
        """
        u = random.random()
        return math.floor(math.log(u) / math.log(1 - self.p)) + 1
class LogUniforme(RandomVariableSimulator):
    """
    Generador de una variable aleatoria Log-Uniforme en el intervalo [1, e].

    La variable X tiene densidad de probabilidad:

        f(x) = 1 / x,   1 <= x <= e

    Hereda de RandomVariableSimulator y sobrescribe los métodos generator() y pdf().

    Atributos:
    ----------
    kind : str
        Tipo de variable aleatoria: "continuous".
    """

    def __init__(self):
        """
        Inicializa la variable Log-Uniforme en [1, e].
        """
     
        self.kind = "continuous"  # Variable continua

    def pdf(self, x):
        """
        Función de densidad de la distribución Log-Uniforme.

        Parámetros:
        -----------
        x : float
            Punto donde se evalúa la densidad.

        Retorna:
        --------
        float
            Valor de la densidad f(x). Retorna 0 si x no está en [1, e].
        """
        if 1.0 <= x <= math.e:
            return 1 / x
        else:
            return 0.0

    def generator(self):
        """
        Genera una realización de la variable Log-Uniforme usando la transformación inversa.

        Procedimiento:
        --------------
        1. Genera U ~ Uniforme(0,1)
        2. Aplica la transformación inversa:
               X = exp(U)

        Retorna:
        --------
        float
            Número pseudoaleatorio con distribución Log-Uniforme en [1, e].
        """
        u = random.random()
        return math.exp(u)

class Normal(RandomVariableSimulator):
    """
    Generador de una variable aleatoria Normal (Gaussiana) N(mu, sigma^2).

    La variable X tiene densidad de probabilidad:

        f(x) = 1 / (sigma * sqrt(2*pi)) * exp(-(x - mu)^2 / (2 * sigma^2))

    Generación:
    -----------
    Se utiliza el método de Box-Muller para generar números normales a partir
    de uniformes independientes U(0,1).

    Hereda de RandomVariableSimulator y sobrescribe los métodos generator() y pdf().

    Atributos:
    ----------
    mu : float
        Media de la distribución.
    sigma : float
        Desviación estándar (sigma > 0).
    kind : str
        Tipo de variable aleatoria: "continuous".
    """

    def __init__(self, mu=0, sigma=1):
        """
        Inicializa la variable Normal con media mu y desviación sigma.

        Parámetros:
        -----------
        mu : float, opcional
            Media de la distribución (por defecto 0).
        sigma : float, opcional
            Desviación estándar (por defecto 1).

        Excepciones:
        ------------
        ValueError
            Si sigma <= 0.
        """
        if sigma <= 0:
            raise ValueError("sigma debe ser positiva")

        self.mu = mu
        self.sigma = sigma
        self.kind = "continuous"  # Variable continua

    def pdf(self, x):
        """
        Función de densidad de la distribución Normal.

        Parámetros:
        -----------
        x : float
            Punto donde se evalúa la densidad.

        Retorna:
        --------
        float
            Valor de la densidad f(x) en el punto x.
        """
        coef = 1 / (self.sigma * math.sqrt(2 * math.pi))
        expo = math.exp(-((x - self.mu) ** 2) / (2 * self.sigma ** 2))
        return coef * expo

    def generator(self):
        """
        Genera una realización de la variable Normal usando Box-Muller.

        Procedimiento:
        --------------
        1. Genera dos uniformes independientes U1, U2 ~ Uniforme(0,1)
        2. Aplica la transformación Box-Muller:
               Z = sqrt(-2 * ln(U1)) * cos(2 * pi * U2)
        3. Escala y desplaza para obtener X = mu + sigma * Z

        Retorna:
        --------
        float
            Número pseudoaleatorio con distribución Normal(mu, sigma^2).
        """
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return self.mu + self.sigma * z
class Poisson(RandomVariableSimulator):
    """
    Generador de una variable aleatoria Poisson(lambda).

    La variable X cuenta el número de eventos en un intervalo de tiempo fijo
    bajo un proceso de Poisson con tasa lambda:

        P(X = k) = exp(-lambda) * lambda^k / k!,   k = 0, 1, 2, ...

    Generación:
    -----------
    Se utiliza un método basado en tiempos de espera exponenciales:
        1. Generar tiempos de espera E_i ~ Exponencial(lambda)
        2. Contar cuántos E_i se acumulan hasta superar 1

    Hereda de RandomVariableSimulator y sobrescribe los métodos generator() y pdf().

    Atributos:
    ----------
    lam : float
        Tasa de ocurrencia (lambda > 0).
    kind : str
        Tipo de variable aleatoria: "discrete".
    """

    def __init__(self, lam):
        """
        Inicializa la variable Poisson con parámetro lambda.

        Parámetros:
        -----------
        lam : float
            Tasa de ocurrencia de eventos (lambda > 0).

        Excepciones:
        ------------
        ValueError
            Si lambda <= 0.
        """
       

        if lam <= 0:
            raise ValueError("lambda debe ser positiva")

        self.lam = lam
        self.kind = "discrete"  # Variable discreta

    def pdf(self, k):
        """
        Función de probabilidad de la distribución Poisson.

        Parámetros:
        -----------
        k : int
            Número de eventos a evaluar (k >= 0).

        Retorna:
        --------
        float
            Probabilidad P(X=k). Retorna 0 si k no es entero >= 0.
        """
        if k >= 0 and int(k) == k:
            return math.exp(-self.lam) * (self.lam ** k) / math.factorial(k)
        else:
            return 0.0

    def generator(self):
        """
        Genera una realización de la variable Poisson usando tiempos de espera exponenciales.

        Procedimiento:
        --------------
        1. Inicializa t = 0 y n = 0
        2. Mientras t <= 1:
            a. Genera u ~ Uniforme(0,1)
            b. Calcula e = -ln(u) / lambda (Exponencial(lambda))
            c. Incrementa t += e
            d. Si t <= 1, incrementa n
        3. Retorna n

        Retorna:
        --------
        int
            Número de eventos ocurridos en el intervalo.
        """
        t = 0.0
        n = 0
        while t <= 1:
            u = random.random()
            e = -math.log(u) / self.lam
            t += e
            if t <= 1:
                n += 1
        return n
class Semicircular(RandomVariableSimulator):
    """
    Generador de una variable aleatoria con densidad semicircular.

    La densidad de probabilidad es:

        f(x) = (2/pi) * sqrt(1 - x^2),   -1 <= x <= 1

    Generación:
    -----------
    Se utiliza el método de aceptación-rechazo con propuesta Uniforme(-1,1):

        1. Genera X ~ Uniforme(-1,1)
        2. Genera U ~ Uniforme(0,1)
        3. Acepta X si U <= sqrt(1 - X^2)
        4. Si se rechaza, X se guarda en self.rejected

    Hereda de RandomVariableSimulator y sobrescribe los métodos generator(), pdf() y sample().

    Atributos:
    ----------
    kind : str
        Tipo de variable aleatoria: "continuous".
    rejected : list
        Lista de realizaciones rechazadas durante la generación.
    T_ar : float
        Constante máxima de la función de aceptación para AR.
    """

    def __init__(self):
        """
        Inicializa la variable Semicircular.
        """
        self.kind = "continuous"
        self.rejected = []  # Para almacenar valores rechazados
        self.T_ar = math.pi / 4  # Constante de aceptación-rechazo

    def generator(self):
        """
        Genera una realización de la variable Semicircular usando
        aceptación-rechazo con propuesta Uniforme(-1,1).

        Procedimiento:
        --------------
        1. Genera x ~ Uniforme(-1,1)
        2. Genera u ~ Uniforme(0,1)
        3. Acepta x si u <= sqrt(1 - x^2), si no, guarda x en self.rejected
        4. Repite hasta aceptar un valor

        Retorna:
        --------
        float
            Número pseudoaleatorio con distribución semicircular.
        """
        while True:
            x = 2 * random.random() - 1
            u = random.random()
            if u <= math.sqrt(1 - x**2):
                return x
            self.rejected.append(x)

    def pdf(self, x):
        """
        Función de densidad de la distribución semicircular.

        Parámetros:
        -----------
        x : float
            Punto donde se evalúa la densidad.

        Retorna:
        --------
        float
            Valor de la densidad f(x). Retorna 0 si x no está en [-1,1].
        """
        if -1 <= x <= 1:
            return 2 / math.pi * math.sqrt(1 - x**2)
        else:
            return 0.0

    def sample(self, n):
        """
        Genera n realizaciones de la variable semicircular y reinicia la lista
        de valores rechazados.

        Parámetros:
        -----------
        n : int
            Número de realizaciones a generar.

        Retorna:
        --------
        list
            Lista de n números pseudoaleatorios con distribución semicircular.
        """
        self.rejected = []  # Reinicia valores rechazados
        return [self.generator() for _ in range(n)]
class GammaAR(RandomVariableSimulator):
    """
    Generador de una variable aleatoria Gamma(alpha, beta) usando
    el método de aceptación-rechazo (AR).

    Esta implementación requiere alpha >= 1 y utiliza una propuesta
    basada en Gamma(entera) para generar la variable.

    Distribución:
    -------------
        f(x) = (beta^alpha / Gamma(alpha)) * x^(alpha-1) * exp(-beta * x),  x > 0

    Generación (AR):
    ----------------
    1. Se elige una propuesta Gamma(alpha_p, beta_p) con alpha_p = floor(alpha) >= 1
    2. Se genera x ~ propuesta
    3. Se acepta x con probabilidad f(x)/(K * proposal_pdf(x)), sino se rechaza
       y se guarda en self.rejected

    Hereda de RandomVariableSimulator y sobrescribe generator(), pdf() y sample().

    Atributos:
    ----------
    kind : str
        Tipo de variable aleatoria: "continuous".
    rejected : list
        Lista de realizaciones rechazadas durante la generación.
    alpha : float
        Parámetro de forma de la Gamma (alpha >= 1).
    beta : float
        Parámetro de tasa de la Gamma (beta > 0).
    alpha_p : int
        Parte entera usada para la propuesta.
    beta_p : float
        Tasa usada para la propuesta.
    Kp : float
        Constante de aceptación-rechazo.
    """

    def __init__(self, alpha, beta):
        """
        Inicializa la variable GammaAR con parámetros alpha y beta.

        Parámetros:
        -----------
        alpha : float
            Parámetro de forma (alpha >= 1).
        beta : float
            Parámetro de tasa (beta > 0).

        Excepciones:
        ------------
        ValueError
            Si alpha < 1.
        """
        if alpha < 1:
            raise ValueError("Este método requiere alpha >= 1")

        self.kind = 'continuous'
        self.rejected = []

        self.alpha = alpha
        self.beta = beta

        self.alpha_p = int(math.floor(alpha))

        # Parámetros de la propuesta
        if alpha >= 2:
            self.beta_p = beta * (self.alpha_p - 1) / (alpha - 1)
            self.Kp = math.exp(self.alpha_p - alpha) * ((alpha - 1) / beta) ** (alpha - self.alpha_p)
        else:
            self.alpha_p = 1
            self.beta_p = beta / alpha
            self.Kp = math.exp(1 - alpha) * (alpha / beta) ** (alpha - 1)

    def proposal(self):
        """
        Genera una realización de la propuesta Gamma(alpha_p, beta_p)
        usando el producto de uniformes.

        Retorna:
        --------
        float
            Valor de la propuesta.
        """
        prod = 1.0
        for _ in range(self.alpha_p):
            prod *= random.random()
        return -math.log(prod) / self.beta_p

    def pdf(self, x):
        """
        Función de densidad de la distribución Gamma(alpha, beta).

        Parámetros:
        -----------
        x : float
            Punto donde se evalúa la densidad.

        Retorna:
        --------
        float
            Valor de la densidad f(x). Retorna 0 si x <= 0.
        """
        if x > 0:
            coef = (self.beta ** self.alpha) / math.gamma(self.alpha)
            return coef * (x ** (self.alpha - 1)) * math.exp(-self.beta * x)
        else:
            return 0.0

    def generator(self):
        """
        Genera una realización de la variable Gamma(alpha, beta) usando
        aceptación-rechazo.

        Procedimiento:
        --------------
        1. Genera x ~ propuesta
        2. Calcula ratio = f(x) / (Kp * propuesta_pdf(x))
        3. Acepta x si U <= ratio, sino lo rechaza y lo guarda en self.rejected
        4. Repite hasta aceptar un valor

        Retorna:
        --------
        float
            Número pseudoaleatorio con distribución Gamma(alpha, beta).
        """
        while True:
            x = self.proposal()
            ratio = (x ** (self.alpha - 1) * math.exp(-self.beta * x)) / \
                    (self.Kp * x ** (self.alpha_p - 1) * math.exp(-self.beta_p * x))
            if random.random() <= ratio:
                return x
            self.rejected.append(x)

    def sample(self, n):
        """
        Genera n realizaciones de la variable GammaAR y reinicia la lista
        de valores rechazados.

        Parámetros:
        -----------
        n : int
            Número de realizaciones a generar.

        Retorna:
        --------
        list
            Lista de n números pseudoaleatorios con distribución Gamma(alpha, beta).
        """
        self.rejected = []  # Reinicia valores rechazados
        return [self.generator() for _ in range(n)]





        
        
