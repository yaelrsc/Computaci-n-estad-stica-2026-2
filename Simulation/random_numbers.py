import math
import matplotlib.pyplot as plt
import random


def cuadrado_mitad(seed: int, pasos: int):
    """
    Generador pseudoaleatorio usando el método cuadrado–mitad.

    Parámetros
    ----------
    seed : int
        Semilla inicial x0 .
    pasos : int
        Cantidad de números a generar.
    Retorna
    -------
    list
        Lista con los números generados.
    """

    x = seed
    resultados = []
    n = len(str(seed))
    i = 0
    while i < pasos:
        # 1) elevar al cuadrado
        cuadrado = x**2

        # 2) convertir a string con al menos 2n dígitos
        s = str(cuadrado).zfill(2 * n)

        # 3) extraer los n dígitos centrales
        inicio = (len(s) - n) //  2
        medio = s[inicio:inicio + n]

        # 4) nueva semilla
        x = int(medio)

        # 5) guardar resultado
        
        resultados.append(x)
        i = i+1 

    return resultados



class GLC:
    """
    Generador Lineal Congruencial (Park & Miller por default)

    x_{n+1} = a (x_n + c) mod m
    u_n = x_n / m
    """

    def __init__(self, m=2**31 - 1, a=7**5, c=0, seed=1):
        """
        Parámetros
        ----------
        m : int
            Módulo
        a : int
            Multiplicador
        c : int
            Incremento
        seed : int
            Semilla inicial
        """
        self.m = m
        self.a = a
        self.c = c
        self.state = seed

    # ---------------------------------------------------
    # Genera un solo número
    # ---------------------------------------------------
    def _next(self):
        self.state = (self.a * (self.state + self.c)) % self.m
        return self.state / self.m

    # ---------------------------------------------------
    # Genera una muestra de tamaño n
    # ---------------------------------------------------
    def sample(self, n):
        """
        Genera n números pseudoaleatorios en (0,1).
        """
        return [self._next() for _ in range(n)]

    # ---------------------------------------------------
    # Histograma de una muestra
    # ---------------------------------------------------
    def histogram(self, n, bins=30):
        """
        Genera n números y grafica su histograma.
        """
        data = self.sample(n)

        plt.figure()
        plt.hist(data, bins=bins, density=True)
        plt.title("Histograma - Generador Lineal Congruencial")
        plt.xlabel("u")
        plt.ylabel("Frecuencia")
        plt.show()


class Bernoulli:
    """
    Generador de variables aleatorias Bernoulli(p)
    usando U ~ Uniforme(0,1) vía random.random().
    """

    def __init__(self, p):
        if not (0 <= p <= 1):
            raise ValueError("p debe estar en [0,1]")
        self.p = p

    # ---------------------------------------------------
    # Genera una sola realización
    # ---------------------------------------------------
    def _next(self):
        u = random.random()
        return 1 if u <= self.p else 0

    # ---------------------------------------------------
    # Genera una muestra
    # ---------------------------------------------------
    def sample(self, n):
        return [self._next() for _ in range(n)]

    # ---------------------------------------------------
    # Bar plot (correcto para variable discreta)
    # ---------------------------------------------------
    def barplot(self, n):
        data = self.sample(n)

        # frecuencias empíricas
        count_0 = data.count(0) / n
        count_1 = data.count(1) / n

        plt.figure()
        plt.bar([0, 1], [count_0, count_1])
        plt.title(f"Bar plot Bernoulli(p={self.p})")
        plt.xlabel("Valor")
        plt.ylabel("Frecuencia relativa")
        plt.xticks([0, 1])
        plt.show()

class DiscretaGeneral:
    """
    Generador para una variable aleatoria discreta general.

    Valores posibles: x1, x2, ..., xn
    Probabilidades:  p1, p2, ..., pn   con sum(pi)=1

    Usa U ~ Uniforme(0,1) vía random.random()
    y el método de probabilidades acumuladas.
    """

    def __init__(self, valores, probabilidades):
        if len(valores) != len(probabilidades):
            raise ValueError("valores y probabilidades deben tener la misma longitud")

        if abs(sum(probabilidades) - 1.0) > 1e-8:
            raise ValueError("Las probabilidades deben sumar 1")

        self.valores = valores
        self.probabilidades = probabilidades

        # Construir probabilidades acumuladas
        self.acumuladas = []
        s = 0.0
        for p in probabilidades:
            s += p
            self.acumuladas.append(s)

    # ---------------------------------------------------
    # Genera una sola realización
    # ---------------------------------------------------
    def _next(self):
        u = random.random()

        for valor, p_acum in zip(self.valores, self.acumuladas):
            if u <= p_acum:
                return valor


    # ---------------------------------------------------
    # Genera muestra
    # ---------------------------------------------------
    def sample(self, n):
        return [self._next() for _ in range(n)]

    # ---------------------------------------------------
    # Bar plot (correcto para distribución discreta)
    # ---------------------------------------------------
    def barplot(self, n):
        data = self.sample(n)

        # frecuencias empíricas
        freqs = []
        for v in self.valores:
            freqs.append(data.count(v) / n)

        plt.figure()
        plt.bar(self.valores, freqs)
        plt.title("Bar plot - Distribución discreta general")
        plt.xlabel("Valores")
        plt.ylabel("Frecuencia relativa")
        plt.show()

class UniformeAB:
    """
    Generador Uniforme(a,b) usando U ~ Uniforme(0,1)
    vía la transformación Y = (b-a)X + a.
    """

    def __init__(self, a, b):
        if b <= a:
            raise ValueError("Se requiere a < b")
        self.a = a
        self.b = b

    # ---------------------------------------------------
    # Genera una sola realización
    # ---------------------------------------------------
    def _next(self):
        x = random.random()  # Uniforme(0,1)
        return (self.b - self.a) * x + self.a

    # ---------------------------------------------------
    # Genera muestra
    # ---------------------------------------------------
    def sample(self, n):
        return [self._next() for _ in range(n)]

    # ---------------------------------------------------
    # Histograma
    # ---------------------------------------------------
    def histogram(self, n, bins=30):
        data = self.sample(n)

        plt.figure()
        plt.hist(data, bins=bins, density=True)
        plt.title(f"Histograma Uniforme({self.a},{self.b})")
        plt.xlabel("Valor")
        plt.ylabel("Densidad")
        plt.show()

class Cauchy:
    """
    Generador Cauchy(0,1) usando U ~ Uniforme(0,1)

    Transformación:
        X = tan( pi (U - 1/2) )
    """

    def __init__(self):
        pass

    # ---------------------------------------------------
    # Genera una sola realización
    # ---------------------------------------------------
    def _next(self):
        u = random.random()          # U ~ Uniforme(0,1)
        return math.tan(math.pi * (u - 0.5))

    # ---------------------------------------------------
    # Genera una muestra
    # ---------------------------------------------------
    def sample(self, n):
        return [self._next() for _ in range(n)]

    # ---------------------------------------------------
    # Histograma
    # ---------------------------------------------------
    def histogram(self, n, bins=60, rango=(-10, 10)):
        """
        Nota:
        La Cauchy tiene colas muy pesadas, por eso
        se limita el rango del histograma.
        """
        data = self.sample(n)

        plt.figure()
        plt.hist(data, bins=bins, density=True, range=rango)
        plt.title("Histograma Cauchy(0,1)")
        plt.xlabel("x")
        plt.ylabel("Densidad")
        plt.show()
class Exponencial:
    """
    Generador Exponencial(lambda) usando U ~ Uniforme(0,1)

        X = - (1/lambda) * log(U)
    """

    def __init__(self, lam):
        if lam <= 0:
            raise ValueError("lambda debe ser positiva")
        self.lam = lam

    # ---------------------------------------------------
    # Genera una sola realización
    # ---------------------------------------------------
    def _next(self):
        u = random.random()     # U ~ Uniforme(0,1)
        return -math.log(u) / self.lam

    # ---------------------------------------------------
    # Genera muestra
    # ---------------------------------------------------
    def sample(self, n):
        return [self._next() for _ in range(n)]

    # ---------------------------------------------------
    # Histograma
    # ---------------------------------------------------
    def histogram(self, n, bins=50):
        data = self.sample(n)

        plt.figure()
        plt.hist(data, bins=bins, density=True)
        plt.title(f"Histograma Exponencial(lambda={self.lam})")
        plt.xlabel("x")
        plt.ylabel("Densidad")
        plt.show()

class Gamma:
    """
    Generador Gamma(k, lambda) con k entero positivo
    usando el producto de uniformes.
    """

    def __init__(self, k, lam):
        if k <= 0 or int(k) != k:
            raise ValueError("k debe ser entero positivo")
        if lam <= 0:
            raise ValueError("lambda debe ser positiva")

        self.k = int(k)
        self.lam = lam

    # ---------------------------------------------------
    # Genera una sola realización
    # ---------------------------------------------------
    def _next(self):
        prod = 1.0
        for _ in range(self.k):
            prod *= random.random()  # producto de Uniformes(0,1)

        return -math.log(prod) / self.lam

    # ---------------------------------------------------
    # Genera muestra
    # ---------------------------------------------------
    def sample(self, n):
        return [self._next() for _ in range(n)]

    # ---------------------------------------------------
    # Histograma
    # ---------------------------------------------------
    def histogram(self, n, bins=50):
        data = self.sample(n)

        plt.figure()
        plt.hist(data, bins=bins, density=True)
        plt.title(f"Histograma Gamma(k={self.k}, lambda={self.lam})")
        plt.xlabel("x")
        plt.ylabel("Densidad")
        plt.show()
class Binomial:
    """
    Generador Binomial(n, p) como suma de n Bernoulli(p)
    """

    def __init__(self, n, p):
        if n <= 0 or int(n) != n:
            raise ValueError("n debe ser entero positivo")
        if not (0 <= p <= 1):
            raise ValueError("p debe estar en [0,1]")

        self.n = int(n)
        self.p = p
        self.bern = Bernoulli(p)

    # ---------------------------------------------------
    # Genera una sola realización
    # ---------------------------------------------------
    def _next(self):
        total = 0
        for _ in range(self.n):
            total += self.bern._next()
        return total

    # ---------------------------------------------------
    # Genera muestra
    # ---------------------------------------------------
    def sample(self, size):
        return [self._next() for _ in range(size)]

    # ---------------------------------------------------
    # Bar plot (discreta)
    # ---------------------------------------------------
    def barplot(self, size):
        data = self.sample(size)

        valores = list(range(self.n + 1))
        conteos = [data.count(v) / size for v in valores]  # 🔥 frecuencia relativa

        plt.figure()
        plt.bar(valores, conteos)
        plt.title(f"Bar plot Binomial(n={self.n}, p={self.p})")
        plt.xlabel("k")
        plt.ylabel("Probabilidad empírica")
        plt.show()