import torch
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math
from scipy.stats import norm
from tqdm import tqdm 

class PoissonProcess:
    """
    Clase base para un Proceso de Poisson.

    Este objeto representa un proceso de conteo {N(t), t ≥ 0}
    con intensidad λ (lambda).

    Parámetros
    ----------
    lam : float
        Tasa (intensidad) del proceso de Poisson.
        Representa el número esperado de eventos por unidad de tiempo.

    Notas
    -----
    Esta es una clase abstracta.
    El método `simulate` debe implementarse en una clase hija.
    """

    def __init__(self, lam):
        """
        Constructor del proceso.

        Parámetros
        ----------
        lam : float
            Intensidad λ del proceso.
        """
        self.lam = lam  # Guardamos la tasa del proceso


    def simulate(self, t):
        """
        Simula una trayectoria del proceso hasta tiempo t.

        Parámetros
        ----------
        t : float
            Horizonte temporal de simulación.

        Returns
        -------
        S : array-like
            Tiempos de llegada de los eventos.
        N : array-like
            Valores del proceso de conteo en esos tiempos.

        Raises
        ------
        NotImplementedError
            Este método debe implementarse en una clase hija.
        """
        raise NotImplementedError(
            "Debes implementar el metodo en clase hija"
        )


    def plot(self, t, color="blue", linestyle="-", marker=None):
        """
        Grafica una trayectoria simulada del proceso.

        La gráfica es tipo escalón (step function), 
        característica de procesos de conteo.

        Parámetros
        ----------
        t : float
            Horizonte temporal.
        color : str, opcional
            Color de la gráfica.
        linestyle : str, opcional
            Estilo de línea.
        marker : str o None, opcional
            Marcador para los puntos.
        """

        # Simulamos una trayectoria hasta tiempo t
        S, N = self.simulate(t)

        # Creamos nueva figura
        plt.figure()

        # Graficamos usando función escalón.
        # where="post" indica que el salto ocurre después del evento,
        # lo cual es consistente con procesos de conteo.
        plt.step(
            S,
            N,
            where="post",
            color=color,
            linestyle=linestyle,
            marker=marker
        )

        # Etiquetas de los ejes
        plt.xlabel("t")       # Tiempo
        plt.ylabel("N(t)")    # Conteo acumulado

        # Título con el valor de lambda
        plt.title(f"Proceso de Poisson (λ={self.lam})")

        # Mostramos la figura
        plt.show()
    
        
    def monte_carlo_lambda(self, t, n_sim=10000, alpha=0.05):
        
    
        """
        Estima la intensidad λ de un Proceso de Poisson homogéneo
        mediante simulación Monte Carlo y grafica su convergencia
        junto con intervalos de confianza asintóticos.
    
        Fundamentación teórica
        -----------------------
        Para un proceso de Poisson homogéneo:
    
            N(t) ~ Poisson(λ t)
    
        Un estimador natural de λ es:
    
            λ̂ = N(t) / t
    
        Si simulamos n trayectorias independientes:
    
            λ̂_n = (1 / (n t)) Σ_{i=1}^n N_i(t)
    
        entonces:
    
            E[λ̂_n] = λ
            Var(λ̂_n) = λ / (n t)
    
        Usando el Teorema Central del Límite:
    
            λ̂_n ≈ Normal(λ, λ/(n t))
    
        Intervalo de confianza aproximado:
    
            λ̂_n ± z_{1-α/2} sqrt(λ̂_n / (n t))
    
        Parámetros
        ----------
        t : float
            Horizonte temporal.
    
        n_sim : int, opcional
            Número de simulaciones.
    
        alpha : float, opcional
            Nivel de significancia.
    
        Returns
        -------
        lam_hat : torch.Tensor
            Estimador acumulado de λ.
    
        lower : torch.Tensor
            Límite inferior del IC.
    
        upper : torch.Tensor
            Límite superior del IC.
        """

    
        lam_hat = torch.zeros(n_sim)
        running_sum = 0
    
        for i in tqdm(range(n_sim), desc="Simulando trayectorias"):
    
            S, N = self.simulate(t)
    
            N_t = N[-1] if len(N) > 0 else 0
    
            running_sum += N_t
            lam_hat[i] = running_sum / ((i + 1) * t)
    
        n_vals = torch.arange(1, n_sim + 1)
        z = norm.ppf(1 - alpha / 2)
    
        se = torch.sqrt(lam_hat / (n_vals * t))
    
        lower = lam_hat - z * se
        upper = lam_hat + z * se
    
        plt.figure(figsize=(8, 5))
    
        x = n_vals
        y = lam_hat
    
        plt.plot(x, y, label="Estimador Monte Carlo", color="red")
    
        plt.fill_between(
            x,
            lower,
            upper,
            alpha=0.3,
            label=f"IC {(1-alpha)*100:.0f}%"
        )
    
        plt.axhline(self.lam, linestyle="--",
                    label="Valor verdadero λ")
    
        plt.xlabel("Número de simulaciones")
        plt.ylabel("Estimación de λ")
        plt.title("Convergencia Monte Carlo del estimador de λ")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
    
        return lam_hat, lower, upper

class PoissonProcess1(PoissonProcess):
    """
    Proceso de Poisson homogéneo.

    Esta clase implementa la simulación de un proceso de Poisson
    con intensidad constante λ utilizando el método de tiempos
    entre llegadas exponenciales.

    Recordatorio teórico
    --------------------
    Si {N(t)} es un proceso de Poisson con tasa λ, entonces:

        - Los tiempos entre llegadas son iid ~ Exp(λ)
        - N(t) cuenta el número de eventos ocurridos hasta tiempo t
    """

    def __init__(self, lam):
        """
        Constructor del proceso de Poisson homogéneo.

        Parámetros
        ----------
        lam : float
            Intensidad λ del proceso (eventos por unidad de tiempo).
        """
        super().__init__(lam)

        # Distribución exponencial para los tiempos entre llegadas
        # Si T_i ~ Exp(λ), entonces E[T_i] = 1/λ
        self.expo = torch.distributions.Exponential(self.lam)


    def simulate(self, t):
        """
        Simula una trayectoria del proceso hasta tiempo t.

        Parámetros
        ----------
        t : float
            Horizonte temporal de simulación.

        Returns
        -------
        S : torch.Tensor
            Vector de tiempos incluyendo:
                0 (inicio),
                tiempos de llegada,
                t (final del horizonte).
        N : torch.Tensor
            Valores del proceso de conteo en los tiempos S.

        Notas
        -----
        Se usa el método clásico:
            - Simular tiempos entre llegadas exponenciales
            - Acumular hasta exceder t
        """

        time = 0          # Tiempo actual acumulado
        S = []            # Lista para guardar tiempos de llegada

        while True:

            # Generamos un tiempo entre llegadas T ~ Exp(λ)
            T = self.expo.sample().item()

            # Actualizamos el tiempo acumulado
            time += T

            # Si excede el horizonte temporal, detenemos
            if time > t:
                break

            # Guardamos el tiempo de llegada válido
            S.append(time)

        # Convertimos lista de tiempos a tensor
        S = torch.tensor(S)

        # Número total de eventos ocurridos hasta tiempo t
        n = len(S)

        # Construimos el proceso de conteo:
        # N(t) toma valores 0,1,2,...,n
        N = torch.arange(0, n + 1)

        # Agregamos el valor final para mantener consistencia
        N = torch.cat((N, torch.tensor([n])))

        # Agregamos tiempo inicial 0 y tiempo final t
        # Esto permite graficar correctamente como función escalón
        S = torch.cat((torch.tensor([0.0]), S, torch.tensor([t])))

        return S, N

class PoissonProcess2(PoissonProcess):
    """
    Proceso de Poisson homogéneo simulado mediante discretización temporal.

    Este método utiliza la propiedad de incrementos independientes:

        N(t_{i}) - N(t_{i-1}) ~ Poisson(λ Δt)

    donde Δt es el tamaño de cada subintervalo.

    A diferencia de la clase PoissonProcess1, aquí no se simulan
    tiempos entre llegadas, sino directamente los incrementos
    del proceso en una malla uniforme.
    """

    def simulate(self, t, n):
        """
        Simula una trayectoria del proceso hasta tiempo t
        usando discretización en n intervalos.

        Parámetros
        ----------
        t : float
            Horizonte temporal.
        n : int
            Número de subdivisiones del intervalo [0, t].

        Returns
        -------
        S : torch.Tensor
            Malla uniforme de tiempos.
        N : torch.Tensor
            Valores del proceso en cada punto de la malla.

        Notas
        -----
        Se usa que:

            N(t_i) - N(t_{i-1}) ~ Poisson(λ Δt)

        con Δt = t / n.
        """

        # Construimos malla uniforme:
        # 0 = t0 < t1 < ... < tn = t
        S = torch.linspace(0, t, n + 1)

        # Inicializamos vector del proceso
        # N[0] = 0 (condición inicial)
        N = torch.zeros(n + 1)

        # Longitud de cada intervalo
        delta = t / n  

        # Simulación de incrementos independientes
        for i in range(1, n + 1):

            # Incremento en el intervalo [t_{i-1}, t_i]
            # ~ Poisson(λ Δt)
            X = torch.distributions.Poisson(self.lam * delta).sample()
            
            # Propiedad fundamental:
            # N(t_i) = N(t_{i-1}) + incremento independiente
            N[i] = N[i - 1] + X

        return S, N


    def plot(self, t, n, color="blue", linestyle="-", marker=None):
        """
        Grafica una trayectoria simulada usando discretización.

        Parámetros
        ----------
        t : float
            Horizonte temporal.
        n : int
            Número de subdivisiones.
        color : str, opcional
            Color de la gráfica.
        linestyle : str, opcional
            Estilo de línea.
        marker : str o None, opcional
            Marcador en los puntos.
        """

        # Simulamos trayectoria en la malla
        S, N = self.simulate(t, n)

        # Creamos figura
        plt.figure()

        # Gráfica tipo escalón
        # where="post" mantiene el valor constante hasta el siguiente punto
        plt.step(
            S,
            N,
            where="post",
            color=color,
            linestyle=linestyle,
            marker=marker
        )

        # Etiquetas
        plt.xlabel("t")
        plt.ylabel("N(t)")

        # Título con intensidad
        plt.title(f"Proceso de Poisson (λ={self.lam})")

        # Mostrar gráfica
        plt.show()

class PoissonProcess3(PoissonProcess):
    """
    Proceso de Poisson homogéneo usando la propiedad de
    ordenación condicional.

    Este método utiliza el siguiente resultado fundamental:

        Condicionalmente a N(t) = n,
        los tiempos de llegada tienen la misma distribución
        que los estadísticos de orden de n variables Uniforme(0, t).

    Es decir:

        1) Se simula primero N(t) ~ Poisson(λ t)
        2) Luego se generan n variables Uniforme(0,t)
        3) Se ordenan para obtener los tiempos de llegada

    Este método es exacto y completamente equivalente
    al método basado en tiempos entre llegadas exponenciales.
    """

    def simulate(self, t):
        """
        Simula una trayectoria del proceso hasta tiempo t.

        Parámetros
        ----------
        t : float
            Horizonte temporal.

        Returns
        -------
        S : torch.Tensor
            Vector de tiempos incluyendo:
                0 (inicio),
                tiempos ordenados de llegada,
                t (final del horizonte).
        N : torch.Tensor
            Valores del proceso de conteo en esos tiempos.

        Fundamento teórico
        ------------------
        Si {N(t)} es un proceso de Poisson con tasa λ, entonces:

            N(t) ~ Poisson(λ t)

        y condicionalmente a N(t)=n:

            (S1,...,Sn) ≍ ordenados de Uniforme(0,t)
        """

        # Paso 1: número total de saltos
        # N(t) ~ Poisson(λ t)
        n = int(
            torch.distributions.Poisson(self.lam * t)
            .sample()
            .item()
        )

        # Paso 2: generar tiempos no ordenados
        # U_i ~ Uniforme(0, t)
        U = torch.rand(n) * t

        # Paso 3: ordenar los tiempos
        # Esto produce los tiempos de llegada del proceso
        S = torch.sort(U).values

        # Agregamos tiempo inicial 0 y final t
        # Esto facilita la gráfica como proceso escalonado
        S = torch.cat((torch.tensor([0.0]), S, torch.tensor([t])))

        # Construimos el proceso de conteo:
        # valores 0,1,2,...,n
        N = torch.arange(0, n + 1)

        # Repetimos el valor final para mantener constante hasta t
        N = torch.cat((N, torch.tensor([n])))

        return S, N

class NonHomogeneousPoissonProcess(PoissonProcess):
    """
    Proceso de Poisson No Homogéneo (NHPP).

    Este proceso generaliza el proceso de Poisson clásico
    permitiendo que la intensidad dependa del tiempo:

        λ = λ(t)

    Propiedad fundamental:
        Para 0 ≤ s < t,

        N(t) - N(s) ~ Poisson( ∫_s^t λ(u) du )

    Es decir, los incrementos siguen siendo independientes,
    pero su media depende de la integral de la intensidad.
    """
    
    def simulate(self, t, n):
        """
        Simula una trayectoria del proceso hasta tiempo t
        usando discretización en n intervalos.

        Parámetros
        ----------
        t : float
            Horizonte temporal.
        n : int
            Número de subdivisiones del intervalo [0, t].

        Returns
        -------
        S : torch.Tensor
            Malla uniforme de tiempos.
        N : torch.Tensor
            Valores del proceso en cada punto de la malla.

        Notas
        -----
        En cada intervalo [t_{i-1}, t_i] se usa que:

            N(t_i) - N(t_{i-1}) ~ Poisson( ∫_{t_{i-1}}^{t_i} λ(u) du )

        La integral se calcula numéricamente usando `quad`.
        """
        
        # Malla uniforme:
        # 0 = t0 < t1 < ... < tn = t
        S = torch.linspace(0, t, n + 1)

        # Inicializamos vector del proceso
        # N(0) = 0
        N = torch.zeros(n + 1)

        # Simulación de incrementos independientes
        for i in range(1, n + 1):

            # Calculamos la intensidad acumulada en el intervalo:
            # ∫_{S[i-1]}^{S[i]} λ(u) du
            lamb = quad(self.lam, S[i-1], S[i])[0]
            
            # Incremento en el intervalo
            # ~ Poisson( integral de λ )
            X = torch.distributions.Poisson(lamb).sample()
            
            # Propiedad de incrementos independientes
            N[i] = N[i - 1] + X

        return S, N
        
    def plot(self, t, n, color="blue", linestyle="-", marker=None):
        """
        Grafica una trayectoria simulada del proceso
        de Poisson no homogéneo.

        Parámetros
        ----------
        t : float
            Horizonte temporal.
        n : int
            Número de subdivisiones.
        color : str, opcional
            Color de la gráfica.
        linestyle : str, opcional
            Estilo de línea.
        marker : str o None, opcional
            Marcador.
        """

        # Simulamos trayectoria
        S, N = self.simulate(t, n)

        # Creamos figura
        plt.figure()

        # Gráfica tipo escalón
        # where="post" mantiene el valor constante
        # hasta el siguiente punto
        plt.step(
            S,
            N,
            where="post",
            color=color,
            linestyle=linestyle,
            marker=marker
        )

        # Etiquetas
        plt.xlabel("t")
        plt.ylabel("N(t)")

        # Título descriptivo
        plt.title(f"Proceso de Poisson No Homegeneo")

        # Mostrar gráfica
        plt.show()
        
    def monte_carlo_lambda(self, t, n, n_sim=10000, alpha=0.05):
        
        """
        Estima E[N(t)] para un Proceso de Poisson No Homogéneo
        mediante simulación Monte Carlo y grafica su convergencia
        con intervalos de confianza asintóticos.
    
        Fundamentación teórica
        -----------------------
        Para un NHPP con intensidad λ(t):
    
            E[N(t)] = Λ(t) = ∫₀ᵗ λ(s) ds
    
        Si simulamos n trayectorias:
    
            Λ̂_n = (1/n) Σ_{i=1}^n N_i(t)
    
        entonces:
    
            E[Λ̂_n] = Λ(t)
            Var(Λ̂_n) = Λ(t) / n
    
        Aproximación normal (CLT):
    
            Λ̂_n ≈ Normal(Λ(t), Λ(t)/n)
    
        Intervalo de confianza:
    
            Λ̂_n ± z_{1-α/2} sqrt(Λ̂_n / n)
    
        El valor verdadero Λ(t) se calcula numéricamente
        usando scipy.integrate.quad.
    
        Parámetros
        ----------
        t : float
            Horizonte temporal.
        n : int
            Número de subdivisiones.
        n_sim : int
            Número de simulaciones.
    
        alpha : float
            Nivel de significancia.
    
        Returns
        -------
        estimates : torch.Tensor
            Estimador acumulado de E[N(t)].
    
        lower : torch.Tensor
            Límite inferior del IC.
    
        upper : torch.Tensor
            Límite superior del IC.
        """
    
        true_value, _ = quad(self.lam, 0, t)
    
        estimates = torch.zeros(n_sim)
        running_sum = 0
    
        for i in tqdm(range(n_sim), desc="Simulando trayectorias"):
    
            S, N = self.simulate(t,n)
    
            N_t = N[-1] if len(N) > 0 else 0
    
            running_sum += N_t
            estimates[i] = running_sum / (i + 1)
    
        n_vals = torch.arange(1, n_sim + 1)
        z = norm.ppf(1 - alpha / 2)
    
        se = torch.sqrt(estimates / n_vals)
    
        lower = estimates - z * se
        upper = estimates + z * se
    
        plt.figure(figsize=(8, 5))
    
        x = n_vals
        y = estimates
    
        plt.plot(x, y, label="Estimador Monte Carlo", color="red")
    
        plt.fill_between(
            x,
            lower,
            upper,
            alpha=0.3,
            label=f"IC {(1-alpha)*100:.0f}%"
        )
    
        plt.axhline(true_value,
                    linestyle="--",
                    label="Valor verdadero ∫₀ᵗ λ(s)ds")
    
        plt.xlabel("Número de simulaciones")
        plt.ylabel("Estimación de E[N(t)]")
        plt.title("Convergencia Monte Carlo - NHPP")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
    
        return estimates, lower, upper

class CompoundPoisson:
    """
    Proceso de Poisson Compuesto.

    Este proceso se define como:

        X(t) = sum_{i=1}^{N(t)} Y_i

    donde:

        - N(t) es un proceso de Poisson con intensidad λ
        - {Y_i} son variables iid independientes de N(t)
          que representan los tamaños de salto

    Interpretación:
        - N(t) = número de eventos hasta tiempo t
        - Y_i = tamaño del i-ésimo salto
        - X(t) = suma acumulada de los saltos
    """

    def __init__(self, lam, jump_distribution):
        """
        Constructor del proceso compuesto.

        Parámetros
        ----------
        lam : float
            Intensidad del proceso de Poisson.
        jump_distribution : torch.distributions.Distribution
            Distribución de los tamaños de salto Y_i.
        """

        self.lam = lam

        # Proceso de Poisson base
        self.poisson = PoissonProcess3(lam)

        # Distribución de los saltos
        self.jump_dist = jump_distribution


    def simulate(self, t):
        """
        Simula una trayectoria del proceso compuesto hasta tiempo t.

        Implementa directamente:

            1) N(t) ~ Poisson(λ t)
            2) S_i = tiempos ordenados Uniform(0,t)
            3) Y_i ~ distribución de saltos
            4) X(t) = suma acumulada de Y_i

        Parámetros
        ----------
        t : float
            Horizonte temporal.

        Returns
        -------
        S : torch.Tensor
            Tiempos incluyendo 0 y t.
        X : torch.Tensor
            Valores del proceso compuesto en esos tiempos.
        """

        # Paso 1: número total de saltos
        n = int(
            torch.distributions.Poisson(self.lam * t)
            .sample()
            .item()
        )
        
        # Paso 2: tiempos de llegada (no ordenados)
        U = torch.rand(n) * t

        # Ordenamos tiempos
        S = torch.sort(U).values

        # Paso 3: tamaños de salto Y_i ~ jump_distribution
        Y = self.jump_dist.sample((n,))

        # Paso 4: suma acumulada
        # X_k = Y_1 + ... + Y_k
        X = torch.cumsum(Y, dim=0)

        # Valor final en tiempo t
        X_t = X[-1].item()

        # Agregamos tiempo inicial 0 y final t
        S = torch.cat((torch.tensor([0.0]), S, torch.Tensor([t])))

        # Agregamos valor inicial 0 y mantenemos constante hasta t
        X = torch.cat((torch.tensor([0.0]), X, torch.Tensor([X_t])))

        return S, X


    def plot(self, t, color="blue", linestyle="-", marker=None):
        """
        Grafica una trayectoria del proceso compuesto.

        Parámetros
        ----------
        t : float
            Horizonte temporal.
        color : str, opcional
            Color de la gráfica.
        linestyle : str, opcional
            Estilo de línea.
        marker : str o None, opcional
            Marcador.
        """

        # Simulamos trayectoria
        S, X = self.simulate(t)

        # Creamos figura
        plt.figure()

        # Gráfica tipo escalón (proceso puro de saltos)
        plt.step(
            S,
            X,
            where="post",
            color=color,
            linestyle=linestyle,
            marker=marker
        )

        # Etiquetas
        plt.xlabel("t")
        plt.ylabel("X(t)")

        # Título
        plt.title("Proceso de Poisson Compuesto")

        # Mostrar gráfica
        plt.show()

    def monte_carlo_lambda(self, t, n_sim=10000, alpha=0.05):
        
        """
        Estima la intensidad λ de un Proceso de Poisson Compuesto
        mediante simulación Monte Carlo y grafica su convergencia
        con intervalos de confianza asintóticos.
    
        Fundamentación teórica
        -----------------------
        Para el proceso compuesto:
    
            X(t) = sum_{i=1}^{N(t)} Y_i
    
        donde N(t) ~ Poisson(λ t) y Y_i son iid independientes.
    
        Se sabe que:
    
            E[X(t)] = λ t E[Y]
    
        Por lo tanto, un estimador natural de λ es:
    
            λ̂ = X̄_n(t) / (t E[Y])
    
        donde:
    
            X̄_n(t) = (1/n) Σ X_i(t)
    
        Además:
    
            Var(X(t)) = λ t E[Y²]
    
        Usando CLT y plug-in, un intervalo de confianza aproximado
        de nivel (1 - α) es:
    
            λ̂ ± z_{1-α/2}
            sqrt( λ̂ E[Y²] / (n t (E[Y])²) )
    
        Parámetros
        ----------
        t : float
            Horizonte temporal.
    
        n_sim : int
            Número de trayectorias simuladas.
    
        alpha : float
            Nivel de significancia.
    
        Returns
        -------
        lam_hat : torch.Tensor
            Estimador acumulado de λ.
    
        lower : torch.Tensor
            Límite inferior del IC.
    
        upper : torch.Tensor
            Límite superior del IC.
        """
    
        
    
        # Momentos teóricos de Y
        EY = self.jump_dist.mean
        EY2 = self.jump_dist.variance + EY**2
    
        lam_hat = torch.zeros(n_sim)
        running_sum = 0
    
        for i in tqdm(range(n_sim), desc="Simulando trayectorias"):
    
            S, X = self.simulate(t)
    
            X_t = X[-1] if len(X) > 0 else 0.0
    
            running_sum += X_t
    
            mean_X = running_sum / (i + 1)
    
            lam_hat[i] = mean_X / (t * EY)
    
        n_vals = torch.arange(1, n_sim + 1)
        z = norm.ppf(1 - alpha / 2)
    
        # Error estándar asintótico (plug-in)
        se = torch.sqrt(
            lam_hat * EY2 / (n_vals * t * (EY**2))
        )
    
        lower = lam_hat - z * se
        upper = lam_hat + z * se
    
        # --- Gráfica ---
        plt.figure(figsize=(8, 5))
    
        x = n_vals
        y = lam_hat
    
        plt.plot(x, y, label="Estimador Monte Carlo", color="red")
    
        plt.fill_between(
            x,
            lower,
            upper,
            alpha=0.3,
            label=f"IC {(1-alpha)*100:.0f}%"
        )
    
        plt.axhline(self.lam,
                    linestyle="--",
                    label="Valor verdadero λ")
    
        plt.xlabel("Número de simulaciones")
        plt.ylabel("Estimación de λ")
        plt.title("Convergencia Monte Carlo - Proceso Compuesto")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
    
        return lam_hat, lower, upper


class CramerLundberg:
    """
    Modelo clásico de Cramér–Lundberg.

    El capital de la aseguradora evoluciona como:

        U(t) = u + c t - X(t)

    donde:

        u  : capital inicial
        c  : tasa de prima (ingreso lineal en el tiempo)
        X(t) : proceso de Poisson compuesto que representa
               el total acumulado de reclamaciones

    Interpretación:
        - El capital crece linealmente entre reclamaciones.
        - En cada reclamación hay un salto negativo.
        - Ocurre ruina si U(t) < 0 en algún momento.
    """

    def __init__(self, u, c, lam, claim_distribution):
        """
        Constructor del modelo.

        Parámetros
        ----------
        u : float
            Capital inicial (u ≥ 0).

        c : float
            Tasa de prima (c > 0).

        lam : float
            Intensidad del proceso de Poisson (λ > 0),
            número esperado de reclamaciones por unidad de tiempo.

        claim_distribution : torch.distributions.Distribution
            Distribución de los tamaños de reclamación Y_i.
            Debe permitir el método `.sample()`.

        Notas
        -----
        El proceso de reclamaciones se modela como:

            X(t) = sum_{i=1}^{N(t)} Y_i

        donde N(t) ~ Poisson(λt).
        """

        self.u = u
        self.c = c
        self.lam = lam
        self.claim_distribution = claim_distribution

        # Proceso de Poisson compuesto
        self.compound = CompoundPoisson(
            self.lam,
            self.claim_distribution
        )


    def simulate(self, t):
        """
        Simula la trayectoria del capital hasta tiempo t.

        Parámetros
        ----------
        t : float
            Horizonte temporal (t > 0).

        Returns
        -------
        S : torch.Tensor
            Tiempos relevantes (incluye 0 y t).

        U : torch.Tensor
            Valores del capital en los tiempos S.

        ruined : bool
            True si ocurre ruina (U(t) < 0 en algún momento),
            False en caso contrario.
        """

        # Simulación del proceso compuesto de reclamaciones
        S, X = self.compound.simulate(t)

        # Capital
        U = self.u + self.c * S - X

        # Indicador de ruina
        ruined = torch.any(U < 0).item()

        return S, U, ruined


    def ruin_probability(self, T, n, alpha=0.05):
        """
        Estima la probabilidad de ruina en horizonte finito
        usando simulación Monte Carlo.

        Parámetros
        ----------
        T : float
            Horizonte temporal.

        n : int
            Número de simulaciones.

        alpha : float, optional
            Nivel de significancia (default 0.05 → 95%).

        Returns
        -------
        p_hat : float
            Estimador Monte Carlo de la probabilidad de ruina.

        lower : float
            Límite inferior del intervalo de confianza.

        upper : float
            Límite superior del intervalo de confianza.

        Método
        ------
        Sea I_i = 1{ruina en simulación i}. Entonces:

            p_hat = (1/n) ∑ I_i

        Por el Teorema Central del Límite:

            p_hat ≈ Normal(p, p(1-p)/n)

        Intervalo de confianza:

            p_hat ± z_{1-α/2} sqrt(p_hat(1-p_hat)/n)
        """

        ruin_count = 0

        # Monte Carlo
        for _ in range(n):
            ruined = self.simulate(T)[2]
            ruin_count += ruined

        p_hat = ruin_count / n

        # Cuantil normal dinámico
        z = norm.ppf(1 - alpha / 2)

        # Error estándar
        se = math.sqrt(p_hat * (1 - p_hat) / n)

        lower = p_hat - z * se
        upper = p_hat + z * se

        return p_hat, lower, upper

    def ruin_probability_mc(self, T, n_sim=1000, alpha=0.05):
        """
        Estima la probabilidad de ruina mediante simulación Monte Carlo.
    
        La probabilidad de ruina se define como:
    
            ψ(u, T) = P( τ ≤ T )
    
        donde:
            τ = inf{ t ≥ 0 : U(t) < 0 }
    
        Se estima usando:
    
            ψ̂_n = (1/n) ∑ I{ruina}
    
        y se construye un intervalo de confianza tipo CLT:
    
            ψ̂ ± z_{1-α/2} sqrt( ψ̂(1-ψ̂) / n )
    
        Parámetros
        ----------
        T : float
            Horizonte temporal.
        n_sim : int
            Número de simulaciones Monte Carlo.
        alpha : float
            Nivel de significancia (default 0.05 para IC 95%).
    
        Returns
        -------
        psi_hat : float
            Estimador final de la probabilidad de ruina.
        lower : float
            Límite inferior del intervalo de confianza.
        upper : float
            Límite superior del intervalo de confianza.
        """
    
        estimates = torch.zeros(n_sim)
        running_sum = 0.0
    
        # Simulación Monte Carlo
        for i in tqdm(range(n_sim)):
    
            # Se asume que simulate(T) regresa (..., ruined)
            ruined = self.simulate(T)[2]
    
            running_sum += ruined
            p_hat = running_sum / (i + 1)
    
            estimates[i] = p_hat
    
        # Intervalo tipo CLT dinámico
        z = norm.ppf(1 - alpha/2)
    
        n_vals = torch.arange(1, n_sim + 1)
        se = torch.sqrt(estimates * (1 - estimates) / n_vals)
    
        lower_path = estimates - z * se
        upper_path = estimates + z * se
    
        # ---- Gráfica de convergencia ----
        plt.figure()
    
        plt.plot(estimates, color='red', label="Estimador Monte Carlo")
    
        plt.fill_between(
            n_vals,
            lower_path,
            upper_path,
            alpha=0.3,
            label=f"IC {(1-alpha)*100:.0f}%"
        )
    
        plt.xlabel("Número de simulaciones")
        plt.ylabel("Probabilidad de ruina estimada")
        plt.title("Convergencia Monte Carlo - Probabilidad de Ruina")
        plt.legend()
        plt.show()
    
        # Valores finales del intervalo
        psi_hat = estimates[-1].item()
        lower = lower_path[-1].item()
        upper = upper_path[-1].item()
    
        return psi_hat, lower, upper

    def ruin_vs_u(self, u_values, T, n_sim=1000, alpha=0.05):
        """
        Estima la probabilidad de ruina como función del capital inicial u.

        Este método realiza simulaciones Monte Carlo del modelo
        de Cramér–Lundberg para distintos valores del capital inicial
        y grafica cómo cambia la probabilidad de ruina.

        Modelo
        ------
        El proceso de capital se define como

            U(t) = u + c t - X(t)

        donde

            X(t) = sum_{i=1}^{N(t)} Y_i

        es un proceso de Poisson compuesto.

        La ruina ocurre si

            U(t) < 0

        en algún momento antes del horizonte T.

        Para cada valor de u se estima

            ψ(u) = P(ruina antes de T)

        mediante simulación Monte Carlo.

        Parámetros
        ----------
        u_values : array-like
            Vector de valores de capital inicial a evaluar.

        T : float
            Horizonte temporal de simulación.

        n_sim : int, opcional (default=1000)
            Número de trayectorias simuladas para cada valor de u.

        alpha : float, opcional (default=0.05)
            Nivel de significancia para los intervalos de confianza
            basados en la aproximación normal.

        Notas
        -----
        - La probabilidad de ruina es una función decreciente de u.
        - Para capital inicial grande, la probabilidad de ruina
          tiende a cero.
        - El intervalo de confianza usado es

            p̂ ± z_{1-α/2} sqrt( p̂(1-p̂) / n )

          donde p̂ es el estimador Monte Carlo.
        """

        probs = torch.zeros(len(u_values))
        lower = torch.zeros(len(u_values))
        upper = torch.zeros(len(u_values))

        z = norm.ppf(1 - alpha / 2)

        for i, u in enumerate(tqdm(u_values, desc="Variando capital inicial u")):

            # Actualizamos capital inicial
            self.u = float(u)

            ruin_count = 0

            # Simulaciones Monte Carlo
            for _ in range(n_sim):
                ruined = self.simulate(T)[2]
                ruin_count += ruined

            p_hat = ruin_count / n_sim

            # Guardamos estimador
            probs[i] = p_hat

            # Error estándar
            se = math.sqrt(p_hat * (1 - p_hat) / n_sim)

            lower[i] = p_hat - z * se
            upper[i] = p_hat + z * se

        # --------- Gráfica ---------

        plt.figure()

        plt.plot(u_values, probs, label="Estimador Monte Carlo")

        plt.fill_between(
            u_values,
            lower,
            upper,
            alpha=0.3,
            label=f"IC {(1 - alpha) * 100:.0f}%"
        )

        plt.xlabel("Capital inicial u")
        plt.ylabel("Probabilidad de ruina")
        plt.title("Probabilidad de ruina vs capital inicial")
        plt.grid(alpha=0.3)
        plt.legend()

        plt.show()

    def ruin_vs_T(self, T_values, n_sim=1000, alpha=0.05):
        """
        Estima la probabilidad de ruina como función del horizonte temporal T.

        Este método evalúa cómo cambia la probabilidad de ruina cuando
        se incrementa el tiempo de observación del proceso de capital
        en el modelo de Cramér–Lundberg.

        Modelo
        ------
        El capital evoluciona como

            U(t) = u + c t - X(t)

        donde

            X(t) = sum_{i=1}^{N(t)} Y_i

        es un proceso de Poisson compuesto.

        La ruina ocurre si

            U(t) < 0

        en algún momento antes del tiempo T.

        Para cada valor de T se estima

            ψ(T) = P(ruina antes de T)

        usando simulación Monte Carlo.

        Parámetros
        ----------
        T_values : array-like
            Vector de horizontes temporales a evaluar.

        n_sim : int, opcional (default=1000)
            Número de trayectorias simuladas para cada valor de T.

        alpha : float, opcional (default=0.05)
            Nivel de significancia para el intervalo de confianza.



        Notas
        -----
        - La probabilidad de ruina es **creciente en T**.
        - Cuando T → ∞ se aproxima a la probabilidad de ruina última.
        - El intervalo de confianza usado es

            p̂ ± z_{1-α/2} sqrt(p̂(1-p̂)/n)

          donde p̂ es el estimador Monte Carlo.
        """

        probs = torch.zeros(len(T_values))
        lower = torch.zeros(len(T_values))
        upper = torch.zeros(len(T_values))

        z = norm.ppf(1 - alpha / 2)

        for i, T in enumerate(tqdm(T_values, desc="Variando horizonte T")):

            ruin_count = 0

            for _ in range(n_sim):
                ruined = self.simulate(float(T))[2]
                ruin_count += ruined

            p_hat = ruin_count / n_sim

            probs[i] = p_hat

            se = math.sqrt(p_hat * (1 - p_hat) / n_sim)

            lower[i] = p_hat - z * se
            upper[i] = p_hat + z * se

        # --------- Gráfica ---------

        plt.figure()

        plt.plot(T_values, probs, label="Estimador Monte Carlo")

        plt.fill_between(
            T_values,
            lower,
            upper,
            alpha=0.3,
            label=f"IC {(1 - alpha) * 100:.0f}%"
        )

        plt.xlabel("Horizonte temporal T")
        plt.ylabel("Probabilidad de ruina")
        plt.title("Probabilidad de ruina vs horizonte temporal")
        plt.grid(alpha=0.3)
        plt.legend()

        plt.show()

    def ruin_vs_c(self, c_values, T, n_sim=1000, alpha=0.05):
        """
        Estima la probabilidad de ruina como función de la tasa de prima c.

        En el modelo de Cramér–Lundberg el capital evoluciona como

            U(t) = u + c t - X(t)

        donde

            X(t) = sum_{i=1}^{N(t)} Y_i

        es un proceso de Poisson compuesto.

        La condición fundamental del modelo es la **condición de beneficio neto**

            c > λ E[Y]

        Si esta condición no se cumple, la aseguradora pierde dinero
        en promedio y la ruina es prácticamente inevitable en el largo plazo.

        Este método estima mediante Monte Carlo la probabilidad de ruina
        para distintos valores de la tasa de prima `c`.

        Parámetros
        ----------
        c_values : array-like
            Valores de la tasa de prima que se desean evaluar.

        T : float
            Horizonte temporal de observación.

        n_sim : int, opcional (default=1000)
            Número de simulaciones Monte Carlo para cada valor de c.

        alpha : float, opcional (default=0.05)
            Nivel de significancia para los intervalos de confianza.


        Notas
        -----
        La probabilidad de ruina se estima como

            p̂ = (1/n) ∑ I_i

        donde

            I_i = 1{ruina en simulación i}

        Usando el TCL, un intervalo de confianza aproximado es

            p̂ ± z_{1-α/2} sqrt(p̂(1-p̂)/n)

        En la gráfica se muestra también el punto crítico

            c* = λ E[Y]

        donde ocurre la transición entre pérdida esperada
        y beneficio esperado del modelo.
        """

        probs = torch.zeros(len(c_values))
        lower = torch.zeros(len(c_values))
        upper = torch.zeros(len(c_values))

        z = norm.ppf(1 - alpha / 2)

        # esperanza del tamaño de reclamación
        EY = self.claim_distribution.mean.item()

        # punto crítico
        c_star = self.lam * EY

        for i, c in enumerate(tqdm(c_values, desc="Variando c")):

            self.c = float(c)

            ruin_count = 0

            for _ in range(n_sim):
                ruined = self.simulate(T)[2]
                ruin_count += ruined

            p_hat = ruin_count / n_sim

            probs[i] = p_hat

            se = math.sqrt(p_hat * (1 - p_hat) / n_sim)

            lower[i] = p_hat - z * se
            upper[i] = p_hat + z * se

        # --------- Gráfica ---------

        plt.figure()

        plt.plot(c_values, probs, label="Estimador Monte Carlo")

        plt.fill_between(
            c_values,
            lower,
            upper,
            alpha=0.3,
            label=f"IC {(1 - alpha) * 100:.0f}%"
        )

        # línea vertical en el punto crítico
        plt.axvline(
            c_star,
            linestyle="--",
            color="red",
            label=r"$c=\lambda E[Y]$"
        )

        plt.xlabel("Tasa de prima c")
        plt.ylabel("Probabilidad de ruina")
        plt.title("Probabilidad de ruina vs tasa de prima")
        plt.grid(alpha=0.3)
        plt.legend()

        plt.show()


    def plot(self, t, color="blue", linestyle="-", marker=None):
        """
        Grafica una trayectoria del capital U(t).

        Parámetros
        ----------
        t : float
            Horizonte temporal hasta el cual se simula.

        color : str, optional
            Color de la línea (default="blue").

        linestyle : str, optional
            Estilo de línea (default="-").
            Ejemplos:
                "-"  : línea continua
                "--" : línea discontinua
                "-." : punto-guion
                ":"  : punteada

        marker : str or None, optional
            Marcador de puntos (default=None).
            Ejemplos:
                "o", "s", "^", "*", etc.

        Returns
        -------
        None
            Solo genera la gráfica.
        """

        S, U = self.simulate(t)[:2]

        plt.figure()

        # Dibujar cada tramo entre saltos
        for i in range(len(S) - 1):

            # Tramo lineal creciente
            t_vals = torch.linspace(S[i], S[i+1], 2)
            u_vals = U[i] + self.c * (t_vals - S[i])

            plt.plot(
                t_vals,
                u_vals,
                color=color,
                linestyle=linestyle,
                marker=marker
            )

            # Salto vertical (si no es el último)
            if i < len(S) - 2:
                plt.plot(
                    [S[i+1], S[i+1]],
                    [u_vals[-1], U[i+1]],
                    color=color,
                    linestyle=linestyle,
                    marker=marker
                )

        plt.xlabel("t")
        plt.ylabel("U(t)")
        plt.title("Modelo de Cramér–Lundberg")
        plt.show()

class MixtureDistribution(torch.distributions.Distribution):
    """
    Distribución mezcla finita.

    Sea K el número de componentes. La densidad de la mezcla es:

        f(x) = sum_{k=1}^K w_k f_k(x)

    donde:

        w_k ≥ 0,  sum w_k = 1
        f_k(x) = densidad del componente k

    La simulación se realiza en dos pasos:
        1) Seleccionar un índice Z ~ Categorical(w)
        2) Generar X ~ f_Z
    """

    def __init__(self, weights, densities):
        """
        Constructor.

        Parámetros
        ----------
        weights : torch.Tensor
            Vector de pesos (w_1, ..., w_K).
            Deben ser no negativos y sumar 1.

        densities : list
            Lista de objetos torch.distributions.Distribution.
            Cada elemento corresponde a un componente f_k.

        Notas
        -----
        Si Z ~ Categorical(weights), entonces

            X | Z = k ~ densities[k]

        y la distribución marginal de X es la mezcla.
        """

        super().__init__(validate_args=False)

        # Validaciones básicas
        if not torch.isclose(weights.sum(), torch.tensor(1.0)):
            raise ValueError("Los pesos deben sumar 1.")

        if len(weights) != len(densities):
            raise ValueError("Número de pesos y densidades debe coincidir.")

        self.weights = weights
        self.densities = densities
        self.num_components = len(densities)

        # Distribución categórica para seleccionar componente
        self.Ind = torch.distributions.Categorical(probs=self.weights)


    def sample(self, sample_shape=torch.Size()):
        """
        Genera muestras de la distribución mezcla.

        Parámetros
        ----------
        sample_shape : torch.Size or tuple, optional
            Forma deseada de las muestras.

        Returns
        -------
        samples : torch.Tensor
            Tensor con muestras de la mezcla.

        Algoritmo
        ---------
        1) Generar índices Z_i ~ Categorical(w)
        2) Para cada componente k:
               Generar tantas muestras como veces aparezca k
        3) Reordenar y devolver
        """

        # Número total de muestras
        n = torch.Size(sample_shape).numel() if sample_shape else 1

        # Selección de componente
        indices = self.Ind.sample((n,))

        samples = torch.empty(n)

        # Generación por componente
        for k in range(self.num_components):
            mask = (indices == k)
            count = mask.sum()

            if count > 0:
                samples[mask] = self.densities[k].sample((count,))

        return samples.reshape(sample_shape)