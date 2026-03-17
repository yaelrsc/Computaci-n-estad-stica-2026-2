import torch
from math import sqrt
import math
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MonteCarloIntegration:

    def __init__(self, function):

        self.function = function
        self.Z_norm = torch.distributions.Normal(0, 1)

    def compute_integral(self, a, b, n=100, alpha=0.05):

        dist = torch.distributions.Uniform(a, b)

        unif_sample = dist.sample((n,)).to(device)

        alpha_t = torch.tensor([1-alpha/2], device=device)

        z_alpha = self.Z_norm.icdf(alpha_t).item()

        values = self.function(unif_sample)

        integral = (b-a) * values.mean().item()

        se = (b-a) * values.std().item() / sqrt(n)

        conf_int = (integral - z_alpha*se, integral + z_alpha*se)

        print(
            "integral: {:.4f}, Confidence Interval: ({:.4f}, {:.4f}), Var: {:.4f}"
                .format(integral, conf_int[0], conf_int[1], se**2)
        )

        return integral, conf_int, se**2
class EmpiricalCDF:
    def __init__(self, data):
        self.data = torch.sort(data).values.to(device)
        self.n = len(data)

    def compute(self, x, alpha=0.05, print_=True):
        x = x.to(device)

        count = torch.searchsorted(self.data, x, right=True)
        Fnx = count.float() / self.n

        ep = math.sqrt(math.log(2 / alpha) / (2 * self.n))
        low = torch.clamp(Fnx - ep, 0., 1.)
        up = torch.clamp(Fnx + ep, 0., 1.)

        if print_ and Fnx.numel() == 1:
            print(f"ECDF: {Fnx.item():.4f}, Band: ({low.item():.4f}, {up.item():.4f})")
        return Fnx, (low, up)

    def plot(self, x_min, x_max, alpha=0.05, n_points=1000):
        points = torch.linspace(x_min, x_max, n_points, device=device)

        Fnx, (low, up) = self.compute(points, alpha=alpha, print_=False)

        plt.figure(figsize=(10, 6))
        plt.fill_between(points.cpu(), low.cpu(), up.cpu(), color='red', alpha=0.2, label='Banda de Confianza')
        plt.step(points.cpu(), Fnx.cpu(), where='post', label='ECDF', color='blue')

        plt.title(f"Empirical CDF con Banda de Confianza (α={alpha})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
class EmpiricalPDF:

    def __init__(self, distribution):
        self.distribution = distribution

    def compute(self, x, n_sim=1000, n_groups=100, h=None):
        if h == None:
            h = n_groups ** (-1 / 3)

        x = x.to(device)

        sample_ = self.distribution.sample((n_sim, n_groups, 1)).to(device)
        # sample_ = sample_.unsqueeze(-1)

        mask = (sample_ > x) & (sample_ <= x + h)

        fx = mask.sum(dim=(0, 1)) / (n_sim * n_groups * h)

        return fx

    def plot(self, x_min, x_max, n_points=200, n_sim=1000, n_groups=100, h=None):
        x = torch.linspace(x_min, x_max, n_points).to(device)

        # estimador empírico
        fx_est = self.compute(x, n_sim=n_sim, n_groups=n_groups, h=h)

        # densidad real
        fx_true = torch.exp(self.distribution.log_prob(x))

        # mover a CPU para graficar
        x = x.cpu()
        fx_est = fx_est.cpu()
        fx_true = fx_true.cpu()

        plt.figure(figsize=(10, 6))

        plt.plot(x, fx_true, label="True density", linewidth=2, color='red')
        plt.plot(x, fx_est, '--', label="Empirical PDF", color='blue')

        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Empirical PDF vs True PDF")

        plt.legend()
        plt.grid(alpha=0.3)

        plt.show()
class ParametricVaRCVaR:

    def __init__(self, distribution):
        self.distribution = distribution
        self.Z = torch.distributions.Normal(0, 1)

    def compute_VaR(self, n=1000, alpha=0.05, beta=0.05):
        sample_ = self.distribution.sample((n,)).to(device)
        VaR_hat = torch.quantile(sample_, alpha)

        beta2 = torch.Tensor([1 - beta / 2]).to(device)
        z_beta2 = self.Z.icdf(beta2)

        se = math.sqrt(alpha * (1 - alpha)) / (self.distribution.log_prob(VaR_hat).exp() * math.sqrt(n))

        low = VaR_hat - z_beta2 * se
        up = VaR_hat + z_beta2 * se

        conf_int = (low, up)

        print("VaR : {:.4f},\nConf. Int: ({:.4f}, {:.4f})".format(VaR_hat.item(), low.item(), up.item()))

        return VaR_hat, conf_int

    def compute_CVaR(self, n=1000, alpha=0.05, beta=0.05):
        sample_ = self.distribution.sample((n,)).to(device)
        VaR_hat = torch.quantile(sample_, alpha)

        CVaR_hat = torch.where(sample_ <= VaR_hat, sample_, 0.0).mean() / alpha

        se = (1 / alpha) * torch.where(sample_ <= VaR_hat, sample_, 0.0).std() / math.sqrt(n)

        beta2 = torch.Tensor([1 - beta / 2]).to(device)
        z_beta2 = self.Z.icdf(beta2)

        low = CVaR_hat - z_beta2 * se
        up = CVaR_hat + z_beta2 * se

        conf_int = (low, up)

        print("CVaR : {:.4f},\nConf. Int: ({:.4f}, {:.4f})".format(CVaR_hat.item(), low.item(), up.item()))

        return CVaR_hat, conf_int

        
        
        
        

        
        