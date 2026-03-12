import torch
from math import sqrt

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

        
        
        
        

        
        