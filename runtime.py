"""Load the released inference weights and fixed diffusion schedule."""
from pathlib import Path
import torch
from models.lstm import LSTMB_cum
from models.cfg_diffusion import ConditionalDiffusionNet

ROOT = Path(__file__).resolve().parent


def load_models(device):
    surrogate = LSTMB_cum(output_size=3, input_size=6, hidden_size=64, num_layers=5).to(device)
    diffusion = ConditionalDiffusionNet(dropout=0.05).to(device)
    for name, model in [('surrogate', surrogate), ('diffusion', diffusion)]:
        model.load_state_dict(torch.load(ROOT / 'checkpoints' / f'{name}.pt',
                                         map_location=device, weights_only=True))
        model.eval()
    betas = torch.linspace(1e-4, 0.02, 500, device=device)
    alphas = 1 - betas
    cumulative = torch.cumprod(alphas, dim=0)
    previous = torch.cat([torch.ones(1, device=device), cumulative[:-1]])
    schedule = {
        'betas': betas,
        'sqrt_one_minus_alpha_cumprod': torch.sqrt(1 - cumulative),
        'sqrt_recip_alphas': torch.sqrt(1 / alphas),
        'posterior_variance': betas * (1 - previous) / (1 - cumulative),
    }
    return surrogate, diffusion, schedule
