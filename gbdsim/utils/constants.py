import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ACTIVATION_FUNCTIONS = [
    torch.nn.Tanh(),
    torch.nn.LeakyReLU(),
    torch.nn.ELU(),
    torch.nn.Identity(),
]
