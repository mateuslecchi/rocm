import torch

size = (3, 3)  # ou o valor que você estiver usando
device = "hip" if torch.backends.hip.is_built() else "cpu"

print(f"Usando dispositivo: {device}")
a = torch.rand(size, device=device)
print(a)
