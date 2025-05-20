import torch

print(torch.version)
print(hasattr(torch.backends, "cuda"))  # Deve ser True
print(torch.version.hip)               # Deve mostrar algo como '6.3.0'
print(torch.cuda.get_device_name(0))   # Deve mostrar sua GPU AMD
